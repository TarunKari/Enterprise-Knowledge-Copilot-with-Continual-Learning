"""Phase 2 – Embedding service + PostgreSQL/pgvector storage with HNSW index.

Design:
* ``Embedder`` wraps sentence-transformers (bge-large-en-v1.5 by default) and
  exposes a deterministic hash-embedding fallback so the whole pipeline is
  testable on machines without model downloads.
* SQLAlchemy models define ``documents``, ``chunks`` (pgvector ``Vector`` column +
  ``tsv`` tsvector for BM25-ish full-text search) and ``user_memory`` (Phase 3).
* ``init_db`` creates the pgvector extension, tables and an **HNSW** ANN index
  (cosine ops) plus a GIN index on the tsvector.
* Upserts are idempotent on ``chunk_id``; embeddings batched for throughput.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sqlalchemy import (JSON, Column, DateTime, Float, Index, Integer, String, Text,
                        create_engine, func, text)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import declarative_base, sessionmaker

from configs.settings import settings

logger = logging.getLogger("vector_store")
Base = declarative_base()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class Embedder:
    """Sentence-transformer wrapper with graceful offline fallback."""

    def __init__(self, model_name: str | None = None, dim: int | None = None):
        self.model_name = model_name or settings.rag.embedding_model
        self.dim = dim or settings.rag.embed_dim
        self._st = None
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.model_name)
            self.dim = self._st.get_sentence_embedding_dimension() or self.dim
            logger.info("Loaded embedding model %s (dim=%d)", self.model_name, self.dim)
        except Exception as exc:
            logger.warning("Embedding model unavailable (%s); using deterministic fallback.", exc)

    def embed(self, texts: Sequence[str], *, batch_size: int = 64, query: bool = False) -> np.ndarray:
        if self._st is not None:
            # bge models recommend an instruction prefix for queries.
            if query and "bge" in self.model_name:
                texts = [f"Represent this sentence for retrieval: {t}" for t in texts]
            vecs = self._st.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                   show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        return self._fallback_embed(texts)

    def _fallback_embed(self, texts: Sequence[str]) -> np.ndarray:
        """Hashing bag-of-words projection: stable, cheap, good enough for smoke tests."""
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # allow import even when pgvector package missing
    Vector = lambda dim: JSON  # noqa: E731


class DocumentRow(Base):
    __tablename__ = "documents"
    doc_id = Column(String(32), primary_key=True)
    source = Column(String(512), nullable=False)
    title = Column(String(512), default="")
    author = Column(String(256), default="")
    department = Column(String(128), default="general", index=True)
    last_updated = Column(String(32), default="")
    language = Column(String(8), default="en")
    pages = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, default=dict)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())


class ChunkRow(Base):
    __tablename__ = "chunks"
    chunk_id = Column(String(96), primary_key=True)
    doc_id = Column(String(32), index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.rag.embed_dim))
    metadata_ = Column("metadata", JSON, default=dict)
    token_count = Column(Integer, default=0)
    tsv = Column(TSVECTOR)
    __table_args__ = (
        Index("chunks_tsv_idx", "tsv", postgresql_using="gin"),
    )


class UserMemoryRow(Base):
    """Long-term per-user memory consumed by the agent (Phase 3)."""

    __tablename__ = "user_memory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), index=True, nullable=False)
    kind = Column(String(32), default="preference")   # preference | fact | history
    key = Column(String(256), default="")
    value = Column(Text, nullable=False)
    weight = Column(Float, default=1.0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# DB bootstrap
# ---------------------------------------------------------------------------
def get_engine(url: str | None = None):
    url = url or settings.rag.database_url
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine, *, with_pgvector: bool = True) -> None:
    """Create extensions, tables, and the HNSW vector index (idempotent)."""
    with engine.begin() as conn:
        if with_pgvector:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    Base.metadata.create_all(engine)
    if with_pgvector:
        cfg = settings.rag
        with engine.begin() as conn:
            conn.execute(text(
                f"""CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {cfg.hnsw_m}, ef_construction = {cfg.hnsw_ef_construction})"""
            ))


# ---------------------------------------------------------------------------
# Ingestion into Postgres
# ---------------------------------------------------------------------------
_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "that", "for"}


def tokenize_for_bm25(text_: str) -> str:
    toks = [t for t in re.findall(r"\w+", text_.lower()) if t not in _STOPWORDS]
    return " ".join(toks)


def upsert_documents(session, docs: Iterable[dict]) -> int:
    n = 0
    for d in docs:
        row = session.get(DocumentRow, d["doc_id"])
        payload = {k: d.get(k, "") for k in ("source", "title", "author", "department",
                                             "last_updated", "language", "pages")}
        if row:
            for k, v in payload.items():
                setattr(row, k, v)
        else:
            session.add(DocumentRow(doc_id=d["doc_id"], **payload))
        n += 1
    session.flush()
    return n


def embed_and_store_chunks(session, chunks: list[dict], embedder: Embedder,
                           batch_size: int = 64) -> int:
    """Batch-embed chunk texts and upsert rows with embedding + BM25 tsvector."""
    stored = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vecs = embedder.embed([c["text"] for c in batch])
        for c, v in zip(batch, vecs):
            session.execute(text(
                """INSERT INTO chunks (chunk_id, doc_id, text, embedding, metadata, token_count, tsv)
                   VALUES (:cid, :did, :txt, CAST(:emb AS vector), CAST(:meta AS jsonb), :tok,
                           to_tsvector('english', :bm))
                   ON CONFLICT (chunk_id) DO UPDATE SET
                     text = EXCLUDED.text, embedding = EXCLUDED.embedding,
                     metadata = EXCLUDED.metadata, token_count = EXCLUDED.token_count,
                     tsv = EXCLUDED.tsv"""
            ), {
                "cid": c["chunk_id"], "did": c["doc_id"], "txt": c["text"],
                "emb": json.dumps(v.tolist()),
                "meta": json.dumps(c.get("metadata", {})),
                "tok": c.get("metadata", {}).get("tokens", 0),
                "bm": tokenize_for_bm25(c["text"]),
            })
            stored += 1
        session.commit()
        logger.info("Embedded %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks))
    return stored


def load_corpus_jsonl(path: Path | str) -> tuple[list[dict], list[dict]]:
    """Read chunks.jsonl produced by ingestion+chunking stages."""
    docs, chunks = [], []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    for line in p.open(encoding="utf-8"):
        rec = json.loads(line)
        chunks.append(rec)
        md = rec.get("metadata", {})
        docs.append({"doc_id": rec["doc_id"], "source": md.get("source", ""),
                     "title": md.get("title", ""), "author": md.get("author", ""),
                     "department": md.get("department", "general"),
                     "last_updated": md.get("last_updated", ""), "pages": 0})
    seen, uniq_docs = set(), []
    for d in docs:
        if d["doc_id"] not in seen:
            seen.add(d["doc_id"])
            uniq_docs.append(d)
    return uniq_docs, chunks


def ingest_chunks_file(chunks_path: Path | str, database_url: str | None = None,
                       embedder: Embedder | None = None) -> dict:
    engine = get_engine(database_url)
    init_db(engine)
    Session = sessionmaker(bind=engine, future=True)
    embedder = embedder or Embedder()
    docs, chunks = load_corpus_jsonl(chunks_path)
    with Session() as s:
        ndocs = upsert_documents(s, docs)
        nchunks = embed_and_store_chunks(s, chunks, embedder)
    return {"documents": ndocs, "chunks": nchunks, "embed_dim": embedder.dim}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Embed chunks and index them in pgvector.")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()
    print(json.dumps(ingest_chunks_file(args.chunks, args.database_url), indent=2))


if __name__ == "__main__":
    main()
