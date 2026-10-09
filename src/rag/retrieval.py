"""Phase 2 – Query processing: rewriting, HyDE, hybrid retrieval + RRF fusion.

Components
----------
* ``QueryRewriter``   – lightweight LLM (or heuristic) expansion into N variants.
* ``hyde_generate``   – Hypothetical Document Embeddings: fabricate a plausible
  answer document, embed *that* for better recall on question-shaped queries.
* ``dense_search``    – pgvector cosine ANN over the HNSW index.
* ``bm25_search``     – Postgres full-text ts_rank search (BM25-flavoured sparse leg).
* ``reciprocal_rank_fusion`` – merge ranked lists (k=60 default) -> top-20.
* ``InMemoryRetriever`` – pure-Python twin of the Postgres path so the entire RAG
  pipeline runs/tests without a database server.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from configs.settings import settings

logger = logging.getLogger("retrieval")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str            # "dense" | "sparse" | "fused" | "reranked"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query rewriting & HyDE
# ---------------------------------------------------------------------------
_REWRITE_PROMPT = (
    "You are a query-understanding module. Given the user's question, produce {n} "
    "distinct reformulations that would improve document retrieval (synonyms, "
    "expansions of abbreviations, related formal phrasings). Respond ONLY with a "
    "JSON array of strings.\nQuestion: {q}\nReformulations:"
)

_HYDE_PROMPT = (
    "Write a short factual paragraph (4-6 sentences) that would plausibly ANSWER "
    "the following question. Do not hedge; it is used as a retrieval probe.\n"
    "Question: {q}\nHypothetical document:"
)


class QueryRewriter:
    """LLM-backed rewriter with deterministic heuristic fallback."""

    def __init__(self, llm=None, n_variants: int | None = None):
        self.llm = llm                      # callable(prompt:str)->str ; optional
        self.n = n_variants or settings.rag.query_rewrite_variants

    def rewrite(self, query: str) -> list[str]:
        if self.llm is not None:
            try:
                raw = self.llm(_REWRITE_PROMPT.format(n=self.n, q=query))
                arr = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
                out = [str(x).strip() for x in arr if str(x).strip()]
                if out:
                    return [query] + out[: self.n]
            except Exception as exc:
                logger.debug("LLM rewrite failed (%s); using heuristics", exc)
        return [query] + self._heuristic(query)

    @staticmethod
    def _heuristic(query: str) -> list[str]:
        q = query.strip().rstrip("?")
        variants = []
        # Drop interrogatives -> keyword form
        kw = re.sub(r"^(how|what|why|when|where|who|can|do|does|is|are|should)\s+(?:i|we|you|the)?\s*", "", q, flags=re.I)
        if kw and kw != q:
            variants.append(kw)
        # Common enterprise expansions
        expansions = {"vpn": "virtual private network access", "pto": "paid time off leave",
                      "hr": "human resources", "it": "information technology",
                      "401k": "retirement savings plan", "sla": "service level agreement"}
        lowered = q.lower()
        for abbr, full in expansions.items():
            if re.search(rf"\b{abbr}\b", lowered):
                variants.append(re.sub(rf"\b{abbr}\b", full, lowered, flags=re.I))
                break
        variants.append(q + " policy procedure documentation")
        return variants[: max(1, settings.rag.query_rewrite_variants)]


def hyde_generate(query: str, llm=None) -> str:
    """Return a hypothetical document for embedding-based retrieval probing."""
    if llm is not None:
        try:
            doc = llm(_HYDE_PROMPT.format(q=query)).strip()
            if len(doc) > 40:
                return doc
        except Exception as exc:
            logger.debug("HyDE LLM failed (%s)", exc)
    # Fallback: turn the question into an assertive statement.
    stmt = re.sub(r"^(how|what|why|when|where|who|do|does|can|is|are|should)\b", "The", query.strip().rstrip("?"), flags=re.I)
    return stmt.strip().capitalize() + ". This document describes the relevant policy, steps and details."


# ---------------------------------------------------------------------------
# Dense + sparse legs against Postgres/pgvector
# ---------------------------------------------------------------------------
DENSE_SQL = """
SELECT chunk_id, text, metadata, 1 - (embedding <=> CAST(:qvec AS vector)) AS score
FROM chunks
{filter_clause}
ORDER BY embedding <=> CAST(:qvec AS vector)
LIMIT :top_k
"""

SPARSE_SQL = """
SELECT chunk_id, text, metadata,
       ts_rank_cd(tsv, websearch_to_tsquery('english', :q)) AS score
FROM chunks
WHERE tsv @@ websearch_to_tsquery('english', :q)
{filter_clause}
ORDER BY score DESC
LIMIT :top_k
"""


def _build_filter(filters: dict | None) -> tuple[str, dict]:
    clauses, params = [], {}
    if filters:
        if filters.get("department"):
            clauses.append("metadata->>'department' = :dept")
            params["dept"] = filters["department"]
        if filters.get("source"):
            clauses.append("metadata->>'source' = :src")
            params["src"] = filters["source"]
        if filters.get("doc_id"):
            clauses.append("doc_id = :doc_id")
            params["doc_id"] = filters["doc_id"]
    fc = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return fc, params


def dense_search(session, query_vec: Sequence[float], *, top_k: int | None = None,
                 filters: dict | None = None) -> list[RetrievedChunk]:
    from sqlalchemy import text as sa_text

    fc, fp = _build_filter(filters)
    sql = sa_text(DENSE_SQL.replace("{filter_clause}", fc))
    rows = session.execute(sql, {"qvec": json.dumps(list(map(float, query_vec))),
                                 "top_k": top_k or settings.rag.dense_top_k, **fp}).fetchall()
    return [RetrievedChunk(r.chunk_id, r.text, float(r.score), "dense", _as_dict(r.metadata)) for r in rows]


def bm25_search(session, query: str, *, top_k: int | None = None,
                filters: dict | None = None) -> list[RetrievedChunk]:
    from sqlalchemy import text as sa_text

    fc, fp = _build_filter(filters)
    sql = sa_text(SPARSE_SQL.replace("{filter_clause}", fc))
    rows = session.execute(sql, {"q": query, "top_k": top_k or settings.rag.sparse_top_k, **fp}).fetchall()
    return [RetrievedChunk(r.chunk_id, r.text, float(r.score), "sparse", _as_dict(r.metadata)) for r in rows]


def _as_dict(meta):
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta or {}


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(ranked_lists: list[list[RetrievedChunk]], k: int | None = None,
                           top_n: int | None = None) -> list[RetrievedChunk]:
    """RRF: score(d) = Σ_lists 1/(k + rank_i(d)); merges dense+sparse into one top-N."""
    k = k or settings.rag.rrf_k
    top_n = top_n or settings.rag.fuse_top_n
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievedChunk] = {}
    for lst in ranked_lists:
        for rank, ch in enumerate(lst, start=1):
            scores[ch.chunk_id] += 1.0 / (k + rank)
            by_id.setdefault(ch.chunk_id, ch)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    out = []
    for cid, s in fused:
        c = by_id[cid]
        out.append(RetrievedChunk(c.chunk_id, c.text, s, "fused", c.metadata))
    return out


# ---------------------------------------------------------------------------
# In-memory hybrid retriever (DB-free mode for dev/CI/local-first serving)
# ---------------------------------------------------------------------------
class InMemoryRetriever:
    """TF-IDF cosine (dense surrogate) + BM25 (true Okapi) over an in-memory corpus."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized = [self._tok(c["text"]) for c in chunks]
        self.df = defaultdict(int)
        for toks in self.tokenized:
            for t in set(toks):
                self.df[t] += 1
        self.N = max(1, len(chunks))
        self.avgdl = sum(len(t) for t in self.tokenized) / self.N
        self.idf = {t: __import__("math").log(1 + (self.N - d + 0.5) / (d + 0.5))
                    for t, d in self.df.items()}

    @staticmethod
    def _tok(text_: str) -> list[str]:
        return re.findall(r"\w+", text_.lower())

    def dense(self, query: str, top_k: int) -> list[RetrievedChunk]:
        qt = set(self._tok(query))
        scored = []
        for i, toks in enumerate(self.tokenized):
            inter = len(qt & set(toks))
            if inter == 0:
                continue
            tfidf_q = sum(self.idf.get(t, 0) for t in qt & set(toks))
            norm = (len(toks) ** 0.5) + 1e-6
            scored.append((tfidf_q * inter / norm, i))
        scored.sort(reverse=True)
        return [RetrievedChunk(self.chunks[i]["chunk_id"], self.chunks[i]["text"], s,
                               "dense", self.chunks[i].get("metadata", {}))
                for s, i in scored[:top_k]]

    def bm25(self, query: str, top_k: int, k1: float = 1.5, b: float = 0.75) -> list[RetrievedChunk]:
        q_terms = self._tok(query)
        scores = defaultdict(float)
        for i, toks in enumerate(self.tokenized):
            dl = len(toks)
            tf_map = defaultdict(int)
            for t in toks:
                tf_map[t] += 1
            for t in q_terms:
                tf = tf_map.get(t, 0)
                if tf:
                    idf = self.idf.get(t, 0.0)
                    scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [RetrievedChunk(self.chunks[i]["chunk_id"], self.chunks[i]["text"], s,
                               "sparse", self.chunks[i].get("metadata", {}))
                for s, i in ranked]

    def hybrid(self, query: str, *, top_k_dense: int = 50, top_k_sparse: int = 50,
               top_n: int = 20, filters: dict | None = None) -> list[RetrievedChunk]:
        dense = self.dense(query, top_k_dense)
        sparse = self.bm25(query, top_k_sparse)
        if filters:
            def ok(lst):
                return [c for c in lst if all(str(c.metadata.get(k, "")) == str(v) for k, v in filters.items())]
            dense, sparse = ok(dense), ok(sparse)
        return reciprocal_rank_fusion([dense, sparse], top_n=top_n)
