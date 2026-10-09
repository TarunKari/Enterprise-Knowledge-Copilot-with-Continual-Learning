"""Phase 2 – Intelligent chunking: recursive-character and semantic strategies.

Default target: ~500-token chunks with 50-token overlap (configurable), preserving
sentence/paragraph boundaries via a LangChain-style recursive splitter, plus an
optional embedding-based *semantic* splitter that cuts where sentence similarity
drops (topic shifts).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


def approximate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used when no HF tokenizer is loaded."""
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recursive character splitting (token-aware)
# ---------------------------------------------------------------------------
def _split_with_separators(text: str, separators: list[str]) -> list[str]:
    if not separators:
        return [text]
    sep = separators[0]
    rest = separators[1:]
    splits = re.split(f"(?<={re.escape(sep)})", text) if sep.strip() else re.split(sep, text)
    docs: list[str] = []
    for s in splits:
        if not s:
            continue
        docs.append(s)
    final: list[str] = []
    for d in docs:
        if len(d) <= len(text):  # placeholder; real check happens on token budget
            final.append(d)
    merged: list[str] = []
    buf = ""
    for d in final or [text]:
        candidate = (buf + d) if buf else d
        merged.append(candidate)
        buf = ""
    return merged or [text]


def recursive_split(text: str, keep: int, overlap: int,
                    tokens_fn: Callable[[str], int] = approximate_tokens,
                    separators: list[str] | None = None) -> list[str]:
    """Token-budgeted recursive splitter with overlap carry-over.

    Splits on the first separator whose pieces fit within ``keep`` tokens;
    otherwise recurses to finer separators. Consecutive windows share the last
    ``overlap`` tokens of the previous chunk.
    """
    separators = separators or DEFAULT_SEPARATORS
    units: list[str] = []
    seps = separators
    remaining = [text]
    while seps:
        nxt: list[str] = []
        progressed = False
        for piece in remaining:
            parts = [p for p in re.split(f"(?<={re.escape(seps[0])})", piece) if p] \
                if seps[0].strip() else re.split(seps[0], piece)
            if len(parts) > 1:
                nxt.extend(parts)
                progressed = True
            elif tokens_fn(piece) <= keep or not seps[1:]:
                nxt.append(piece)
            else:
                nxt.append(piece)
        if not progressed:
            seps = seps[1:]
            remaining = nxt
            continue
        if all(tokens_fn(p) <= keep for p in nxt):
            units = nxt
            break
        seps = seps[1:]
        remaining = nxt
    units = units or [text]

    # Merge small neighbours up to budget, then build overlapping windows.
    merged: list[str] = []
    buf = ""
    for u in units:
        if buf and tokens_fn(buf + u) > keep:
            merged.append(buf.strip())
            tail = _tail_tokens(buf, overlap, tokens_fn)
            buf = (tail + " " + u).strip() if tail else u
        else:
            buf = (buf + u) if buf else u
    if buf.strip():
        merged.append(buf.strip())

    # Hard-safety pass: force-split anything still over budget (long lines w/o spaces).
    final: list[str] = []
    for m in merged:
        if tokens_fn(m) <= keep * 1.2:
            final.append(m)
        else:
            words = m.split(" ")
            window, cur = [], 0
            for w in words:
                cur += len(w) / 4 + 0.25
                window.append(w)
                if cur >= keep:
                    final.append(" ".join(window))
                    overlap_words = max(1, int(overlap * 4) // 5)
                    window = window[-overlap_words:]
                    cur = sum(len(x) / 4 for x in window)
            if window:
                final.append(" ".join(window))
    return [f for f in final if f.strip()]


def _tail_tokens(text: str, n_tokens: int, tokens_fn) -> str:
    if n_tokens <= 0:
        return ""
    words = text.split(" ")
    need = max(1, int(n_tokens * 4))  # approx chars
    tail = words[-max(1, need // 6):] if len(words) > 3 else []
    out = " ".join(tail)
    while tokens_fn(out) > n_tokens and len(out) > 5:
        out = " ".join(out.split(" ")[1:])
    return out


# ---------------------------------------------------------------------------
# Semantic splitting (embedding-driven topic shift detection)
# ---------------------------------------------------------------------------
def semantic_split(text: str, keep: int, overlap: int, embed_fn=None,
                   threshold: float = 0.75) -> list[str]:
    """Group consecutive sentences while cosine similarity stays above threshold;
    flush a chunk on topic shift or when the token budget is exceeded."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if embed_fn is None:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            def embed_fn(texts):  # noqa: E306
                return st_model.encode(texts, normalize_embeddings=True)
        except Exception:
            embed_fn = None

    if embed_fn is None:  # fallback: pure recursive splitter
        return recursive_split(text, keep, overlap)

    import numpy as np

    embs = np.asarray(embed_fn(sentences), dtype="float32")
    chunks, cur, cur_idx = [], [], 0
    for i, sent in enumerate(sentences):
        cur.append(sent)
        over_budget = approximate_tokens(" ".join(cur)) > keep
        shifted = False
        if i + 1 < len(sentences):
            sim = float(np.dot(embs[i], embs[i + 1]) /
                        (np.linalg.norm(embs[i]) * np.linalg.norm(embs[i + 1]) + 1e-9))
            shifted = sim < threshold
        if over_budget or (shifted and approximate_tokens(" ".join(cur)) > keep // 3):
            chunks.append(" ".join(cur).strip())
            tail = _tail_tokens(" ".join(cur), overlap, approximate_tokens)
            cur = [tail] if tail else []
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Corpus-level API
# ---------------------------------------------------------------------------
def chunk_document(doc: dict, *, strategy: str = "recursive",
                   size: int = 500, overlap: int = 50) -> list[Chunk]:
    out: list[Chunk] = []
    for i, sec in enumerate(doc.get("sections", [])):
        body = (("[" + sec["heading"] + "] ") if sec.get("heading") else "") + sec["text"]
        pieces = (semantic_split(body, size, overlap) if strategy == "semantic"
                  else recursive_split(body, size, overlap))
        for j, piece in enumerate(pieces):
            out.append(Chunk(
                chunk_id=f"{doc['doc_id']}::{i:03d}::{j:03d}",
                doc_id=doc["doc_id"],
                text=piece,
                metadata={
                    "source": doc["source"], "title": doc.get("title", ""),
                    "author": doc.get("author", ""), "department": doc.get("department", "general"),
                    "last_updated": doc.get("last_updated", ""),
                    "page": sec.get("page"), "heading": sec.get("heading", ""),
                    "section_index": i, "chunk_index": j,
                    "tokens": approximate_tokens(piece), "strategy": strategy,
                },
            ))
    return out


def chunk_corpus(corpus_path: Path | str, out_path: Path | str, **kw) -> int:
    n = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            doc = json.loads(line)
            for ch in chunk_document(doc, **kw):
                fout.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk the ingested corpus.")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--out", default="data/processed/chunks.jsonl")
    ap.add_argument("--strategy", choices=["recursive", "semantic"], default="recursive")
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()
    n = chunk_corpus(args.corpus, args.out, strategy=args.strategy, size=args.size, overlap=args.overlap)
    print(json.dumps({"chunks": n, "output": args.out, "strategy": args.strategy}, indent=2))


if __name__ == "__main__":
    main()
