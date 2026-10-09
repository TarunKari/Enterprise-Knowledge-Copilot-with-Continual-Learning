"""Phase 2 – Cross-encoder reranking (bge-reranker-v2-m3) with offline fallback.

Takes the fused top-20 candidates and deeply scores each (query, chunk) pair,
returning the true top-5. When the cross-encoder is unavailable, a lexical
overlap + metadata-freshness scorer stands in so the pipeline contract holds.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from configs.settings import settings
from src.rag.retrieval import RetrievedChunk

logger = logging.getLogger("reranker")


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rag.reranker_model
        self._ce = None
        try:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded: %s", self.model_name)
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s); using lexical fallback.", exc)

    # -- public API ---------------------------------------------------------
    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_k: int | None = None) -> list[RetrievedChunk]:
        if not candidates:
            return []
        top_k = top_k or settings.rag.rerank_top_k
        if self._ce is not None:
            pairs = [[query, c.text] for c in candidates]
            scores = self._ce.predict(pairs, show_progress_bar=False)
            scored = [
                RetrievedChunk(c.chunk_id, c.text, float(s), "reranked", c.metadata)
                for c, s in zip(candidates, scores)
            ]
        else:
            scored = [
                RetrievedChunk(c.chunk_id, c.text, self._lexical_score(query, c), "reranked", c.metadata)
                for c in candidates
            ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    # -- fallback scorer ----------------------------------------------------
    @staticmethod
    def _lexical_score(query: str, cand: RetrievedChunk) -> float:
        q_terms = set(re.findall(r"\w+", query.lower()))
        c_tokens = re.findall(r"\w+", cand.text.lower())
        c_set = set(c_tokens)
        overlap = len(q_terms & c_set) / max(1, len(q_terms))
        density = min(1.0, len(q_terms & c_set) / 8.0)
        length_penalty = 1.0 / math.sqrt(max(64, len(c_tokens)) / 256.0)
        freshness = 0.0
        lu = str(cand.metadata.get("last_updated", ""))
        m = re.search(r"(20\d{2})[-/]?(\d{2})?", lu)
        if m:
            year = int(m.group(1))
            freshness = max(0.0, min(0.25, (year - 2019) * 0.05))
        base = 0.7 * overlap + 0.3 * density
        return round(base * length_penalty + freshness, 6)


# Singleton-style convenience used by the pipeline & agent tools
_default_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker()
    return _default_reranker
