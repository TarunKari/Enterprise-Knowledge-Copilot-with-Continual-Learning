"""Phase 2 – End-to-end RAG pipeline orchestrator.

Flow: query -> rewrite(+HyDE) -> hybrid dense/BM25 -> RRF top-20 -> cross-encoder
rerank top-5 -> grounded prompt with citations -> LLM answer.

Supports two backends:
* Postgres/pgvector (production)
* In-memory JSONL corpus (dev/CI, zero infrastructure)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from configs.settings import settings
from src.finetune.prompting import build_rag_prompt, extract_answer
from src.rag.reranker import get_reranker
from src.rag.retrieval import (InMemoryRetriever, QueryRewriter, RetrievedChunk,
                               bm25_search, dense_search, hyde_generate,
                               reciprocal_rank_fusion)

logger = logging.getLogger("rag_pipeline")


@dataclass
class RagAnswer:
    question: str
    answer: str
    contexts: list[dict]                 # the final top-k chunks fed to the model
    citations: list[str]
    timings_ms: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)


def _load_local_chunks(path: str | None = None) -> list[dict]:
    p = Path(path or "data/processed/chunks.jsonl")
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open(encoding="utf-8")]


class RagPipeline:
    def __init__(self, llm_fn: Callable[[str], str] | None = None, *,
                 use_postgres: bool | None = None, chunks_path: str | None = None):
        """``llm_fn`` is any prompt->completion callable (vLLM proxy, OpenAI SDK, stub)."""
        self.llm_fn = llm_fn or self._default_llm
        self.rewriter = QueryRewriter(llm=self.llm_fn if settings.rag.hyde_enabled else None)
        self.reranker = get_reranker()
        self.use_postgres = use_postgres
        self._mem: InMemoryRetriever | None = None
        self._chunks_path = chunks_path
        if use_postgres is None:
            try:
                from src.rag.vector_store import get_engine

                engine = get_engine()
                with engine.connect() as conn:
                    n = conn.exec_driver_sql("SELECT COUNT(*) FROM chunks").scalar()
                self.use_postgres = bool(n)
            except Exception:
                self.use_postgres = False
        if not self.use_postgres:
            chunks = _load_local_chunks(self._chunks_path)
            self._mem = InMemoryRetriever(chunks)
            logger.info("RAG running in in-memory mode (%d chunks)", len(self._mem.chunks))

    # ------------------------------------------------------------------
    def retrieve(self, question: str, *, filters: dict | None = None,
                 top_n: int | None = None) -> tuple[list[RetrievedChunk], dict]:
        t0 = time.perf_counter()
        trace: dict = {"filters": filters or {}}

        variants = self.rewriter.rewrite(question)
        hyde_doc = hyde_generate(question, llm=self.llm_fn) if settings.rag.hyde_enabled else ""
        trace["query_variants"] = variants
        trace["hyde"] = hyde_doc[:160]

        candidate_lists: list[list[RetrievedChunk]] = []
        if self.use_postgres:
            from sqlalchemy.orm import sessionmaker

            from src.rag.vector_store import Embedder, get_engine

            embedder = Embedder()
            Session = sessionmaker(bind=get_engine(), future=True)
            probe_texts = variants + ([hyde_doc] if hyde_doc else [])
            vecs = embedder.embed(probe_texts, query=True)
            with Session() as s:
                for v, vec in zip(probe_texts, vecs):
                    candidate_lists.append(dense_search(s, vec, filters=filters))
                for v in variants:
                    candidate_lists.append(bm25_search(s, v, filters=filters))
        else:
            for v in variants:
                candidate_lists.append(self._mem.dense(v, settings.rag.dense_top_k))
                candidate_lists.append(self._mem.bm25(v, settings.rag.sparse_top_k))
            if hyde_doc:
                candidate_lists.append(self._mem.dense(hyde_doc, settings.rag.dense_top_k))

        fused = reciprocal_rank_fusion(candidate_lists, top_n=top_n or settings.rag.fuse_top_n)
        trace["n_candidates"] = sum(len(l) for l in candidate_lists)
        trace["n_fused"] = len(fused)

        t1 = time.perf_counter()
        top_k = self.reranker.rerank(question, fused, top_k=settings.rag.rerank_top_k)
        trace["reranked_ids"] = [c.chunk_id for c in top_k]
        t2 = time.perf_counter()
        trace["timings_ms"] = {
            "retrieve_fuse": round((t1 - t0) * 1000, 1),
            "rerank": round((t2 - t1) * 1000, 1),
        }
        return top_k, trace

    # ------------------------------------------------------------------
    def answer(self, question: str, *, filters: dict | None = None,
               history: list[dict] | None = None) -> RagAnswer:
        chunks, trace = self.retrieve(question, filters=filters)
        ctx_dicts = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        prompt = build_rag_prompt(question, ctx_dicts)
        if history:
            turn_blob = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history[-4:])
            prompt = f"Conversation so far:\n{turn_blob}\n\n" + prompt
        t0 = time.perf_counter()
        raw = self.llm_fn(prompt)
        answer = extract_answer(raw) if raw else "I don't know."
        trace["timings_ms"] = {**trace.get("timings_ms", {}),
                               "generate": round((time.perf_counter() - t0) * 1000, 1)}
        citations = []
        for c in chunks:
            md = c.metadata
            tag = f"[Source: {md.get('source','doc')}"
            if md.get("page"):
                tag += f", Page {md['page']}"
            tag += "]"
            if tag not in citations:
                citations.append(tag)
        return RagAnswer(
            question=question, answer=answer, contexts=ctx_dicts,
            citations=citations, timings_ms=trace.get("timings_ms", {}), trace=trace,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _default_llm(prompt: str) -> str:
        """OpenAI-compatible call against the vLLM-served fine-tuned model."""
        try:
            from openai import OpenAI

            client = OpenAI(base_url=settings.agent.inference_base_url,
                            api_key=settings.agent.inference_api_key)
            resp = client.chat.completions.create(
                model="local", messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("No inference server reachable (%s); returning extractive fallback.", exc)
            marker = "User Question:"
            head = prompt.split(marker)[0]
            lines = [ln.strip() for ln in head.splitlines()
                     if ln.strip() and not ln.startswith(("System:", "-----", "Use ONLY"))]
            return (lines[0][:300] if lines else "I don't know.") + " [extractive-fallback]"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run a query through the full RAG pipeline.")
    ap.add_argument("--question", required=True)
    ap.add_argument("--department", default=None)
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    args = ap.parse_args()
    pipe = RagPipeline(chunks_path=args.chunks)
    res = pipe.answer(args.question,
                      filters={"department": args.department} if args.department else None)
    print(json.dumps({"answer": res.answer, "citations": res.citations,
                      "n_contexts": len(res.contexts), "timings_ms": res.timings_ms}, indent=2))


if __name__ == "__main__":
    main()
