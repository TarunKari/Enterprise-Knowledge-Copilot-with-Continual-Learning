"""Phase 4 – FastAPI application: /chat, /ingest, /feedback, approval callback, metrics.

Wires together Phase-2 RAG, the Phase-3 LangGraph agent (with Slack human-gate),
the feedback store and observability layer. Runs with zero external services by
auto-selecting in-memory fallbacks; point DATABASE_URL / vLLM at real infra for
production behaviour.

Start:  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
Docs:   http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from configs.settings import settings
from src.agent.graph import AgentGraph, AgentState, _dict_to_state
from src.agent.memory import MemoryManager
from src.agent.tools import build_default_registry
from src.continual.feedback_loop import FeedbackStore
from src.observability.tracing import get_monitor, get_tracer
from src.rag.pipeline import RagPipeline

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Enterprise LLM Platform", version="1.0.0",
              description="Fine-tuned LLM + Hybrid RAG + Agentic orchestration + continual learning")

# ---------------------------------------------------------------------------
# Dependency container (lazy singletons)
# ---------------------------------------------------------------------------
class Container:
    def __init__(self):
        self._rag: RagPipeline | None = None
        self._graph: AgentGraph | None = None
        self._store: FeedbackStore | None = None
        self._memory: MemoryManager | None = None
        self._sessions: dict[str, dict] = {}      # short-term conversation store

    @property
    def rag(self) -> RagPipeline:
        if self._rag is None:
            self._rag = RagPipeline()
        return self._rag

    @property
    def store(self) -> FeedbackStore:
        if self._store is None:
            self._store = FeedbackStore()
        return self._store

    @property
    def memory(self) -> MemoryManager:
        if self._memory is None:
            sf = None
            try:
                from sqlalchemy.orm import sessionmaker

                from src.rag.vector_store import get_engine, init_db
                engine = get_engine()
                init_db(engine, with_pgvector=False)
                sf = sessionmaker(bind=engine, future=True)
            except Exception:
                sf = None
            self._memory = MemoryManager(session_factory=sf)
        return self._memory

    @property
    def graph(self) -> AgentGraph:
        if self._graph is None:
            tools = build_default_registry(rag_pipeline=self.rag,
                                           db_session_factory=self.memory.session_factory)
            self._graph = AgentGraph(llm_fn=self._llm, tools=tools,
                                     memory_loader=lambda uid: self.memory.load(uid))
        return self._graph

    @staticmethod
    def _llm(prompt: str) -> str:
        return RagPipeline._default_llm(prompt)


container = Container()


def get_container() -> Container:
    return container


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    user_id: str = "anon"
    session_id: Optional[str] = None
    mode: str = Field(default="agent", pattern="^(agent|rag)$")
    department_filter: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    interaction_id: int
    answer: str
    citations: list[str] = []
    approval_pending: bool = False
    pending_action: Optional[dict] = None
    timings_ms: dict = {}


class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: str = Field(pattern="^(up|down)$")
    comment: str = ""
    user_id: str = "anon"


class IngestRequest(BaseModel):
    path: str = "data/raw"
    strategy: str = Field(default="recursive", pattern="^(recursive|semantic)$")
    use_postgres: bool = True


class ApprovalRequest(BaseModel):
    session_id: str
    approve: bool
    reviewer: str = "human"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background: BackgroundTasks,
         c: Container = Depends(get_container)):
    tracer = get_tracer()
    session_id = req.session_id or str(uuid.uuid4())
    history = c._sessions.get(session_id, [])
    t0 = time.perf_counter()

    if req.mode == "rag":
        with tracer.span("retrieval", "rag-answer", q=req.message[:120]) as holder:
            res = c.rag.answer(req.message,
                               filters={"department": req.department_filter}
                               if req.department_filter else None, history=history)
        answer, citations = res.answer, res.citations
        pending, pending_action = False, None
        contexts = res.contexts
    else:
        with tracer.span("state", "agent-run", user=req.user_id) as holder:
            st = c.graph.run(req.message, user_id=req.user_id, history=history)
        answer = st.answer or "The agent paused awaiting human approval."
        citations = [ctx.get("metadata", {}).get("source", "") for ctx in st.retrieved_context
                     if isinstance(ctx, dict) and ctx.get("metadata")]
        pending = st.approval_status == "pending"
        pending_action = st.pending_action
        contexts = st.retrieved_context[:5]
        if pending:
            c._sessions[session_id + "::pending"] = st.to_dict()

    latency = round((time.perf_counter() - t0) * 1000, 1)
    iid = c.store.log_interaction(question=req.message, answer=answer, user_id=req.user_id,
                                  session_id=session_id, contexts=contexts,
                                  citations=citations, meta={"mode": req.mode, "latency_ms": latency})

    # update short-term memory + capture long-term preferences
    c._sessions[session_id] = (history + [{"role": "user", "content": req.message},
                                          {"role": "assistant", "content": answer}])[-20:]
    background.add_task(c.memory.capture_from_conversation, req.user_id,
                        [{"role": "user", "content": req.message}])
    background.add_task(lambda: get_monitor().check())

    return ChatResponse(session_id=session_id, interaction_id=iid, answer=answer,
                        citations=[x for x in citations if x], approval_pending=pending,
                        pending_action=pending_action,
                        timings_ms={"total": latency})


@app.post("/approvals/callback", response_model=ChatResponse)
def approval_callback(req: ApprovalRequest, c: Container = Depends(get_container)):
    """Slack interactive-action / UI button endpoint resumes the parked graph."""
    key = req.session_id + "::pending"
    snap = c._sessions.get(key)
    if not snap:
        raise HTTPException(404, "no pending action for this session")
    st = _dict_to_state(snap)
    with get_tracer().span("state", "human-gate-resume", approved=req.approve):
        st = c.graph.resolve_approval(st, req.approve)
    c._sessions.pop(key, None)
    c._sessions[req.session_id] = st.messages[-20:]
    iid = c.store.log_interaction(question=f"[approval {st.pending_action}]",
                                  answer=st.answer, user_id=req.reviewer,
                                  session_id=req.session_id,
                                  meta={"approved": req.approve})
    return ChatResponse(session_id=req.session_id, interaction_id=iid, answer=st.answer,
                        timings_ms={})


@app.post("/feedback")
def feedback(req: FeedbackRequest, c: Container = Depends(get_container)):
    ok = c.store.rate(req.interaction_id, req.rating, comment=req.comment, user_id=req.user_id)
    if not ok:
        raise HTTPException(404, "interaction not found or bad rating")
    get_tracer().record(__import__("src.observability.tracing", fromlist=["TraceEvent"]).TraceEvent(
        kind="feedback", name=req.rating, duration_ms=0.0))
    return {"stored": True, "interaction_id": req.interaction_id}


@app.post("/ingest")
def ingest(req: IngestRequest, c: Container = Depends(get_container)):
    """Parse → chunk → embed → index a directory of domain documents."""
    from src.rag.chunking import chunk_corpus
    from src.rag.ingestion import ingest_directory, save_corpus

    docs = list(ingest_directory(req.path))
    if not docs:
        raise HTTPException(400, f"no parseable documents under {req.path}")
    corpus_tmp = "data/processed/corpus.jsonl"
    chunks_tmp = "data/processed/chunks.jsonl"
    save_corpus(docs, corpus_tmp)
    n_chunks = chunk_corpus(corpus_tmp, chunks_tmp, strategy=req.strategy)
    result = {"documents": len(docs), "chunks": n_chunks}
    if req.use_postgres:
        try:
            from src.rag.vector_store import ingest_chunks_file

            result.update(ingest_chunks_file(chunks_tmp))
        except Exception as exc:
            logger.warning("Postgres indexing unavailable (%s); serving from memory.", exc)
            c._rag = None  # force reload into in-memory mode
            result["backend"] = "in-memory"
    return result


@app.get("/metrics/summary")
def metrics_summary():
    mon = get_monitor()
    return {"window_metrics": mon.compute_metrics(), "rules": [r.name for r in mon.rules]}


@app.get("/flywheel/status")
def flywheel_status(c: Container = Depends(get_container)):
    rows = c.store.export_labeled()
    ups = sum(1 for r in rows if r["score"] > 0)
    downs = sum(1 for r in rows if r["score"] < 0)
    return {"interactions": len(rows), "net_upvotes": ups, "net_downvotes": downs,
            "next_batch_ready": ups >= 10}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.platform.api_host, port=settings.platform.api_port)
