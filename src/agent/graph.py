"""Phase 3 – LangGraph agent: state machine, planning, memory, human gate.

Graph topology
--------------
                        ┌────────────┐
      user msg ───────▶ │ load_memory│ (long-term Postgres + short-term history)
                        └─────┬──────┘
                              ▼
                        ┌──────────┐   simple     ┌────────────┐
                        │  Router  ├────────────▶ │ generate   │──┐
                        └────┬─────┘              └────────────┘  │
                             │ complex (plan)                     │
                             ▼                                    │
                        ┌──────────┐        ┌──────────────┐      │
                        │ Retriever│──────▶ │ Tool Executor│      │
                        └──────────┘        └──────┬───────┘      │
                                                   │ destructive  │
                                                   ▼              │
                                            ┌─────────────┐       │
                                            │ Human Gate  │       │
                                            │ (interrupt +│       │
                                            │  Slack btns)│       │
                                            └──────┬──────┘       │
                                       approved ◀──┴──▶ rejected  │
                                           │            │         │
                                     Tool Executor   finalize ◀───┘
                                           │            │
                                           ▼            ▼
                                      finalize ───▶ END

If ``langgraph`` is installed the real StateGraph is compiled (with
``interrupt()`` support for the human gate); otherwise a faithful drop-in
executor with identical nodes/edges/semantics runs — so this module works in CI.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from configs.settings import settings
from src.agent.tools import ToolRegistry, parse_tool_call

logger = logging.getLogger("agent_graph")

APPROVAL_TIMEOUT_NOTE = "approval pending"


@dataclass
class AgentState:
    """LangGraph State schema (TypedDict-equivalent)."""

    messages: list[dict] = field(default_factory=list)          # short-term memory
    retrieved_context: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    memory: list[dict] = field(default_factory=list)            # long-term rows
    plan: list[str] = field(default_factory=list)
    approval_status: Literal["not_required", "pending", "approved", "rejected"] = "not_required"
    pending_action: dict | None = None
    answer: str = ""
    iterations: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


ROUTER_PROMPT = """You are the Router of an enterprise agent.
Long-term memory about this user:
{memory}

Available tools (JSON schemas):
{tools}

Decide between:
- "answer": respond directly (greeting/chit-chat/general knowledge)
- "retrieve": needs enterprise knowledge → search_knowledge_base
- "plan": multi-step task → output numbered steps and required tool calls

Reply ONLY JSON:
{{"decision": "answer|retrieve|plan", "plan": ["step1", ...], "tool": "...", "args": {{...}}}}

User message: {question}
JSON:"""

EXECUTE_PROMPT = """Given the goal, plan, retrieved context and prior results, choose the NEXT action.
Context: {context}
Prior results: {results}
Remaining failure analysis: {analysis}

Tools:
{tools}

Reply ONLY JSON {{"tool": "...", "args": {{...}}}} or {{"final": "..."}} when done."""


class AgentGraph:
    def __init__(self, llm_fn: Callable[[str], str], tools: ToolRegistry,
                 memory_loader: Callable[[str], list[dict]] | None = None,
                 notifier: Callable[[dict], None] | None = None,
                 max_iterations: int | None = None):
        self.llm = llm_fn
        self.tools = tools
        self.memory_loader = memory_loader or (lambda uid: [])
        self.notifier = notifier or slack_approval_notifier
        self.max_iterations = max_iterations or settings.agent.max_iterations
        self._compiled_langgraph = None
        try:
            self._compiled_langgraph = self._build_langgraph()
            logger.info("Using native LangGraph StateGraph.")
        except Exception as exc:
            logger.info("LangGraph unavailable (%s); using built-in executor.", exc)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def node_load_memory(self, st: AgentState, user_id: str) -> AgentState:
        st.memory = self.memory_loader(user_id)
        return st

    def node_router(self, st: AgentState) -> AgentState:
        question = st.messages[-1]["content"] if st.messages else ""
        prompt = ROUTER_PROMPT.format(
            memory=json.dumps(st.memory[:8], ensure_ascii=False) or "[]",
            tools=self.tools.schemas_for_prompt(), question=question)
        decision = self._json_or_default(self.llm(prompt),
                                         {"decision": "retrieve", "plan": [], "tool": "", "args": {}})
        d = decision.get("decision", "answer")
        if decision.get("tool"):
            d = "plan" if len(decision.get("plan", [])) > 1 else "act"
        st.plan = [p for p in decision.get("plan", []) if isinstance(p, str)]
        st.tool_calls.append({"from": "router", **({k: v for k, v in decision.items() if k in ("tool", "args")})})
        st.retrieved_context.append({"router_decision": d})
        if d == "answer":
            st.answer = self._direct_answer(question)
        elif d == "retrieve":
            st.tool_calls.append({"from": "router-auto", "tool": "search_knowledge_base",
                                  "args": {"query": question}})
        st._router_decision = d  # type: ignore[attr-defined]
        return st

    def node_retriever(self, st: AgentState) -> AgentState:
        q = st.messages[-1]["content"] if st.messages else ""
        res = self.tools.execute("search_knowledge_base", {"query": q})
        if res.ok:
            st.retrieved_context.extend(res.output or [])
        else:
            st.errors.append(f"retriever: {res.error}")
        return st

    def node_executor(self, st: AgentState) -> AgentState:
        st.iterations += 1
        call = self._next_pending_call(st)
        if call is None:
            prompt = EXECUTE_PROMPT.format(
                context=json.dumps(st.retrieved_context[:5], ensure_ascii=False)[:2000],
                results=json.dumps(st.tool_results[-4:], ensure_ascii=False)[:1500],
                analysis="; ".join(st.errors[-2:]) or "none",
                tools=self.tools.schemas_for_prompt())
            parsed = self._json_or_default(self.llm(prompt), {"final": "I could not complete the task."})
            if "final" in parsed:
                st.answer = str(parsed["final"])
                return st
            call = {"tool": parsed.get("tool", ""), "args": parsed.get("args", {})}
            st.tool_calls.append({"from": "llm", **call})

        spec = self.tools.tools.get(call["tool"])
        if spec is None:
            st.errors.append(f"unknown tool {call['tool']}")
            st.tool_calls.remove(call)
            return st
        if spec.destructive and settings.agent.require_approval_for_writes \
                and st.approval_status != "approved":
            st.pending_action = call
            st.approval_status = "pending"
            self.notifier({"state": st.to_dict(), "action": call})
            return st
        result = self.tools.execute(call["tool"], call.get("args", {}))
        st.tool_results.append({"tool": call["tool"], "ok": result.ok,
                                "output": result.output, "error": result.error,
                                "attempts": result.attempts})
        st.tool_calls.remove(call)
        if not result.ok:
            st.errors.append(result.analysis or result.error or "tool failed")
            if result.attempts >= spec.max_retries + 1:
                # Failure recovery exhausted → escalate to user after one LLM retry pass
                if st.iterations < self.max_iterations:
                    st.tool_calls.append({"from": "retry-plan", "tool": "", "args": {}})
                    st.tool_calls.pop()  # let router-driven LLM pick next action on loop
        if not st.answer and st.iterations >= self.max_iterations:
            st.answer = ("Escalating to you: I hit my iteration budget. "
                         f"Issues: {'; '.join(st.errors[-2:]) or 'none'}")
        return st

    def node_human_gate(self, st: AgentState) -> AgentState:
        # In LangGraph-native mode this node calls interrupt(); here it simply
        # parks until resolve_approval() flips approval_status.
        if st.approval_status == "pending":
            st.answer = ""  # remain paused
        return st

    def node_finalize(self, st: AgentState) -> AgentState:
        if st.approval_status == "rejected":
            st.answer = "Action cancelled by human reviewer."
        elif not st.answer:
            ctx = st.retrieved_context[:3]
            st.answer = ("Grounded summary: " + (ctx[0]["text"][:220] if ctx and "text" in ctx[0]
                                                 else "Based on tool results, no further action needed."))
        st.messages.append({"role": "assistant", "content": st.answer})
        return st

    # ------------------------------------------------------------------
    # Orchestration (fallback executor mirrors the graph edges)
    # ------------------------------------------------------------------
    def run(self, user_message: str, *, user_id: str = "anon",
            history: list[dict] | None = None) -> AgentState:
        if self._compiled_langgraph is not None:
            return self._run_native(user_message, user_id=user_id, history=history)
        st = AgentState(messages=(history or []) + [{"role": "user", "content": user_message}])
        st = self.node_load_memory(st, user_id)
        st = self.node_router(st)
        decision = getattr(st, "_router_decision", "retrieve")
        if decision == "answer":
            return self.node_finalize(st)
        if decision == "retrieve":
            st = self.node_retriever(st)
            return self.node_finalize(st)
        while st.iterations < self.max_iterations and not st.answer and st.approval_status != "pending":
            st = self.node_executor(st)
            if st.approval_status == "pending":
                st = self.node_human_gate(st)
                break
        if st.approval_status == "pending":
            return st
        return self.node_finalize(st)

    def resolve_approval(self, st: AgentState, approved: bool) -> AgentState:
        st.approval_status = "approved" if approved else "rejected"
        if approved and st.pending_action:
            call = st.pending_action
            st.pending_action = None
            res = self.tools.execute(call["tool"], call.get("args", {}))
            st.tool_results.append({"tool": call["tool"], "ok": res.ok,
                                    "output": res.output, "error": res.error, "approved": True})
        return self.node_finalize(st)

    # ------------------------------------------------------------------
    # Native LangGraph construction (used when package present)
    # ------------------------------------------------------------------
    def _build_langgraph(self):
        from langgraph.graph import END, START, StateGraph
        from typing import TypedDict

        class S(TypedDict, total=False):
            messages: list
            retrieved_context: list
            tool_calls: list
            tool_results: list
            memory: list
            plan: list
            approval_status: str
            pending_action: dict
            answer: str
            iterations: int
            errors: list

        g = StateGraph(S)
        bridge = _StateBridge(self)
        g.add_node("load_memory", bridge.load_memory)
        g.add_node("router", bridge.router)
        g.add_node("retriever", bridge.retriever)
        g.add_node("executor", bridge.executor)
        g.add_node("human_gate", bridge.human_gate)
        g.add_node("finalize", bridge.finalize)
        g.add_edge(START, "load_memory")
        g.add_edge("load_memory", "router")
        g.add_conditional_edges("router", bridge.route_after_router,
                                {"retriever": "retriever", "executor": "executor",
                                 "finalize": "finalize"})
        g.add_edge("retriever", "finalize")
        g.add_conditional_edges("executor", bridge.route_after_executor,
                                {"human_gate": "human_gate", "executor": "executor",
                                 "finalize": "finalize"})
        g.add_conditional_edges("human_gate", bridge.route_after_gate,
                                {"executor": "executor", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    def _run_native(self, message: str, *, user_id: str, history: list[dict] | None) -> AgentState:
        initial = AgentState(messages=(history or []) + [{"role": "user", "content": message}])
        payload = initial.to_dict()
        payload["_user_id"] = user_id
        out = self._compiled_langgraph.invoke(payload)  # type: ignore[union-attr]
        return _dict_to_state(out)

    # ------------------------------------------------------------------
    @staticmethod
    def _json_or_default(text: str, default: dict) -> dict:
        parsed = parse_tool_call(text) if "tool" in (text or "") else None
        if parsed:
            return parsed
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return default

    def _next_pending_call(self, st: AgentState) -> dict | None:
        for c in st.tool_calls:
            if c.get("tool"):
                return c
        return None

    def _direct_answer(self, question: str) -> str:
        prompt = ("Answer briefly and helpfully. Memory: "
                  + json.dumps(self._mem_hint[:6], ensure_ascii=False) if hasattr(self, "_mem_hint") else "")
        return (self.llm(f"{prompt}\nQuestion: {question}\nAnswer:") or "").strip()[:600] or \
               "Hello! How can I help with your documents or workflows today?"


def _dict_to_state(d: dict) -> AgentState:
    st = AgentState()
    for k in AgentState.__dataclass_fields__:
        if k in d:
            setattr(st, k, d[k])
    return st


class _StateBridge:
    """Adapts dict-based LangGraph state onto the AgentState node methods."""

    def __init__(self, graph: "AgentGraph"):
        self.g = graph

    def _wrap(self, d: dict):
        st = _dict_to_state(d)
        st.__dict__["_user_id"] = d.get("_user_id", "anon")
        return st

    def load_memory(self, d: dict):
        st = self._wrap(d)
        st = self.g.node_load_memory(st, d.get("_user_id", "anon"))
        return st.to_dict() | {"_user_id": d.get("_user_id")}

    def router(self, d: dict):
        st = self._wrap(d)
        st = self.g.node_router(st)
        out = st.to_dict()
        out["_router_decision"] = getattr(st, "_router_decision", "retrieve")
        return out

    def route_after_router(self, d: dict) -> str:
        dec = d.get("_router_decision", "retrieve")
        return {"answer": "finalize", "retrieve": "retriever"}.get(dec, "executor")

    def retriever(self, d: dict):
        return self.g.node_retriever(self._wrap(d)).to_dict()

    def executor(self, d: dict):
        return self.g.node_executor(self._wrap(d)).to_dict()

    def route_after_executor(self, d: dict) -> str:
        if d.get("approval_status") == "pending":
            return "human_gate"
        if d.get("answer"):
            return "finalize"
        return "executor"

    def human_gate(self, d: dict):
        from langgraph.types import interrupt

        decision = interrupt({"action": d.get("pending_action"),
                              "question": "Approve this action?"})
        out = dict(d)
        out["approval_status"] = "approved" if decision in (True, "approve", "yes") else "rejected"
        return out

    def route_after_gate(self, d: dict) -> str:
        return "executor" if d.get("approval_status") == "approved" else "finalize"

    def finalize(self, d: dict):
        return self.g.node_finalize(self._wrap(d)).to_dict()


# ---------------------------------------------------------------------------
# Slack approval webhook (Phase 3 step 5)
# ---------------------------------------------------------------------------
def slack_approval_notifier(payload: dict) -> None:
    """Push an Approve/Reject interactive message to the configured Slack webhook."""
    url = settings.agent.slack_webhook_url
    action = payload.get("action", {})
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "🤖 Agent approval required"}},
        {"type": "section", "text": {"type": "mrkdwn",
         "text": f"*Tool:* `{action.get('tool')}`\n*Args:* ```{json.dumps(action.get('args', {}))}```"}},
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "✅ Approve"},
             "style": "primary", "action_id": "approve"},
            {"type": "button", "text": {"type": "plain_text", "text": "❌ Reject"},
             "style": "danger", "action_id": "reject"},
        ]},
    ]
    if not url:
        logger.info("Slack webhook not configured; approval parked locally: %s",
                    json.dumps(action)[:200])
        return
    try:
        import httpx

        httpx.post(url, json={"blocks": blocks, "text": "Agent approval required"}, timeout=10)
    except Exception as exc:
        logger.warning("Slack notify failed: %s", exc)
