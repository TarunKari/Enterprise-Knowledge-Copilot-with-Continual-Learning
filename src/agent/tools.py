"""Phase 3 – Agent tools with structured (JSON-mode) arguments and safe execution.

Each tool declares a JSON schema; the Router LLM must emit
``{"tool": "...", "args": {...}}`` which is validated here before dispatch.
Execution is wrapped in retry/error-analysis logic (failure recovery, Phase 3 step 6).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("agent_tools")


@dataclass
class ToolResult:
    ok: bool
    name: str
    output: Any = None
    error: str | None = None
    attempts: int = 1
    retriable: bool = False
    analysis: str = ""      # agent-facing error analysis for parameter adjustment
    latency_ms: float = 0.0


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict            # JSON-schema fragment
    fn: Callable[..., Any]
    destructive: bool = False   # triggers Human Gate when True
    max_retries: int = 2


# ---------------------------------------------------------------------------
# JSON argument parsing (structured output)
# ---------------------------------------------------------------------------
def parse_tool_call(text: str) -> dict | None:
    """Extract {"tool","args"} from raw LLM output (handles code fences/prose)."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or [text]
    for blob in candidates:
        m = re.search(r"\{.*\}", blob, re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "tool" in obj:
            obj.setdefault("args", {})
            return obj
    return None


def validate_args(schema: dict, args: dict) -> tuple[bool, str]:
    """Tiny required/type validator (subset of JSON schema semantics)."""
    for req in schema.get("required", []):
        if req not in args or args[req] in (None, ""):
            return False, f"missing required argument '{req}'"
    for k, spec in schema.get("properties", {}).items():
        if k in args:
            t = spec.get("type")
            py = {"string": str, "integer": int, "number": (int, float),
                  "boolean": bool, "array": list, "object": dict}.get(t)
            if py and not isinstance(args[k], py):
                try:
                    args[k] = py(args[k])
                except Exception:
                    return False, f"argument '{k}' must be {t}"
    return True, ""


# ---------------------------------------------------------------------------
# Registry + resilient executor
# ---------------------------------------------------------------------------
class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def schemas_for_prompt(self) -> str:
        lines = []
        for t in self.tools.values():
            lines.append(json.dumps({
                "name": t.name, "description": t.description,
                "parameters": t.parameters, "destructive": t.destructive,
            }))
        return "\n".join(lines)

    def execute(self, name: str, args: dict) -> ToolResult:
        spec = self.tools.get(name)
        if spec is None:
            return ToolResult(False, name, error=f"unknown tool '{name}'",
                              analysis="Pick one of the registered tool names.")
        ok, msg = validate_args(spec.parameters, args)
        if not ok:
            return ToolResult(False, name, error=msg,
                              analysis=f"Fix arguments then retry: {msg}", retriable=True)
        last_err = ""
        t0 = time.perf_counter()
        for attempt in range(1, spec.max_retries + 2):
            try:
                out = spec.fn(**args)
                return ToolResult(True, name, output=out, attempts=attempt,
                                  latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("tool %s failed (attempt %d): %s", name, attempt, last_err)
                time.sleep(min(2 ** attempt * 0.05, 1.0))  # short backoff
        return ToolResult(
            False, name, error=last_err, attempts=spec.max_retries + 1, retriable=True,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            analysis=(f"Tool '{name}' failed after {spec.max_retries + 1} attempts with '{last_err}'. "
                      "Analyze the traceback: adjust parameters (e.g. shorter query, different filters) "
                      "and retry once, otherwise escalate to the user."),
        )


# ---------------------------------------------------------------------------
# Concrete tools
# ---------------------------------------------------------------------------
def build_default_registry(rag_pipeline=None, db_session_factory=None,
                           email_sink: Callable[[dict], None] | None = None) -> ToolRegistry:
    reg = ToolRegistry()

    # -- search_knowledge_base --------------------------------------------
    def search_kb(query: str, department: str = "", top_k: int = 5) -> list[dict]:
        if rag_pipeline is None:
            raise RuntimeError("RAG pipeline not initialized")
        filters = {"department": department} if department else None
        res = rag_pipeline.answer(query, filters=filters)
        return [{"text": c["text"][:400], "metadata": c["metadata"],
                 "score": r.score}
                for c, r in zip(res.contexts, _scores(res))]

    def _scores(res):
        from src.rag.retrieval import RetrievedChunk
        return getattr(res, "trace", {}).get("_chunks", []) or [RetrievedChunk("", "", 0, "", {})] * len(res.contexts)

    reg.register(ToolSpec(
        name="search_knowledge_base",
        description="Hybrid RAG search over enterprise documents; returns grounded chunks.",
        parameters={"type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "natural-language question"},
                        "department": {"type": "string", "description": "optional filter e.g. hr/it/finance"},
                        "top_k": {"type": "integer"}},
                    "required": ["query"]},
        fn=search_kb,
    ))

    # -- query_hr_api -------------------------------------------------------
    def query_hr_api(employee_id: str, field: str = "profile") -> dict:
        """Read-only HR record lookup (mock adapter — swap base URL for real Workday/BambooHR)."""
        allowed = {"profile", "leave_balance", "manager", "start_date"}
        if field not in allowed:
            raise ValueError(f"field must be one of {sorted(allowed)}")
        # In production: httpx.get(f"{HR_API}/employees/{employee_id}/{field}", headers=...)
        return {"employee_id": employee_id, "field": field,
                "value": f"<mock {field} for {employee_id}>", "source": "hr-api-mock"}

    reg.register(ToolSpec(
        name="query_hr_api",
        description="Look up read-only HR data for an employee id.",
        parameters={"type": "object",
                    "properties": {"employee_id": {"type": "string"},
                                   "field": {"type": "string"}},
                    "required": ["employee_id"]},
        fn=query_hr_api,
    ))

    # -- check_permissions ---------------------------------------------------
    def check_permissions(user_id: str, action: str) -> dict:
        roles = {}
        if db_session_factory is not None:
            try:
                from sqlalchemy import text as sa_text

                with db_session_factory() as s:
                    row = s.execute(sa_text(
                        "SELECT value FROM user_memory WHERE user_id=:u AND kind='role'"
                        " ORDER BY updated_at DESC LIMIT 1"), {"u": user_id}).fetchone()
                    if row:
                        roles[user_id] = row.value
            except Exception:
                pass
        role = roles.get(user_id, "employee")
        policy = {"admin": "*", "manager": {"read", "draft_email"},
                  "employee": {"read"}}
        allowed = policy.get(role, {"read"})
        ok = allowed == "*" or action in allowed
        return {"user_id": user_id, "role": role, "action": action, "allowed": ok}

    reg.register(ToolSpec(
        name="check_permissions",
        description="Verify whether a user may perform an action (read/write/email).",
        parameters={"type": "object",
                    "properties": {"user_id": {"type": "string"},
                                   "action": {"type": "string"}},
                    "required": ["user_id", "action"]},
        fn=check_permissions,
    ))

    # -- send_email (destructive → Human Gate) -------------------------------
    def send_email(to: str, subject: str, body: str) -> dict:
        sink = email_sink
        if sink is None:
            logger.info("[email-dry-run] to=%s subject=%s", to, subject)
            return {"queued": True, "dry_run": True, "to": to, "subject": subject}
        payload = {"to": to, "subject": subject, "body": body, "ts": time.time()}
        sink(payload)
        return {"queued": True, **payload}

    reg.register(ToolSpec(
        name="send_email",
        description="Send an email to a recipient. DESTRUCTIVE: requires human approval.",
        parameters={"type": "object",
                    "properties": {"to": {"type": "string"},
                                   "subject": {"type": "string"},
                                   "body": {"type": "string"}},
                    "required": ["to", "subject", "body"]},
        fn=send_email,
        destructive=True,
    ))

    # -- save_memory -----------------------------------------------------------
    def save_memory(user_id: str, key: str, value: str) -> dict:
        if db_session_factory is None:
            raise RuntimeError("database unavailable for memory writes")
        from sqlalchemy import text as sa_text

        with db_session_factory() as s:
            s.execute(sa_text(
                "INSERT INTO user_memory (user_id, kind, key, value) VALUES (:u,'preference',:k,:v)"),
                {"u": user_id, "k": key, "v": value})
            s.commit()
        return {"saved": True, "user_id": user_id, "key": key}

    reg.register(ToolSpec(
        name="save_memory",
        description="Persist a long-term user preference/fact.",
        parameters={"type": "object",
                    "properties": {"user_id": {"type": "string"},
                                   "key": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["user_id", "value"]},
        fn=save_memory,
    ))

    return reg
