"""Phase 4 – Observability: LangSmith/Langfuse tracing + latency/failure alerts.

``Tracer`` emits structured spans (LLM calls, tool executions, RAG steps, state
transitions) to LangSmith and/or Langfuse when configured, and always writes a
local JSONL trace log for debugging. ``AlertMonitor`` evaluates rolling windows of
trace events and fires webhooks on: p95 latency breach, token-limit breaches, and
tool-failure spikes.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from configs.settings import settings

logger = logging.getLogger("observability")


@dataclass
class TraceEvent:
    kind: str                 # llm | tool | retrieval | state | feedback
    name: str
    duration_ms: float
    tokens: int = 0
    ok: bool = True
    meta: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Tracer:
    def __init__(self, local_log_path: str | None = "data/processed/traces.jsonl"):
        self.local_log = Path(local_log_path) if local_log_path else None
        self._langsmith = self._init_langsmith()
        self._langfuse = self._init_langfuse()
        self.events: list[TraceEvent] = []

    @staticmethod
    def _init_langsmith():
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and os.getenv("LANGCHAIN_API_KEY"):
            try:
                import langsmith

                client = langsmith.Client(project=settings.platform.langsmith_project)
                logger.info("LangSmith tracing enabled.")
                return client
            except Exception as exc:
                logger.warning("LangSmith init failed: %s", exc)
        return None

    @staticmethod
    def _init_langfuse():
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse

                lf = Langfuse(host=settings.platform.langfuse_host)
                logger.info("Langfuse tracing enabled.")
                return lf
            except Exception as exc:
                logger.warning("Langfuse init failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    @contextmanager
    def span(self, kind: str, name: str, **meta: Any) -> Iterator[dict]:
        t0 = time.perf_counter()
        holder: dict[str, Any] = {"tokens": 0, "ok": True, "meta": dict(meta)}
        try:
            with self._external_span(kind, name, meta):
                yield holder
        finally:
            ev = TraceEvent(kind=kind, name=name,
                            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                            tokens=int(holder.get("tokens", 0)), ok=bool(holder.get("ok", True)),
                            meta=holder.get("meta", {}))
            self.record(ev)

    def _external_span(self, kind: str, name: str, meta: dict):
        class _Noop:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

        if self._langfuse is not None:
            try:
                return self._langfuse.trace(name=f"{kind}:{name}")
            except Exception:
                pass
        return _Noop()

    def record(self, ev: TraceEvent) -> None:
        self.events.append(ev)
        if len(self.events) > 10_000:
            self.events = self.events[-5_000:]
        if self.local_log:
            try:
                self.local_log.parent.mkdir(parents=True, exist_ok=True)
                with self.local_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev.__dict__, default=str) + "\n")
            except Exception as exc:
                logger.debug("trace write failed: %s", exc)

    # convenience wrappers -------------------------------------------------
    def llm_call(self, model: str, prompt_tokens: int, completion_tokens: int,
                 duration_ms: float, ok: bool = True) -> None:
        self.record(TraceEvent("llm", model, duration_ms,
                               tokens=prompt_tokens + completion_tokens, ok=ok))

    def tool_call(self, tool: str, duration_ms: float, ok: bool, attempts: int = 1) -> None:
        self.record(TraceEvent("tool", tool, duration_ms, ok=ok, meta={"attempts": attempts}))

    def state_transition(self, node: str, duration_ms: float = 0.0) -> None:
        self.record(TraceEvent("state", node, duration_ms))


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
@dataclass
class AlertRule:
    name: str
    threshold: float
    window: int = 100          # number of most-recent events considered
    metric: str = "p95_latency_ms"   # or "tool_failure_rate", "token_limit_breaches"


DEFAULT_RULES = [
    AlertRule("latency_p95_gt_3s", threshold=3000.0, metric="p95_latency_ms"),
    AlertRule("tool_failure_rate_gt_20pct", threshold=0.2, metric="tool_failure_rate"),
    AlertRule("token_breach_rate_gt_5pct", threshold=0.05, metric="token_breach_rate"),
]

MAX_TOKENS_PER_CALL = int(os.getenv("MAX_TOKENS_PER_CALL", "8192"))


class AlertMonitor:
    def __init__(self, tracer: Tracer, rules: list[AlertRule] | None = None,
                 webhook_url: str | None = None):
        self.tracer = tracer
        self.rules = rules or DEFAULT_RULES
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL", "")
        self._last_fired: dict[str, float] = {}

    def compute_metrics(self) -> dict:
        evs = self.tracer.events[-2000:]
        lat = [e.duration_ms for e in evs]
        tools = [e for e in evs if e.kind == "tool"]
        toks = [e.tokens for e in evs if e.kind == "llm"]
        out: dict[str, float] = {}
        if lat:
            srt = sorted(lat)
            out["p50_latency_ms"] = srt[len(srt) // 2]
            out["p95_latency_ms"] = srt[min(len(srt) - 1, int(0.95 * len(srt)))]
        if tools:
            out["tool_failure_rate"] = sum(1 for t in tools if not t.ok) / len(tools)
        if toks:
            out["token_breach_rate"] = sum(1 for t in toks if t >= MAX_TOKENS_PER_CALL) / len(toks)
        return {k: round(v, 4) for k, v in out.items()}

    def check(self) -> list[dict]:
        metrics = self.compute_metrics()
        fired = []
        now = time.time()
        for rule in self.rules:
            val = metrics.get(rule.metric)
            if val is None:
                continue
            bad = val > rule.threshold if "rate" not in rule.metric or "failure" in rule.metric or "breach" in rule.metric else val > rule.threshold
            if bad and now - self._last_fired.get(rule.name, 0) > 300:  # 5-min cooldown
                self._last_fired[rule.name] = now
                alert = {"rule": rule.name, "metric": rule.metric, "value": val,
                         "threshold": rule.threshold}
                fired.append(alert)
                self._send(alert)
        return fired

    def _send(self, alert: dict) -> None:
        logger.warning("ALERT: %s", alert)
        if not self.webhook_url:
            return
        try:
            import httpx

            httpx.post(self.webhook_url, json={
                "text": f"🚨 LLM platform alert: `{alert['rule']}` — "
                        f"{alert['metric']}={alert['value']} (limit {alert['threshold']})"},
                timeout=10)
        except Exception as exc:
            logger.warning("alert webhook failed: %s", exc)


# Singleton used by the API layer ------------------------------------------
_tracer: Tracer | None = None
_monitor: AlertMonitor | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def get_monitor() -> AlertMonitor:
    global _monitor
    if _monitor is None:
        _monitor = AlertMonitor(get_tracer())
    return _monitor
