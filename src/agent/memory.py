"""Phase 3 – Memory manager: short-term (conversation) + long-term (Postgres user_memory).

Long-term memory is loaded before planning and injected into prompts; the agent
writes back durable preferences via ``remember()``. Falls back to a JSON file when
Postgres is unavailable so single-user deployments still get persistence.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from configs.settings import settings

logger = logging.getLogger("memory")


class MemoryManager:
    def __init__(self, session_factory=None, fallback_path: str | None = None):
        self.session_factory = session_factory
        self.fallback_path = Path(fallback_path or "data/processed/user_memory.json")
        self._lock = threading.Lock()
        self._local: dict[str, list[dict]] = {}
        self._load_local()

    # ------------------------------------------------------------------
    def load(self, user_id: str, limit: int = 20) -> list[dict]:
        """Return long-term rows [{kind,key,value,updated_at}] most-recent-first."""
        if self.session_factory is not None:
            try:
                from sqlalchemy import text as sa_text

                with self.session_factory() as s:
                    rows = s.execute(sa_text(
                        "SELECT kind, key, value, updated_at FROM user_memory "
                        "WHERE user_id=:u ORDER BY weight DESC, updated_at DESC LIMIT :l"
                    ), {"u": user_id, "l": limit}).fetchall()
                return [{"kind": r.kind, "key": r.key, "value": r.value,
                         "updated_at": str(r.updated_at)} for r in rows]
            except Exception as exc:
                logger.warning("DB memory read failed (%s); using local store.", exc)
        return list(reversed(self._local.get(user_id, []))[:limit])

    def remember(self, user_id: str, value: str, *, key: str = "",
                 kind: str = "preference", weight: float = 1.0) -> None:
        if self.session_factory is not None:
            try:
                from sqlalchemy import text as sa_text

                with self.session_factory() as s:
                    s.execute(sa_text(
                        "INSERT INTO user_memory (user_id, kind, key, value, weight) "
                        "VALUES (:u,:knd,:k,:v,:w)"),
                        {"u": user_id, "knd": kind, "k": key, "v": value, "w": weight})
                    s.commit()
                return
            except Exception as exc:
                logger.warning("DB memory write failed (%s); using local store.", exc)
        with self._lock:
            self._local.setdefault(user_id, []).append(
                {"kind": kind, "key": key, "value": value, "weight": weight,
                 "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            self._save_local()

    def inject_into_prompt(self, user_id: str, max_items: int = 8) -> str:
        rows = self.load(user_id, limit=max_items)
        if not rows:
            return ""
        lines = [f"- ({r['kind']}) {r['key']}: {r['value']}" if r.get("key")
                 else f"- ({r['kind']}) {r['value']}" for r in rows]
        return "Known facts about this user:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def extract_preferences(messages: list[dict]) -> list[str]:
        """Very small heuristic extractor: 'I prefer X', 'my Y is Z' utterances."""
        found = []
        pat = re.compile(r"\b(i\s+prefer|my\s+\w+\s+is|always\s+(?:use|prefer)|i\s+like)\s+(.{5,120})", re.I)
        for m in messages:
            if m.get("role") == "user":
                for mm in pat.finditer(m.get("content", "")):
                    found.append(mm.group(0).strip())
        return found

    def capture_from_conversation(self, user_id: str, messages: list[dict]) -> int:
        prefs = self.extract_preferences(messages)
        for p in prefs:
            self.remember(user_id, p, kind="preference")
        return len(prefs)

    # ------------------------------------------------------------------
    def _load_local(self) -> None:
        if self.fallback_path.exists():
            try:
                self._local = json.loads(self.fallback_path.read_text())
            except Exception:
                self._local = {}

    def _save_local(self) -> None:
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path.write_text(json.dumps(self._local, indent=2))
