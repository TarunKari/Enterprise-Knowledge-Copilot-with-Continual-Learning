"""Phase 4 – The continual-learning flywheel: feedback store + SFT/DPO regeneration.

* ``FeedbackStore`` persists every interaction and thumbs-up/down rating
  (SQLite by default, Postgres-ready).
* ``build_training_batch`` converts highly-rated interactions into fresh
  ``sft.jsonl`` rows and low-rated-vs-high pairs into ``preference.jsonl`` rows,
  which the Airflow DAG feeds straight back into Phase-1 training.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from configs.settings import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, user_id TEXT, question TEXT, answer TEXT,
    contexts_json TEXT, citations_json TEXT, meta_json TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER REFERENCES interactions(id),
    rating TEXT CHECK (rating IN ('up','down')),
    comment TEXT, user_id TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_fb_interaction ON feedback(interaction_id);
"""


@dataclass
class StoredInteraction:
    interaction_id: int


class FeedbackStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.platform.feedback_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    def log_interaction(self, *, question: str, answer: str, user_id: str = "anon",
                        session_id: str = "", contexts: list[dict] | None = None,
                        citations: list[str] | None = None, meta: dict | None = None) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO interactions (session_id,user_id,question,answer,contexts_json,"
                "citations_json,meta_json,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (session_id, user_id, question, answer,
                 json.dumps(contexts or []), json.dumps(citations or []),
                 json.dumps(meta or {}), time.time()))
            return int(cur.lastrowid)

    def rate(self, interaction_id: int, rating: str, *, comment: str = "",
             user_id: str = "anon") -> bool:
        if rating not in ("up", "down"):
            return False
        with self._conn() as c:
            ok = c.execute("SELECT 1 FROM interactions WHERE id=?", (interaction_id,)).fetchone()
            if not ok:
                return False
            c.execute("INSERT INTO feedback (interaction_id,rating,comment,user_id,created_at)"
                      " VALUES (?,?,?,?,?)", (interaction_id, rating, comment, user_id, time.time()))
        return True

    # ------------------------------------------------------------------
    def export_labeled(self, since_ts: float = 0.0) -> list[dict]:
        """Interactions joined with their net rating; one row per interaction."""
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                """SELECT i.id, i.session_id, i.user_id, i.question, i.answer,
                          i.contexts_json, i.citations_json, i.meta_json, i.created_at,
                          COALESCE(SUM(CASE f.rating WHEN 'up' THEN 1 ELSE -1 END),0) AS score,
                          GROUP_CONCAT(f.comment) AS comments
                   FROM interactions i LEFT JOIN feedback f ON f.interaction_id = i.id
                   WHERE i.created_at >= ?
                   GROUP BY i.id ORDER BY i.created_at""",
                (since_ts,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Dataset regeneration
# ---------------------------------------------------------------------------
def build_training_batch(db_path: str | None = None, out_dir: str = "data/processed/flywheel",
                         min_score_up: int = 1, max_examples: int = 5000) -> dict:
    store = FeedbackStore(db_path)
    rows = store.export_labeled()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ups = [r for r in rows if r["score"] >= min_score_up]
    downs = [r for r in rows if r["score"] <= -1]

    sft_rows = [{"instruction": r["question"],
                 "input": _context_input(json.loads(r["contexts_json"] or "[]")),
                 "output": r["answer"]} for r in ups][-max_examples:]
    with (out_dir / "sft.jsonl").open("w", encoding="utf-8") as fh:
        for rec in sft_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pref_rows = _make_pairs(ups, downs)[:max_examples]
    with (out_dir / "preference.jsonl").open("w", encoding="utf-8") as fh:
        for rec in pref_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {"interactions": len(rows), "thumbs_up": len(ups), "thumbs_down": len(downs),
             "sft_written": len(sft_rows), "preference_written": len(pref_rows),
             "out_dir": str(out_dir)}
    (out_dir / "batch_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _context_input(contexts: list[dict]) -> str:
    return "\n\n".join(c.get("text", "")[:600] for c in contexts[:3])


def _make_pairs(ups: list[dict], downs: list[dict]) -> list[dict]:
    """Pair similar questions answered well vs poorly (token-overlap matching)."""
    import re

    def toks(q: str) -> set[str]:
        return set(re.findall(r"\w+", q.lower()))

    pairs = []
    used_downs = set()
    for up in ups:
        best, best_j = None, 0.0
        ut = toks(up["question"])
        for i, dn in enumerate(downs):
            if i in used_downs:
                continue
            dt = toks(dn["question"])
            j = len(ut & dt) / max(1, len(ut | dt))
            if j > best_j:
                best, best_j = i, j
        if best is not None and best_j >= 0.3:
            used_downs.add(best)
            pairs.append({"prompt": up["question"], "chosen": up["answer"],
                          "rejected": downs[best]["answer"]})
    # Fallback generic pairs so DPO always has signal when few matched pairs exist
    for up in ups[:50]:
        if len(pairs) >= 200:
            break
        pairs.append({"prompt": up["question"], "chosen": up["answer"],
                      "rejected": "I don't have enough information to answer that request."})
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate SFT/DPO datasets from feedback.")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default="data/processed/flywheel")
    args = ap.parse_args()
    print(json.dumps(build_training_batch(args.db, args.out), indent=2))


if __name__ == "__main__":
    main()
