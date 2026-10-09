"""Phase 2 – RAG quality evaluation with RAGAS + golden test set.

Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
If the ``ragas`` package / judge LLM is unavailable, a dependency-free surrogate
metric suite (lexical overlap based) still runs so CI always produces numbers.

Golden set format (JSONL):
    {"question": "...", "ground_truth": "...", "contexts": ["...", "..."]}
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from src.rag.pipeline import RagPipeline


# ---------------------------------------------------------------------------
# Surrogate metrics (no external deps)
# ---------------------------------------------------------------------------
def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower())) - {"the", "a", "an", "of", "and", "to", "is", "in"}


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences supported lexically by retrieved context."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.split()) > 4]
    if not sents:
        return 0.0
    ctx_tokens = set().union(*[_tokens(c) for c in contexts]) if contexts else set()
    supported = sum(1 for s in sents if len(_tokens(s) & ctx_tokens) >= max(1, 0.5 * len(_tokens(s))))
    return round(supported / len(sents), 4)


def answer_relevancy(question: str, answer: str) -> float:
    qt, at = _tokens(question), _tokens(answer)
    if not qt or not at:
        return 0.0
    coverage = len(qt & at) / len(qt)
    length_ok = 1.0 if 8 <= len(answer.split()) <= 250 else 0.6
    return round(min(1.0, coverage * 1.6) * length_ok, 4)


def context_precision(question: str, ground_truth: str, contexts: list[str]) -> float:
    """Average precision: rank-weighted hit rate of contexts containing GT evidence."""
    gt_toks = _tokens(ground_truth)
    hits = []
    for c in contexts:
        ct = _tokens(c)
        frac = len(gt_toks & ct) / max(1, len(gt_toks))
        hits.append(frac >= 0.35)
    if not hits or not any(hits):
        return 0.0
    score, rel = 0.0, 0
    for i, h in enumerate(hits, 1):
        if h:
            rel += 1
            score += rel / i
    return round(score / max(1, min(rel, len(gt_toks) and sum(hits))), 4)


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    gt_toks = _tokens(ground_truth)
    if not gt_toks:
        return 0.0
    covered = set().union(*[_tokens(c) for c in contexts]) if contexts else set()
    return round(len(gt_toks & covered) / len(gt_toks), 4)


# ---------------------------------------------------------------------------
# RAGAS-based evaluation (preferred when installed)
# ---------------------------------------------------------------------------
def evaluate_with_ragas(samples: list[dict]) -> dict | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (AnswerRelevancy, ContextPrecision,
                                   ContextRecall, Faithfulness)
    except ImportError:
        return None
    rows = [{
        "question": s["question"],
        "answer": s.get("predicted_answer", ""),
        "contexts": s["contexts"],
        "ground_truth": s["ground_truth"],
    } for s in samples]
    result = evaluate(Dataset.from_list(rows),
                      metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()])
    return {k: float(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_evaluation(golden_path: str, *, pipeline: RagPipeline | None = None,
                   use_pipeline: bool = True, out_report: str | None = None) -> dict:
    samples = [json.loads(l) for l in Path(golden_path).open(encoding="utf-8") if l.strip()]
    pipeline = pipeline or RagPipeline(use_postgres=False)

    evaluated = []
    for s in samples:
        if use_pipeline and "predicted_answer" not in s:
            res = pipeline.answer(s["question"])
            s = {**s, "predicted_answer": res.answer,
                 "contexts": [c["text"] for c in res.contexts]}
        pred = s.get("predicted_answer", "")
        evaluated.append({
            "question": s["question"],
            "faithfulness": faithfulness(pred, s["contexts"]),
            "answer_relevancy": answer_relevancy(s["question"], pred),
            "context_precision": context_precision(s["question"], s["ground_truth"], s["contexts"]),
            "context_recall": context_recall(s["ground_truth"], s["contexts"]),
        })

    aggregate = {m: round(statistics.mean(e[m] for e in evaluated), 4)
                 for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}

    ragas_scores = evaluate_with_ragas([{**s, "predicted_answer": s.get("predicted_answer", "")}
                                        for s in samples]) if samples else None
    report = {
        "n_samples": len(samples),
        "surrogate_metrics": aggregate,
        "per_sample": evaluated,
        "ragas_metrics": ragas_scores,
        "pass_thresholds": {"faithfulness": 0.7, "context_precision": 0.6,
                            "answer_relevancy": 0.7},
    }
    report["baseline_pass"] = all(
        aggregate[k] >= v for k, v in report["pass_thresholds"].items()
    )
    if out_report:
        Path(out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(out_report).write_text(json.dumps(report, indent=2))
    return report


def make_golden_template(out_path: str = "data/processed/golden.jsonl") -> int:
    template = [
        {
            "question": "How do I request paid time off?",
            "ground_truth": "Submit a PTO request in the HR portal at least 5 business days in advance; manager approval required.",
            "contexts": ["Employees must submit PTO requests via the HR portal at least five business days ahead. Direct manager approval is required before dates are locked."],
        },
        {
            "question": "What is the VPN reconnect procedure?",
            "ground_truth": "Run 'vpn-cli reconnect', then verify with status command; contact IT if it fails twice.",
            "contexts": ["VPN troubleshooting: run `vpn-cli reconnect`, check `vpn-cli status`. If two consecutive failures occur, file an IT ticket."],
        },
    ]
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(t) for t in template) + "\n")
    return len(template)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline on a golden set.")
    ap.add_argument("--golden", default="data/processed/golden.jsonl")
    ap.add_argument("--report", default="data/processed/rag_eval_report.json")
    ap.add_argument("--offline", action="store_true", help="Score provided predictions only.")
    ap.add_argument("--make-template", action="store_true")
    args = ap.parse_args()

    if args.make_template:
        n = make_golden_template(args.golden)
        print(f"Wrote {n} golden examples to {args.golden}")
        return
    report = run_evaluation(args.golden, use_pipeline=not args.offline, out_report=args.report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
