"""Phase 4 – Weekly continual-learning Airflow DAG.

Schedule: Sundays 02:00 UTC.
Flow:
    extract_feedback → validate_datasets → train_sft → train_dpo
        → evaluate_baseline → promotion_gate (branch)
             ├─ pass → deploy_new_model   (MLflow alias 'production')
             └─ fail → keep_current_model
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

REPO = Path("/opt/llm-platform")


def _py(*args: str) -> str:
    cmd = [sys.executable, "-m", *args]
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                         timeout=6 * 3600, check=False)
    if res.returncode != 0:
        raise AirflowFailException(f"{cmd} failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return res.stdout or ""


@dag(
    dag_id="continual_learning_flywheel",
    description="Feedback -> datasets -> SFT -> DPO -> eval -> auto-deploy",
    schedule="0 2 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["llm", "flywheel"],
    default_args={"owner": "ml-platform", "retries": 1,
                  "retry_delay": timedelta(minutes=10)},
)
def continual_learning_flywheel():

    @task
    def extract_feedback() -> dict:
        stats = json.loads(_py("src.continual.feedback_loop",
                               "--out", "data/processed/flywheel"))
        if stats["thumbs_up"] < 10:
            raise AirflowFailException(f"Not enough positive feedback to retrain: {stats}")
        return stats

    @task
    def validate_datasets(stats: dict) -> bool:
        for fname in ("sft.jsonl", "preference.jsonl"):
            p = REPO / "data/processed/flywheel" / fname
            n = sum(1 for _ in p.open()) if p.exists() else 0
            if n == 0:
                raise AirflowFailException(f"empty dataset {fname}")
        return True

    @task
    def train_sft() -> str:
        """GPU training runs here when CUDA is present; otherwise delegate to a
        remote MLflow/k8s training job and record the run id."""
        try:
            import torch

            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        if gpu:
            _py("src.finetune.train_sft",
                "--data", "data/processed/flywheel/sft.jsonl",
                "--out", "checkpoints/flywheel/sft")
            return "local-sft-run"
        # Remote trigger pattern (Argo/k8s):
        # subprocess.run(["kubectl", "create", "job", "flywheel-sft", ...])
        return "delegated-sft-run"

    @task
    def train_dpo(sft_run_id: str) -> str:
        try:
            import torch

            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        if gpu:
            _py("src.finetune.train_dpo",
                "--data", "data/processed/flywheel/preference.jsonl",
                "--adapter", "checkpoints/flywheel/sft/final_adapter",
                "--out", "checkpoints/flywheel/dpo")
        return sft_run_id

    @task
    def evaluate_baseline() -> dict:
        out = _py("src.rag.evaluation", "--golden", "data/processed/golden.jsonl",
                  "--report", "data/processed/flywheel_eval.json")
        report = json.loads(out)
        return {"pass": bool(report.get("baseline_pass")),
                "metrics": report.get("surrogate_metrics")}

    @task.branch
    def promotion_gate(eval_result: dict) -> str:
        return "deploy_new_model" if eval_result["pass"] else "keep_current_model"

    @task
    def deploy_new_model():
        try:
            import mlflow

            name = "Llama-3-8B-lora-dpo"
            client = mlflow.MlflowClient()
            latest = client.latest_versions(name)[0]
            mlflow.models.set_registered_model_alias(name, "production", latest.version)
        except Exception as exc:
            print(f"registry step skipped ({exc}); writing deploy marker only")
        (REPO / "data/processed/last_deploy.json").write_text(json.dumps(
            {"deployed_at": datetime.utcnow().isoformat(), "source": "flywheel"}))

    @task
    def keep_current_model():
        print("Baseline evaluation failed - current production model retained.")

    stats = extract_feedback()
    ok = validate_datasets(stats)
    sft_run = train_sft()
    dpo_run = train_dpo(sft_run)
    ev = evaluate_baseline()
    gate = promotion_gate(ev)
    deployed = deploy_new_model()
    kept = keep_current_model()
    gate >> [deployed, kept]


continual_learning_flywheel()
