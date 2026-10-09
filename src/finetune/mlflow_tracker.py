"""MLflow experiment tracking helpers (loss + VRAM metrics, artifacts, model registry).

All functions degrade gracefully when ``mlflow`` is not installed or the tracking
server is unreachable, so training never crashes because of observability.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from configs.settings import settings

logger = logging.getLogger(__name__)


def _mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        logger.warning("mlflow not installed; tracking disabled.")
        return None


@contextmanager
def tracked_run(run_name: str, tags: dict | None = None) -> Iterator[object]:
    """Context manager wrapping an MLflow run. Yields the active Run object (or None)."""
    mlflow = _mlflow()
    if mlflow is None:
        yield None
        return
    try:
        mlflow.set_tracking_uri(settings.finetune.mlflow_tracking_uri)
        mlflow.set_experiment(settings.finetune.mlflow_experiment)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    except Exception as exc:  # tracking server down etc. – never break training
        logger.warning("MLflow unavailable (%s); continuing without tracking.", exc)
        yield None


def log_params(params: dict) -> None:
    mlflow = _mlflow()
    if mlflow is None:
        return
    try:
        mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})
    except Exception as exc:
        logger.debug("log_params failed: %s", exc)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log scalars such as train_loss / eval_loss / vram_allocated_gb."""
    mlflow = _mlflow()
    if mlflow is None:
        return
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    try:
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:
        logger.debug("log_metrics failed: %s", exc)


def log_artifact(path: str) -> None:
    mlflow = _mlflow()
    if mlflow is None:
        return
    try:
        mlflow.log_artifact(path)
    except Exception as exc:
        logger.debug("log_artifact failed: %s", exc)


def register_model(local_dir: str, model_name: str, version_alias: str = "staging") -> str | None:
    """Register a HF-format checkpoint dir into the MLflow Model Registry."""
    mlflow = _mlflow()
    if mlflow is None:
        return None
    try:
        from mlflow.models import infer_signature  # optional

        uri = f"models/{model_name}"
        info = mlflow.transformers.log_model(
            transformers_model=local_dir,
            artifact_path=model_name,
            registered_model_name=model_name,
        )
        mlflow.models.set_registered_model_alias(model_name, version_alias, info.model_id)
        return info.model_id
    except Exception as exc:
        logger.warning("Model registration failed: %s", exc)
        return None


class MlflowMetricsCallback:
    """transformers TrainerCallback that streams loss/lr/VRAM to MLflow every log step.

    Instantiated lazily inside training scripts so this module imports cleanly even
    when ``transformers`` callbacks are unavailable.
    """

    def __new__(cls):
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if not logs:
                    return
                from src.finetune.model_loading import gpu_memory_report

                payload = dict(logs)
                payload.update(gpu_memory_report())
                log_metrics(payload, step=state.global_step)

            def on_epoch_end(self, args, state, control, **kwargs):
                log_metrics({"epoch_completed": state.epoch}, step=state.global_step)

        return _Impl()
