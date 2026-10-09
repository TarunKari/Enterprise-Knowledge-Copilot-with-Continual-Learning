"""Central configuration for the entire LLM + RAG + Agent platform.

All values can be overridden via environment variables (12-factor style) or a
``.env`` file located at the repository root.  Nothing here requires a GPU; the
heavy defaults only matter when the corresponding Phase-1 scripts run on a
CUDA machine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default):
    raw = os.getenv(key)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


# ---------------------------------------------------------------------------
# Phase 1 – Fine-tuning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FinetuneConfig:
    base_model: str = _env("FT_BASE_MODEL", "meta-llama/Llama-3-8B")
    # 4-bit NormalFloat quantization knobs (bitsandbytes)
    load_in_4bit: bool = _env("FT_LOAD_IN_4BIT", True)
    bnb_4bit_quant_type: str = _env("FT_BNB_QUANT_TYPE", "nf4")
    bnb_4bit_use_double_quant: bool = _env("FT_BNB_DOUBLE_QUANT", True)
    bnb_4bit_compute_dtype: str = _env("FT_BNB_COMPUTE_DTYPE", "bfloat16")

    # LoRA (peft)
    lora_r: int = _env("LORA_R", 16)
    lora_alpha: int = _env("LORA_ALPHA", 32)
    lora_dropout: float = _env("LORA_DROPOUT", 0.05)
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    # SFT (trl.SFTTrainer)
    sft_dataset_path: str = _env("SFT_DATASET_PATH", str(REPO_ROOT / "data/processed/sft.jsonl"))
    max_seq_length: int = _env("MAX_SEQ_LENGTH", 2048)
    per_device_train_batch_size: int = _env("BSZ", 4)
    gradient_accumulation_steps: int = _env("GRAD_ACCUM", 4)
    learning_rate: float = _env("LR", 2e-4)
    num_epochs: float = _env("NUM_EPOCHS", 3.0)
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = _env("SAVE_STEPS", 100)
    save_total_limit: int = 3
    gradient_checkpointing: bool = _env("GRAD_CKPT", True)
    output_dir: str = _env("FT_OUTPUT_DIR", str(REPO_ROOT / "checkpoints/sft"))

    # DPO (trl.DPOTrainer)
    preference_dataset_path: str = _env(
        "PREF_DATASET_PATH", str(REPO_ROOT / "data/processed/preference.jsonl")
    )
    dpo_beta: float = _env("DPO_BETA", 0.1)
    dpo_output_dir: str = _env("DPO_OUTPUT_DIR", str(REPO_ROOT / "checkpoints/dpo"))

    # MLflow experiment tracking
    mlflow_tracking_uri: str = _env("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment: str = _env("MLFLOW_EXPERIMENT", "llm-platform-finetuning")


# ---------------------------------------------------------------------------
# Phase 2 – RAG / Vector search
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RagConfig:
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    embed_dim: int = _env("EMBED_DIM", 1024)

    chunk_size_tokens: int = _env("CHUNK_SIZE_TOKENS", 500)
    chunk_overlap_tokens: int = _env("CHUNK_OVERLAP_TOKENS", 50)

    dense_top_k: int = _env("DENSE_TOP_K", 50)
    sparse_top_k: int = _env("SPARSE_TOP_K", 50)
    rrf_k: int = _env("RRF_K", 60)
    fuse_top_n: int = _env("FUSE_TOP_N", 20)
    rerank_top_k: int = _env("RERANK_TOP_K", 5)

    # Postgres / pgvector
    database_url: str = _env(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/ragdb",
    )
    hnsw_m: int = _env("HNSW_M", 16)
    hnsw_ef_construction: int = _env("HNSW_EF_CONSTRUCTION", 64)

    hyde_enabled: bool = _env("HYDE_ENABLED", True)
    query_rewrite_variants: int = _env("QUERY_REWRITE_VARIANTS", 3)


# ---------------------------------------------------------------------------
# Phase 3 – Agent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentConfig:
    router_model: str = _env("ROUTER_MODEL", "gpt-4o-mini")
    inference_base_url: str = _env("LLM_INFERENCE_BASE_URL", "http://localhost:8001/v1")
    inference_api_key: str = _env("LLM_INFERENCE_API_KEY", "EMPTY")
    max_iterations: int = _env("AGENT_MAX_ITERATIONS", 8)
    tool_max_retries: int = _env("TOOL_MAX_RETRIES", 2)
    slack_webhook_url: str = _env("SLACK_APPROVAL_WEBHOOK_URL", "")
    require_approval_for_writes: bool = _env("AGENT_APPROVAL_REQUIRED", True)


# ---------------------------------------------------------------------------
# Phase 4 – API / observability / continual learning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlatformConfig:
    api_host: str = _env("API_HOST", "0.0.0.0")
    api_port: int = _env("API_PORT", 8000)
    langsmith_project: str = _env("LANGSMITH_PROJECT", "llm-platform")
    langfuse_host: str = _env("LANGFUSE_HOST", "http://localhost:3000")
    feedback_db_path: str = _env("FEEDBACK_DB_PATH", str(REPO_ROOT / "data/processed/feedback.db"))
    airflow_dags_folder: str = _env("AIRFLOW_DAGS_FOLDER", str(REPO_ROOT / "docker/airflow/dags"))


@dataclass(frozen=True)
class Settings:
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    platform: PlatformConfig = field(default_factory=PlatformConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():  # optional dependency: python-dotenv
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv)
        except ImportError:
            pass
    return Settings()


settings = get_settings()
