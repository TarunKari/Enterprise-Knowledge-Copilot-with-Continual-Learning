"""Base-model loading with 4-bit NormalFloat (NF4) quantization + double quantization.

Wraps ``transformers`` + ``bitsandbytes`` so the rest of the training code never has
to think about VRAM.  Falls back to a plain bf16/fp16 load when bitsandbytes/CUDA is
unavailable (e.g. CI machines), and exposes helpers for LoRA injection and memory
reporting that are logged to MLflow by the trainers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.settings import settings

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class ModelBundle:
    """Everything downstream stages need: model, tokenizer and provenance info."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    quantized: bool
    quantization_config: dict = field(default_factory=dict)
    model_name: str = ""

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)


def build_bnb_config() -> BitsAndBytesConfig | None:
    """4-bit NF4 + double-quant + compute dtype from settings (None if no CUDA/bnb)."""
    cfg = settings.finetune
    if not cfg.load_in_4bit:
        return None
    try:
        import bitsandbytes as bnb  # noqa: F401  (probe availability only)

        if not torch.cuda.is_available():
            logger.warning("bitsandbytes present but no CUDA device; skipping 4-bit quantization.")
            return None
    except Exception as exc:  # ImportError or CUDA-probe failure
        logger.warning("bitsandbytes unavailable (%s); loading model unquantized.", exc)
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,          # "nf4"
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,  # double quantization
        bnb_4bit_compute_dtype=_DTYPE_MAP.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16),
    )


def load_tokenizer(model_name: str | None = None) -> AutoTokenizer:
    name = model_name or settings.finetune.base_model
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        # Llama-family has no pad token; reuse EOS to avoid embedding resize surprises.
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


def load_model(
    model_name: str | None = None,
    *,
    use_4bit: bool = True,
    gradient_checkpointing: bool | None = None,
) -> ModelBundle:
    """Load base LLM with NF4 quantization (when possible) and prepare it for LoRA."""
    name = model_name or settings.finetune.base_model
    cfg = settings.finetune
    tokenizer = load_tokenizer(name)

    bnb_config = build_bnb_config() if use_4bit else None
    kwargs: dict = {"torch_dtype": _DTYPE_MAP.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16)}
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    logger.info("Loading %s (4-bit=%s)", name, bnb_config is not None)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.config.use_cache = False  # required w/ gradient checkpointing

    gc = cfg.gradient_checkpointing if gradient_checkpointing is None else gradient_checkpointing
    if gc:
        model.enable_input_require_grads()  # needed when base weights are frozen (LoRA)
        model.gradient_checkpointing_enable()

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        quantized=bnb_config is not None,
        quantization_config={
            "load_in_4bit": cfg.load_in_4bit,
            "bnb_4bit_quant_type": cfg.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": cfg.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": cfg.bnb_4bit_compute_dtype,
        }
        if bnb_config
        else {},
        model_name=name,
    )


def make_lora_config():
    """peft.LoraConfig targeting attention + MLP projections, r/alpha from settings."""
    from peft import LoraConfig, TaskType

    cfg = settings.finetune
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,                      # 16 or 32
        lora_alpha=cfg.lora_alpha,         # 32
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),  # q_proj,k_proj,v_proj,o_proj,gate/up/down
        bias="none",
    )


def attach_lora(bundle: ModelBundle):
    """Inject LoRA adapters into the loaded base model, returning a PeftModel."""
    from peft import get_peft_model, prepare_model_for_kbit_training

    model = bundle.model
    if bundle.quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=settings.finetune.gradient_checkpointing
        )
    peft_model = get_peft_model(model, make_lora_config())
    peft_model.print_trainable_parameters()
    return peft_model


def gpu_memory_report() -> dict:
    """VRAM metrics logged alongside loss curves in MLflow."""
    if not torch.cuda.is_available():
        return {"cuda_available": 0}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_available": 1,
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "vram_free_gb": round(free / 1e9, 3),
        "vram_total_gb": round(total / 1e9, 3),
        "max_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "params_total_M": round(total / 1e6, 1),
        "params_trainable_M": round(trainable / 1e6, 2),
        "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
    }
