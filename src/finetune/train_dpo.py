"""Phase 1 – Direct Preference Optimization (DPO) alignment stage.

Loads the best SFT LoRA checkpoint as starting policy, freezes a reference copy
(TRL handles ref-model sharding), and trains with DPOTrainer on preference pairs:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

No separate reward model is needed — DPO optimizes the implicit-reward objective
directly, sharpening tone/conciseness and suppressing hallucinations.

GPU run:
    python -m src.finetune.train_dpo --adapter checkpoints/sft/final_adapter
CPU smoke test of the loss plumbing:
    python -m src.finetune.train_dpo --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainingArguments

from configs.settings import settings
from src.finetune import mlflow_tracker as mlt
from src.finetune.model_loading import attach_lora, load_model
from src.finetune.prompting import format_alpaca

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_dpo")


def to_dpo_rows(ex):
    """Map {prompt, chosen, rejected} -> conversational columns TRL's DPOTrainer expects."""
    return {
        "prompt": format_alpaca(ex["prompt"], ex.get("context", "")),
        "chosen": ex["chosen"].strip(),
        "rejected": ex["rejected"].strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO alignment with TRL + MLflow")
    ap.add_argument("--model", default=settings.finetune.base_model)
    ap.add_argument("--adapter", default=str(Path(settings.finetune.output_dir) / "final_adapter"),
                    help="Best SFT LoRA checkpoint to continue from.")
    ap.add_argument("--data", default=settings.finetune.preference_dataset_path)
    ap.add_argument("--out", default=settings.finetune.dpo_output_dir)
    ap.add_argument("--beta", type=float, default=settings.finetune.dpo_beta)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        print(json.dumps(run_smoke_dpo(args.out), indent=2))
        return

    cfg = settings.finetune
    bundle = load_model(args.model, use_4bit=not args.no_4bit)

    # Start from the trained SFT adapter; keep it trainable, add fresh ref weights.
    from peft import PeftModel

    if Path(args.adapter).exists():
        bundle.model = PeftModel.from_pretrained(bundle.model, args.adapter, is_trainable=True)
        logger.info("Resumed LoRA adapter from %s", args.adapter)
    else:
        logger.warning("Adapter %s missing – attaching fresh LoRA instead.", args.adapter)
        bundle.model = attach_lora(bundle)

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(to_dpo_rows, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=max(1, cfg.per_device_train_batch_size // 2),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        optim="paged_adamw_8bit" if bundle.quantized else "adamw_torch",
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=42,
    )

    from trl import DPOConfig, DPOTrainer

    dpo_args = DPOConfig(**{**training_args.__dict__, "beta": args.beta})

    trainer = DPOTrainer(
        model=bundle.model,
        ref_model=None,               # PEFT: TRL reuses base weights w/ adapters disabled as ref
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=bundle.tokenizer,
    )

    with mlt.tracked_run(f"dpo-{Path(args.model).name}", tags={"phase": "1", "stage": "dpo"}) as run:
        mlt.log_params({"base_model": args.model, "sft_adapter": args.adapter,
                        "beta": args.beta, "epochs": args.epochs, "lr": args.lr})
        result = trainer.train()
        mlt.log_metrics({"final_dpo_loss": result.training_loss})
        metrics = trainer.evaluate()
        mlt.log_metrics({f"eval_{k.split('/')[-1]}": v for k, v in metrics.items()
                         if isinstance(v, (int, float))})
        trainer.save_model(str(Path(args.out) / "final_dpo_adapter"))
        mlt.log_artifact(str(Path(args.out) / "final_dpo_adapter"))
        if run is not None:
            mlt.register_model(str(Path(args.out) / "final_dpo_adapter"),
                               model_name=f"{Path(args.model).name}-lora-dpo", version_alias="staging")
        print(json.dumps({"final_dpo_loss": result.training_loss,
                          "eval": metrics,
                          "adapter": f"{args.out}/final_dpo_adapter"}, indent=2))


def run_smoke_dpo(tmp_dir: str = "/tmp/dpo_smoke") -> dict:
    """Tiny CPU DPO run validating data-format + trainer wiring end-to-end."""
    from peft import LoraConfig
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    from trl import DPOConfig, DPOTrainer

    cfg = GPT2Config(vocab_size=512, n_positions=128, n_embd=64, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")

    rows = [
        {"prompt": format_alpaca(f"Question {i}?"),
         "chosen": f"Concise correct answer {i}.",
         "rejected": f"Well, question {i} is complicated and I might be wrong but maybe answer {i}!"}
        for i in range(12)
    ]
    import tempfile

    p = Path(tempfile.mkdtemp()) / "pref.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ds = load_dataset("json", data_files=str(p), split="train")

    args = DPOConfig(output_dir=tmp_dir, max_steps=3, per_device_train_batch_size=2,
                     gradient_checkpointing=False, learning_rate=1e-4, beta=0.1,
                     report_to="none", use_cpu=True, max_length=96, max_prompt_length=48)
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
                         processing_class=tokenizer, peft_config=lora)
    result = trainer.train()
    trainer.save_model(tmp_dir)
    return {"train_loss": result.training_loss, "steps": 3, "out": tmp_dir}


if __name__ == "__main__":
    main()
