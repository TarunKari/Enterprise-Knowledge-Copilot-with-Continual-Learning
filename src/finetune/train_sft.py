"""Phase 1 – Supervised Fine-Tuning with TRL's SFTTrainer.

Pipeline: NF4-quantized base model -> LoRA adapters -> SFTTrainer with
gradient checkpointing -> checkpoints every N steps -> loss/VRAM logged to MLflow.

Run on a GPU box:
    python -m src.finetune.train_sft --data data/processed/sft.jsonl
Smoke-test anywhere (tiny fake dataset, CPU, no download needed beyond tokenizer):
    python -m src.finetune.train_sft --smoke
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
from src.finetune.model_loading import attach_lora, count_parameters, load_model
from src.finetune.prompting import format_alpaca

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_sft")


def tokenize(examples, tokenizer, max_length: int):
    """Map-style formatter: Alpaca template -> input_ids/labels (mask prompt+pad)."""
    texts = [
        format_alpaca(instr, inp, out)
        for instr, inp, out in zip(
            examples["instruction"], examples.get("input", [""] * len(examples["instruction"])),
            examples["output"],
        )
    ]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    # Label masking: train only on the response span.
    labels = []
    for text, ids in zip(texts, enc["input_ids"]):
        prompt_only = format_alpaca(*_split_example(text)) if False else None  # noqa
        n_prompt = len(
            tokenizer(
                text.split("### Response:\n")[0] + "### Response:\n",
                truncation=True, max_length=max_length, add_special_tokens=False,
            )["input_ids"]
        )
        lab = [-100] * min(n_prompt, len(ids)) + list(ids[min(n_prompt, len(ids)):])
        labels.append(lab)
    enc["labels"] = labels
    return enc


def _split_example(text):  # helper kept for clarity/debugging
    head, _, resp = text.partition("### Response:\n")
    instr, _, inp = head.partition("\n\n### Input:\n")
    instr = instr.replace("### Instruction:\n", "").strip()
    return instr, inp.strip(), resp.strip()


def run_smoke_training(tmp_dir: str = "/tmp/sft_smoke") -> dict:
    """CPU-runnable mini end-to-end check: tiny model from scratch + LoRA + SFT loop."""
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    logger.info("SMOKE MODE: building a 3-layer GPT-2 surrogate (no HF download).")
    cfg = GPT2Config(vocab_size=512, n_positions=256, n_embd=64, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        raise RuntimeError("Need cached gpt2 tokenizer for smoke test; run online once.")

    lora = LoraConfig(r=settings.finetune.lora_r, lora_alpha=settings.finetune.lora_alpha,
                      target_modules=["c_attn"], lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    data = [
        {"instruction": f"Say hello like engineer {i}", "input": "", "output": f"Hello engineer {i}!"}
        for i in range(16)
    ]
    ds = load_dataset("json", data_files=json_temp(data), split="train")

    def fmt(ex):
        return {"text": format_alpaca(ex["instruction"], ex["input"], ex["output"])}

    ds = ds.map(fmt, remove_columns=ds.column_names)
    args = SFTConfig(
        output_dir=tmp_dir, max_steps=5, per_device_train_batch_size=2,
        gradient_checkpointing=True, learning_rate=1e-3, logging_steps=1,
        save_steps=5, report_to="none", use_cpu=True, bf16=False, fp16=False,
        dataset_text_field="text", max_seq_length=128,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    result = trainer.train()
    trainer.save_model(str(Path(tmp_dir) / "adapter"))
    metrics = {"train_loss": result.training_loss, "steps": 5}
    logger.info("Smoke SFT finished: %s", metrics)
    return metrics


def json_temp(records: list[dict]) -> str:
    import tempfile

    p = Path(tempfile.mkdtemp()) / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT with TRL + MLflow")
    ap.add_argument("--data", default=settings.finetune.sft_dataset_path)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--model", default=settings.finetune.base_model)
    ap.add_argument("--out", default=settings.finetune.output_dir)
    ap.add_argument("--epochs", type=float, default=settings.finetune.num_epochs)
    ap.add_argument("--lr", type=float, default=settings.finetune.learning_rate)
    ap.add_argument("--save-steps", type=int, default=settings.finetune.save_steps)
    ap.add_argument("--max-len", type=int, default=settings.finetune.max_seq_length)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--smoke", action="store_true", help="Tiny CPU run to validate plumbing.")
    args = ap.parse_args()

    if args.smoke:
        print(json.dumps(run_smoke_training(args.out), indent=2))
        return

    cfg = settings.finetune
    bundle = load_model(args.model, use_4bit=not args.no_4bit)
    peft_model = attach_lora(bundle)
    tokenizer = bundle.tokenizer

    train_ds = load_dataset("json", data_files=args.data, split="train")
    val_ds = None
    if args.val_data and Path(args.val_data).exists():
        val_ds = load_dataset("json", data_files=args.val_data, split="train")

    def fmt(ex):
        return {"text": format_alpaca(ex["instruction"], ex.get("input", ""), ex["output"])}

    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
    if val_ds is not None:
        val_ds = val_ds.map(fmt, remove_columns=val_ds.column_names)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        optim="paged_adamw_8bit" if bundle.quantized else "adamw_torch",
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",           # we log to MLflow ourselves via callback
        seed=42,
    )

    from trl import SFTTrainer

    callbacks = [mlt.MlflowMetricsCallback()] if mlt._mlflow() else []
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        callbacks=callbacks,
    )

    with mlt.tracked_run(f"sft-{Path(args.model).name}", tags={"phase": "1", "stage": "sft"}) as run:
        mlt.log_params({
            "base_model": args.model, "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
            "target_modules": ",".join(cfg.lora_target_modules),
            "quant_4bit_nf4": bundle.quantized, "double_quant": cfg.bnb_4bit_use_double_quant,
            "grad_checkpointing": cfg.gradient_checkpointing, "epochs": args.epochs, "lr": args.lr,
            **count_parameters(peft_model),
        })
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        mlt.log_metrics({"final_train_loss": result.training_loss})
        trainer.save_model(str(Path(args.out) / "final_adapter"))
        mlt.log_artifact(str(Path(args.out) / "final_adapter"))
        if run is not None:
            mlt.register_model(str(Path(args.out) / "final_adapter"),
                               model_name=f"{Path(args.model).name}-lora-sft", version_alias="staging")
        print(json.dumps({"final_train_loss": result.training_loss, "adapter": f"{args.out}/final_adapter"}, indent=2))


if __name__ == "__main__":
    main()
