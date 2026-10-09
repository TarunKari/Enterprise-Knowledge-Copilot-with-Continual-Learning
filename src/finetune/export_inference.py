"""Phase 1 – Inference optimization: LoRA merge, quantized export, vLLM/GGUF serving.

Three deliverables:
1. ``merge_adapter``      – fold LoRA weights into the base model (PEFT merge_and_unload)
2. ``export_gguf``        – convert merged HF checkpoint to GGUF for llama.cpp / Ollama
3. ``vllm_serve_config``  – emit a ready-to-run vLLM launch spec + OpenAI-compatible proxy

Also provides ``dynamic_switch`` so multiple task adapters can share one resident
base model at serving time instead of merging everything.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from configs.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_inference")


# ---------------------------------------------------------------------------
# 1. Merge LoRA -> base
# ---------------------------------------------------------------------------
def merge_adapter(base_model: str, adapter_dir: str, out_dir: str,
                  dtype: str = "bfloat16", use_4bit_base: bool = False) -> str:
    """Merge a trained PEFT adapter into its base weights and save full precision."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    logger.info("Loading base %s (merge requires unquantized weights)", base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
    if use_4bit_base:
        logger.warning("Merging over a 4-bit base is unsupported; ensure --keep-adapter for that setup.")
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    logger.info("Merged checkpoint saved to %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# 2. GGUF export (llama.cpp / Ollama)
# ---------------------------------------------------------------------------
GGUF_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "F16"]


def export_gguf(hf_dir: str, out_gguf: str, qtype: str = "Q4_K_M",
                llama_cpp_repo: str | None = None) -> dict:
    """Convert an HF checkpoint to GGUF then quantize with llama.cpp tools.

    Requires the llama.cpp repo (cloned automatically if absent) and its
    ``convert_hf_to_gguf.py`` + ``build/bin/llama-quantize`` binaries.
    """
    repo = Path(llama_cpp_repo or "vendor/llama.cpp")
    steps = []
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(repo)],
            check=True,
        )
        steps.append("cloned llama.cpp")
    f32 = str(Path(out_gguf).with_suffix("")) + "-f32.gguf"
    subprocess.run(
        ["python", str(repo / "convert_hf_to_gguf.py"), hf_dir, "--outfile", f32, "--outtype", "f32"],
        check=True,
    )
    steps.append(f"converted {hf_dir} -> {f32}")
    quantize = repo / "build" / "bin" / "llama-quantize"
    if quantize.exists():
        subprocess.run([str(quantize), f32, out_gguf, qtype], check=True)
        steps.append(f"quantized -> {qtype}")
    else:
        logger.warning("llama-quantize binary missing; run `cmake -B build && cmake --build build` in llama.cpp")
        out_gguf = f32
    meta = {"gguf_path": out_gguf, "quant": qtype, "steps": steps}
    Path(out_gguf + ".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# 3. vLLM serving spec + dynamic multi-adapter switching
# ---------------------------------------------------------------------------
def vllm_serve_config(model_dir: str, *, tp_size: int = 1, max_model_len: int = 4096,
                      gpu_mem_util: float = 0.90, quantization: str | None = None,
                      port: int = 8001) -> dict:
    """Emit the exact command + flags to serve the exported model via vLLM's
    OpenAI-compatible server (continuous batching, paged attention)."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--port", str(port),
    ]
    if quantization:  # e.g. "awq", "gptq", "fp8" for further inference-time compression
        cmd += ["--quantization", quantization]
    return {
        "command": " ".join(cmd),
        "openai_base_url": f"http://localhost:{port}/v1",
        "notes": "Set LLM_INFERENCE_BASE_URL to openai_base_url for the agent/API layer.",
    }


def dynamic_switch_example() -> dict:
    """Keep ONE quantized base resident and hot-swap LoRA adapters per tenant/task."""
    return {
        "pattern": "peft.PeftModel.switch_adapter",
        "code": (
            "from peft import PeftModel\n"
            "model = PeftModel.from_pretrained(base, 'adapters/sales', adapter_name='sales')\n"
            "model.load_adapter('adapters/hr', adapter_name='hr')\n"
            "model.switch_adapter(['hr'], activation_names=['hr'])  # zero-copy route per request\n"
        ),
        "when": "multiple domain adapters, single GPU, low latency budget",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge/export optimized inference artifacts")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge", help="Merge LoRA adapter into base model")
    m.add_argument("--base", default=settings.finetune.base_model)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--dtype", default="bfloat16")

    g = sub.add_parser("gguf", help="Export merged HF dir to quantized GGUF")
    g.add_argument("--hf-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--quant", default="Q4_K_M", choices=GGUF_QUANTS)

    v = sub.add_parser("vllm", help="Print vLLM serving configuration")
    v.add_argument("--model-dir", required=True)
    v.add_argument("--tp", type=int, default=1)
    v.add_argument("--quantization", default=None)

    args = ap.parse_args()
    if args.cmd == "merge":
        print(json.dumps({"merged_dir": merge_adapter(args.base, args.adapter, args.out, args.dtype)}, indent=2))
    elif args.cmd == "gguf":
        print(json.dumps(export_gguf(args.hf_dir, args.out, args.quant), indent=2))
    elif args.cmd == "vllm":
        cfg = vllm_serve_config(args.model_dir, tp_size=args.tp, quantization=args.quantization)
        cfg["dynamic_multi_adapter"] = dynamic_switch_example()
        print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
