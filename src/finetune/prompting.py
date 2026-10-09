"""Prompt formatting utilities shared by SFT training and inference.

The Alpaca-style template is intentionally simple so the same function can be used
to (a) build training texts and (b) format live prompts against the fine-tuned
checkpoint, guaranteeing train/inference consistency.
"""
from __future__ import annotations

SYSTEM_PROMPT_DEFAULT = (
    "You are a precise enterprise assistant. Answer strictly from the provided "
    "context when present, cite sources as [Source: <doc>, Page <n>], and say "
    "'I don't know' instead of guessing."
)

EOS_MARK = "<|end|>"


def format_alpaca(instruction, input_text="", output_text=None):
    """Render one Alpaca example; append ``output_text`` when building targets."""
    if input_text and input_text.strip():
        block = "### Instruction:\n" + instruction + "\n\n### Input:\n" + input_text + "\n\n### Response:\n"
    else:
        block = "### Instruction:\n" + instruction + "\n\n### Response:\n"
    if output_text is not None:
        return block + output_text.strip() + " " + EOS_MARK
    return block


def format_chatml(messages, add_generation_prompt=True):
    """ChatML rendering for chat-tuned models (Llama-3 style alternates fine too)."""
    out = []
    for m in messages:
        out.append("<|im_start|>" + m["role"] + "\n" + m["content"].strip() + "<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def build_rag_prompt(question, contexts, system=SYSTEM_PROMPT_DEFAULT):
    """Compose the final grounded prompt with numbered context blocks + citations hint."""
    blocks = []
    for i, c in enumerate(contexts, 1):
        meta = c.get("metadata", {})
        src = meta.get("source", f"chunk-{i}")
        page = meta.get("page") or meta.get("pages", "")
        tag = "[Source: " + src + (", Page " + str(page) + "]" if page != "" else "]")
        blocks.append("-----\nContext " + str(i) + " " + tag + "\n" + c["text"].strip())
    context_blob = "\n".join(blocks) if blocks else "(no retrieved context)"
    return (
        "System: " + system + "\n\n"
        "Use ONLY the context below to answer. Cite the bracketed sources.\n\n"
        + context_blob + "\n-----\n\n"
        "User Question: " + question + "\n"
        "Assistant:"
    )


def extract_answer(raw_completion: str) -> str:
    """Trim model output at the stop marker / next turn header."""
    text = raw_completion.split(EOS_MARK)[0]
    for marker in ("### Instruction:", "<|im_start|>", "<|im_end|>"):
        text = text.split(marker)[0]
    return text.strip()
