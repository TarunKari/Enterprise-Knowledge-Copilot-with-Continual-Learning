"""Data curation & formatting for Supervised Fine-Tuning (SFT) and preference pairs.

This module turns raw domain sources (internal wikis, codebases, SOPs, PDFs)
into two JSONL artifacts:

1. ``sft.jsonl``          -> {"instruction", "input", "output"} records
2. ``preference.jsonl``   -> {"prompt", "chosen", "rejected"} records

It also provides deterministic QA-pair mining heuristics so that even without an
LLM labeler you can bootstrap a first training pass from structured documents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
@dataclass
class SFTExample:
    instruction: str
    input: str
    output: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------
SUPPORTED_SUFFIXES = {".md", ".txt", ".rst", ".py", ".json", ".yaml", ".yml", ".pdf", ".docx"}


def iter_source_files(root: Path | str) -> Iterator[Path]:
    """Yield every supported document under *root*."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_document(path: Path) -> tuple[str, dict]:
    """Return ``(text, metadata)`` for a single file of any supported type."""
    suffix = path.suffix.lower()
    metadata = {
        "source": path.name,
        "path": str(path),
        "format": suffix.lstrip("."),
    }
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(path)
            pages = [page.get_text("text") for page in doc]
            text = "\n\n".join(p for p in pages if p.strip())
            metadata["pages"] = doc.page_count
            metadata["last_updated"] = doc.metadata.get("creationDate", "")
            doc.close()
        except ImportError:  # graceful degradation
            text = ""
            metadata["warning"] = "PyMuPDF not installed; skipped PDF extraction"
    elif suffix == ".docx":
        try:
            from docx import Document

            d = Document(str(path))
            text = "\n".join(p.text for p in d.paragraphs)
        except ImportError:
            text = ""
            metadata["warning"] = "python-docx not installed; skipped DOCX extraction"
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return text, metadata


# ---------------------------------------------------------------------------
# Cleaning / normalization
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.M)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown/rST-ish text into (heading, body) sections."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = "General"
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((current_heading, body))

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.*)$", line) or re.match(r"^([A-Z][A-Za-z0-9 ,/&\-]{3,})\s*$", line)
        is_heading = bool(re.match(r"^#{1,4}\s+\S", line))
        if is_heading:
            flush()
            buf = []
            current_heading = re.sub(r"^#+\s*", "", line).strip()
        else:
            buf.append(line)
    flush()
    return sections


# ---------------------------------------------------------------------------
# Heuristic QA mining (bootstrap labels)
# ---------------------------------------------------------------------------
_QA_PATTERNS = [
    re.compile(r"^\s*(?:Q|Question)\s*[:.)]\s*(?P<q>.+?)\s*$", re.M | re.I),
    re.compile(r"^(?P<q>[^\n?]{5,150}\?)\s*$", re.M),
]


def mine_explicit_qa(text: str) -> list[SFTExample]:
    """Mine explicit Q:/A: blocks and 'Question?' followed by answer paragraph."""
    examples: list[SFTExample] = []
    # Pattern A: Q: ... A: ...
    pair_re = re.compile(
        r"(?:^|\n)\s*(?:Q|Question)\s*[:.)]\s*(?P<q>.+?)\s*\n\s*"
        r"(?:A|Answer)\s*[:.)]\s*(?P<a>.+?)(?=\n\s*(?:Q|Question)\s*[:.)]|$)",
        re.S | re.I,
    )
    for m in pair_re.finditer(text):
        q, a = clean_text(m.group("q")), clean_text(m.group("a"))
        if len(a) >= 20:
            examples.append(SFTExample(instruction=q, input="", output=a))
    # Pattern B: heading question + next paragraph
    for heading, body in split_sections(text):
        if heading.endswith("?") and len(body) >= 40:
            examples.append(SFTExample(instruction=heading, input="", output=body[:1500]))
    return examples


def summarize_examples(text: str, max_out: int = 6) -> list[SFTExample]:
    """Generate extractive summarization SFT pairs from long sections."""
    out: list[SFTExample] = []
    for heading, body in split_sections(text):
        if len(body) < 400:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", body)
        scores = Counter(re.findall(r"\w+", body.lower()))
        ranked = sorted(sentences, key=lambda s: sum(scores[w] for w in re.findall(r"\w+", s.lower())), reverse=True)
        summary = " ".join(ranked[: min(3, len(ranked))]).strip()
        if len(summary) >= 60:
            out.append(
                SFTExample(
                    instruction=f"Summarize the following '{heading}' documentation concisely.",
                    input=body[:2000],
                    output=summary,
                )
            )
        if len(out) >= max_out:
            break
    return out


def code_docstring_examples(path: Path, text: str) -> list[SFTExample]:
    """Turn Python docstrings into explain-this-function SFT pairs."""
    if path.suffix != ".py":
        return []
    out: list[SFTExample] = []
    for m in re.finditer(r'(?P<sig>(?:def|class)\s+\w+[^\n]*)\s*\n\s*(?:"""|\'\'\')(?P<doc>.+?)(?:"""|\'\'\')', text, re.S):
        doc = clean_text(m.group("doc"))
        if len(doc) >= 40:
            out.append(
                SFTExample(
                    instruction="Explain what this code does based on its signature and docstring.",
                    input=m.group("sig").strip(),
                    output=doc[:1200],
                )
            )
    return out


# ---------------------------------------------------------------------------
# Preference-pair construction
# ---------------------------------------------------------------------------
def make_preference_pair(example: SFTExample, *, style: str = "verbose") -> PreferenceExample:
    """Build chosen/rejected responses from a gold SFT example.

    ``chosen``   – concise, faithful, citation-friendly version of the gold answer.
    ``rejected`` – a deliberately degraded version (rambling/hedged/padded) used by DPO
    to teach brevity and factuality.  In production these come from human raters or
    an LLM judge; here we synthesize them deterministically so the pipeline is runnable.
    """
    gold = example.output.strip()
    chosen = gold if len(gold) <= 700 else gold[:700].rsplit(".", 1)[0] + "."
    if style == "verbose":
        rejected = (
            f"That's a great question about \"{example.instruction}\"! "
            "There are many factors to consider, and I should note that I might not have "
            "complete information, but broadly speaking: " + " ".join(gold.split()[:40])
            + "... however, this may not be accurate and further research is strongly recommended."
        )
    else:  # hallucinated-style rejection
        rejected = (
            f"Based on my general knowledge, \"{example.instruction}\" is handled automatically "
            "by all modern systems and requires no configuration whatsoever."
        )
    return PreferenceExample(prompt=example.instruction, chosen=chosen, rejected=rejected)


# ---------------------------------------------------------------------------
# Dedup / filtering / splitting
# ---------------------------------------------------------------------------
def dedupe(examples: Iterable[SFTExample]) -> list[SFTExample]:
    seen: set[str] = set()
    out: list[SFTExample] = []
    for ex in examples:
        key = _stable_id(ex.instruction.strip().lower() + "|" + ex.output.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def quality_filter(examples: Iterable[SFTExample]) -> list[SFTExample]:
    out = []
    for ex in examples:
        if len(ex.instruction) < 8 or len(ex.output) < 20:
            continue
        if ex.output.count("\ufffd") > 2:  # mojibake guard
            continue
        out.append(ex)
    return out


def train_val_split(examples: list[SFTExample], val_ratio: float = 0.1, seed: int = 42):
    import random

    rng = random.Random(seed)
    items = examples[:]
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if items else 0
    return items[n_val:], items[:n_val]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_jsonl(records: list[dict], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def build_dataset(source_dir: Path | str, out_dir: Path | str) -> dict:
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    all_examples: list[SFTExample] = []
    per_source_counter: Counter = Counter()

    for path in iter_source_files(source_dir):
        text, meta = load_document(path)
        if not text:
            continue
        text = clean_text(text)
        found = (
            mine_explicit_qa(text)
            + summarize_examples(text)
            + code_docstring_examples(path, text)
        )
        for ex in found:
            ex.metadata = meta
        all_examples.extend(found)
        per_source_counter[path.name] = len(found)

    all_examples = quality_filter(dedupe(all_examples))
    train, val = train_val_split(all_examples)

    stats = {
        "sources_scanned": len(list(iter_source_files(source_dir))),
        "examples_total": len(all_examples),
        "train": len(train),
        "val": len(val),
        "per_source": dict(per_source_counter),
    }

    write_jsonl([e.to_dict() for e in train], out_dir / "sft.jsonl")
    write_jsonl([e.to_dict() for e in val], out_dir / "sft_val.jsonl")

    prefs = [make_preference_pair(e).to_dict() for e in train[: max(50, len(train))]]
    write_jsonl(prefs, out_dir / "preference.jsonl")

    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Curate SFT + preference datasets from raw docs.")
    ap.add_argument("--source", default="data/raw", help="Directory of raw domain documents.")
    ap.add_argument("--out", default="data/processed", help="Output directory for JSONL files.")
    args = ap.parse_args()

    stats = build_dataset(args.source, args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
