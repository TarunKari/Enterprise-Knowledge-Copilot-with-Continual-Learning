"""Phase 2 – Document ingestion & parsing with metadata enrichment.

Extracts text from PDFs (PyMuPDF), Word docs, Markdown/txt and code files via the
``unstructured`` library when available, falling back to lightweight parsers so
ingestion never hard-fails in constrained environments. Every chunk carries full
provenance metadata: source, author, department, page, last_updated, doc_id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("ingestion")


@dataclass
class ParsedDocument:
    doc_id: str
    source: str
    title: str
    author: str
    department: str
    last_updated: str
    language: str
    pages: int
    sections: list[dict] = field(default_factory=list)   # [{page, heading, text}]
    raw_text: str = ""

    @staticmethod
    def make_id(path: Path) -> str:
        return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
_DEPT_HINTS = {
    "hr": ["hr", "people", "recruit", "onboarding", "leave"],
    "engineering": ["engineer", "deploy", "api", "infra", "sre", "runbook"],
    "finance": ["finance", "invoice", "budget", "expense", "procure"],
    "legal": ["legal", "compliance", "policy", "contract"],
    "it": ["it ", "access", "vpn", "password", "account"],
}


def infer_department(name: str, text: str = "") -> str:
    hay = (name + " " + text[:2000]).lower()
    scores = {d: sum(1 for kw in kws if kw in hay) for d, kws in _DEPT_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def parse_date_tag(text: str) -> str:
    """Find ISO-ish dates ('Last updated: 2026-08-01') in a document."""
    m = re.search(r"(?:last[ _-]?updated|date)[:\s]+(\d{4}-\d{2}-\d{2})", text, re.I)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------
def extract_pdf(path: Path) -> tuple[list[dict], dict]:
    """PyMuPDF page-by-page extraction -> (sections, meta)."""
    import fitz

    doc = fitz.open(path)
    sections = []
    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            sections.append({"page": pno, "heading": "", "text": text})
    meta = {
        "author": doc.metadata.get("author", "") or "",
        "title": doc.metadata.get("title", "") or path.stem,
        "last_updated": (doc.metadata.get("creationDate", "") or "").replace("D:", "")[:8],
        "pages": doc.page_count,
    }
    doc.close()
    return sections, meta


def extract_docx(path: Path) -> tuple[list[dict], dict]:
    from docx import Document

    d = Document(str(path))
    props = d.core_properties
    sections, current_heading = [], ""
    for p in d.paragraphs:
        if not p.text.strip():
            continue
        if p.style.name.startswith("Heading"):
            current_heading = p.text.strip()
        sections.append({"page": None, "heading": current_heading, "text": p.text.strip()})
    meta = {"author": props.author or "", "title": props.title or path.stem,
            "last_updated": props.modified.strftime("%Y-%m-%d") if props.modified else "",
            "pages": len(d.sections)}
    return sections, meta


def extract_markdown(path: Path) -> tuple[list[dict], dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections, heading = [], "Introduction"
    buf: list[str] = []
    for line in text.split("\n"):
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            body = "\n".join(buf).strip()
            if body:
                sections.append({"page": None, "heading": heading, "text": body})
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    body = "\n".join(buf).strip()
    if body:
        sections.append({"page": None, "heading": heading, "text": body})
    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    meta = {"author": "", "title": path.stem, "last_updated": "", "pages": 1}
    if fm:
        for kv in fm.group(1).splitlines():
            if ":" in kv:
                k, v = kv.split(":", 1)
                if k.strip().lower() in meta:
                    meta[k.strip().lower()] = v.strip()
    return sections, meta


def extract_with_unstructured(path: Path) -> tuple[list[dict], dict] | None:
    """Preferred universal parser when installed; returns None to trigger fallback."""
    try:
        from unstructured.partition.auto import partition
    except ImportError:
        return None
    try:
        elements = partition(filename=str(path))
        text = "\n".join(str(el) for el in elements)
        return [{"page": getattr(el, "metadata", None).page_number if hasattr(getattr(el, "metadata", None), "page_number") else None,
                 "heading": "", "text": str(el)} for el in elements if str(el).strip()], \
               {"author": "", "title": path.stem, "last_updated": "", "pages": 1}
    except Exception as exc:
        logger.warning("unstructured failed on %s (%s); using fallback parser", path.name, exc)
        return None


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".md": extract_markdown,
    ".markdown": extract_markdown,
    ".txt": lambda p: ([{"page": None, "heading": "", "text": t} for t in
                        [p.read_text(errors="ignore")] ], {"author": "", "title": p.stem, "last_updated": "", "pages": 1}),
    ".py": lambda p: ([{"page": None, "heading": "", "text": p.read_text(errors="ignore")}], {"title": p.stem, "author": "", "last_updated": "", "pages": 1}),
}


def ingest_file(path: Path) -> ParsedDocument | None:
    parsed = extract_with_unstructured(path)
    if parsed is None:
        extractor = EXTRACTORS.get(path.suffix.lower())
        if extractor is None:
            logger.info("Skipping unsupported file type: %s", path.name)
            return None
        try:
            parsed = extractor(path)
        except ImportError as exc:
            logger.warning("Missing dependency for %s: %s", path.name, exc)
            return None
    sections, meta = parsed
    raw = "\n\n".join(s["text"] for s in sections)
    if not raw.strip():
        return None
    return ParsedDocument(
        doc_id=ParsedDocument.make_id(path),
        source=path.name,
        title=meta.get("title") or path.stem,
        author=meta.get("author", ""),
        department=infer_department(path.name, raw),
        last_updated=meta.get("last_updated") or parse_date_tag(raw),
        language="en",
        pages=int(meta.get("pages") or 0),
        sections=sections,
        raw_text=raw,
    )


def ingest_directory(root: Path | str) -> Iterator[ParsedDocument]:
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            doc = ingest_file(path)
            if doc:
                logger.info("Ingested %s (%d sections, dept=%s)", doc.source, len(doc.sections), doc.department)
                yield doc


def save_corpus(docs: list[ParsedDocument], out_path: Path | str) -> str:
    """Serialize ingested corpus to JSONL for the chunking/embedding stages."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            rec = asdict(d)
            rec.pop("raw_text", None)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse domain documents into metadata-rich JSONL.")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--out", default="data/processed/corpus.jsonl")
    args = ap.parse_args()
    docs = list(ingest_directory(args.source))
    path = save_corpus(docs, args.out)
    print(json.dumps({"documents": len(docs), "output": path}, indent=2))


if __name__ == "__main__":
    main()
