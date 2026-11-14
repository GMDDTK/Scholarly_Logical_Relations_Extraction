from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import pymupdf4llm


_INLINE_FORMATTING_TAG_PATTERN = re.compile(
    r"</?(?:sup|u)\s*>",
    flags=re.IGNORECASE,
)
_LINE_BREAK_TAG_PATTERN = re.compile(
    r"<br\s*/?\s*>",
    flags=re.IGNORECASE,
)


def clean_pymupdf4llm_markup(markdown: str) -> str:
    without_line_break_tags = _LINE_BREAK_TAG_PATTERN.sub(" ", markdown)
    return _INLINE_FORMATTING_TAG_PATTERN.sub("", without_line_break_tags)


def write_pdf_markdown(
    *,
    input_pdf: Path,
    output_markdown: Path,
) -> str:
    if not input_pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {input_pdf}")

    markdown = pymupdf4llm.to_markdown(
        str(input_pdf),
        page_chunks=False,
        page_separators=False,
        header=False,
        footer=False,
        use_ocr=False,
    )
    if not isinstance(markdown, str):
        raise TypeError("Expected one Markdown string from PyMuPDF4LLM")
    cleaned = clean_pymupdf4llm_markup(markdown)
    if not cleaned.strip():
        raise ValueError("PyMuPDF4LLM produced no non-whitespace text")

    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_markdown.parent / (
        f".{output_markdown.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(cleaned)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_markdown)
    finally:
        temporary.unlink(missing_ok=True)
    return cleaned
