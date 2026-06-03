"""Chunking — page-aware, size-capped, overlapping Chunks with stable ids.

A chunk never crosses a page boundary, so each one keeps an exact page number
for attribution. Within a page we break text into paragraph/sentence pieces,
then pack them up to CHUNK_SIZE with CHUNK_OVERLAP characters carried between
neighbours so facts split across a boundary still appear whole somewhere.
"""

from __future__ import annotations

import re

from app.config import get_settings
from app.schemas.documents import Chunk
from app.services.ingestion.pdf_loader import PageText

_PARAGRAPH = re.compile(r"\n{2,}")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^\s*(\d+(\.\d+)*\s+\S.*|[A-Z][A-Za-z0-9 ,&\-]{2,60})\s*$")


def _atoms(text: str, max_len: int) -> list[str]:
    """Break text into pieces <= max_len: paragraph -> sentence -> hard window."""
    pieces: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len:
            pieces.append(para)
            continue
        buf = ""
        for sent in _SENTENCE.split(para):
            if len(sent) > max_len:  # sentence too big -> windows
                if buf:
                    pieces.append(buf.strip())
                    buf = ""
                for i in range(0, len(sent), max_len):
                    pieces.append(sent[i : i + max_len].strip())
            elif len(buf) + len(sent) + 1 <= max_len:
                buf = f"{buf} {sent}".strip()
            else:
                pieces.append(buf.strip())
                buf = sent
        if buf.strip():
            pieces.append(buf.strip())
    return pieces


def _pack(atoms: list[str], size: int, overlap: int) -> list[str]:
    """Merge atoms into chunks <= size, carrying `overlap` chars between them."""
    chunks: list[str] = []
    current = ""
    for atom in atoms:
        if current and len(current) + len(atom) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            separator = " " if tail and len(tail) + len(atom) + 1 <= size + overlap else ""
            current = f"{tail}{separator}{atom}".strip()
        else:
            current = f"{current} {atom}".strip() if current else atom
    if current:
        chunks.append(current)
    return chunks


def _detect_heading(page_text: str) -> str | None:
    """Best-effort: first line on the page that looks like a heading."""
    for line in page_text.splitlines():
        if _HEADING.match(line):
            return line.strip()
    return None


def chunk_pages(pages: list[PageText]) -> list[Chunk]:
    settings = get_settings()
    chunks: list[Chunk] = []
    per_doc_index: dict[str, int] = {}

    for page in pages:
        section = _detect_heading(page.text)
        for body in _pack(
            _atoms(page.text, settings.chunk_size), settings.chunk_size, settings.chunk_overlap
        ):
            idx = per_doc_index.get(page.document, 0)
            per_doc_index[page.document] = idx + 1
            chunks.append(
                Chunk(
                    id=f"{page.document}::{idx:04d}",
                    text=body,
                    document=page.document,
                    page=page.page,
                    section=section,
                    chunk_index=idx,
                )
            )
    return chunks
