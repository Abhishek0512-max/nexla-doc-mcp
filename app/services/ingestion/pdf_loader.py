"""PDF loading — PyMuPDF turns each page into clean, page-tagged text.

The only module that knows the PDF format. It emits PageText records; everything
downstream (the chunker) is format-agnostic. Swap to pypdf here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.tools.text import clean_pdf_text


@dataclass
class PageText:
    document: str  # source file name, e.g. "whitepaper.pdf"
    page: int  # 1-based page number
    text: str


def load_pdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        for index, page in enumerate(doc):
            text = clean_pdf_text(page.get_text())
            if text:  # skip blank / image-only pages
                pages.append(PageText(document=path.name, page=index + 1, text=text))
    return pages


def load_pdfs(data_dir: Path) -> list[PageText]:
    pdf_paths = sorted(data_dir.glob("*.pdf"))  # sorted = deterministic order
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir.resolve()}")

    pages: list[PageText] = []
    for path in pdf_paths:
        pages.extend(load_pdf(path))
    return pages
