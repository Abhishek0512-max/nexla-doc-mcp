"""Pure text helpers — no domain knowledge, no I/O."""

from __future__ import annotations

import re

_HORIZONTAL_WS = re.compile(r"[ \t]+")
_EXCESS_BLANKS = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK = re.compile(r"(\w+)-\n\s*(\w+)")


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break: 'infor-\\nmation' -> 'information'."""
    return _HYPHEN_LINEBREAK.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Collapse horizontal whitespace and runs of blank lines; trim edges."""
    text = _HORIZONTAL_WS.sub(" ", text)
    text = _EXCESS_BLANKS.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def clean_pdf_text(text: str) -> str:
    """Standard cleanup for raw PDF-extracted text (order matters: dehyphenate
    needs the line breaks intact, so it runs before whitespace is collapsed)."""
    return normalize_whitespace(dehyphenate(text))
