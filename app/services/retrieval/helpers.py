"""Shared retrieval formatting — citations and LLM context."""

from __future__ import annotations

from app.schemas.documents import Citation, RetrievedChunk

_SNIPPET = 240
_SCORE_DECIMALS = 4


def _snippet(text: str, limit: int = _SNIPPET) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            document=c.document,
            page=c.page,
            section=c.section,
            snippet=_snippet(c.text),
            score=round(c.score, _SCORE_DECIMALS),
        )
        for c in chunks
    ]


def to_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.document} (p{c.page})"
        if c.section:
            header += f" — {c.section}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)
