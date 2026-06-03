"""Inbound request shapes for the MCP tools (the *Request models).

Kept separate from schemas/ (the outputs) so 'what a caller sends' and 'what we
return' are distinct types — the same boundary your contracts/ vs schemas/ split
draws. list_documents takes no input, so it needs no contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryDocumentsRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class GetSectionRequest(BaseModel):
    document: str = Field(min_length=1)
    page: int = Field(ge=1)
