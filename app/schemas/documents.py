"""Pydantic models: the data shapes that flow between layers.

In your house layout these would split across models/ (ORM) and schemas/, but
our only persistence is the Chroma vector store — there is no relational ORM —
so these typed records ARE the models, defined once and reused everywhere.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One indexed unit of text. Produced by ingestion, stored by the repo."""

    id: str  # stable, deterministic id (set by the chunker)
    text: str
    document: str  # source file name, e.g. "whitepaper.pdf"
    page: int  # 1-based page number, for attribution
    section: str | None = None  # nearest heading, when detectable
    chunk_index: int  # position within the document


class RetrievedChunk(BaseModel):
    """A chunk returned from a similarity search, with its score."""

    id: str
    text: str
    document: str
    page: int
    section: str | None = None
    score: float  # cosine similarity in [0, 1] — higher is closer


class Citation(BaseModel):
    """Source attribution attached to an answer."""

    document: str
    page: int
    section: str | None = None
    snippet: str  # short supporting excerpt
    score: float


class Answer(BaseModel):
    """The query_documents result."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    synthesized: bool  # True = LLM prose; False = ranked-chunks fallback


class DocumentInfo(BaseModel):
    """One row of list_documents."""

    document: str
    pages: int
    chunks: int


class Section(BaseModel):
    """The get_document_section result."""

    document: str
    page: int
    text: str
