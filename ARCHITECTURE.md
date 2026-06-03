# Architecture Guide

This document explains what is implemented in the Nexla take-home MCP server and how data flows through the system.

## System Goal

Provide a local MCP server that:

- ingests the provided PDFs,
- indexes them into a vector store,
- answers natural-language questions with source attribution.

## High-Level Flow

```text
PDF files (data/*.pdf)
  -> Ingestion service
      -> PDF loader (page text)
      -> Chunker (stable chunk IDs)
      -> Embeddings (OpenAI)
      -> Document repo upsert (Chroma)

MCP client question
  -> query_documents tool
      -> Retrieval service (embed query + top-k search)
      -> Synthesis service (LLM grounded answer OR fallback passages)
      -> Answer + citations
```

## Package Structure and Responsibilities

- `app/config.py`
  - Centralized typed settings (`get_settings()`).
  - Reads env values and validates key constraints (e.g., overlap < chunk size).

- `app/db/session.py`
  - Creates/caches Chroma `PersistentClient`.
  - Exposes the single project collection.

- `app/repos/base.py`
  - Base repo abstraction over the shared Chroma collection.

- `app/repos/document_repo.py`
  - All document vector-store operations:
    - `add_chunks`
    - `query`
    - `list_documents`
    - `get_section`

- `app/repos/repo_container.py`
  - Lazy DI-style repo container (`get_repos()`).

- `app/services/ingestion/pdf_loader.py`
  - Extracts per-page text from PDFs using PyMuPDF.
  - Emits `PageText(document, page, text)`.

- `app/services/ingestion/chunker.py`
  - Splits page text into bounded chunks with overlap.
  - Builds deterministic chunk IDs (`document::index`) for idempotent upsert.

- `app/services/ingestion/service.py`
  - Orchestrates parse -> chunk -> embed -> persist.
  - Entrypoint for `make ingest`.

- `app/services/embeddings_service.py`
  - Single wrapper for OpenAI embeddings API.
  - Batch embedding for ingestion and single-query embedding for retrieval.

- `app/services/retrieval/service.py`
  - Embeds question and fetches top-k similar chunks from repo.

- `app/services/retrieval/helpers.py`
  - Converts retrieved chunks into:
    - citation objects (`to_citations`)
    - synthesis context blocks (`to_context`)

- `app/services/synthesis/prompts.py`
  - System/user prompt templates for grounded answering.

- `app/services/synthesis/service.py`
  - Produces final `Answer` object.
  - Uses LLM synthesis when enabled and key exists.
  - Graceful fallback to ranked passage output when disabled/failing.

- `app/handlers/documents.py`
  - Thin orchestration layer for each tool action.
  - No heavy business logic.

- `app/routes/documents.py`
  - Registers FastMCP tools:
    - `query_documents`
    - `list_documents`
    - `get_document_section`

- `app/server.py`
  - FastMCP app entrypoint and tool registration.

## Data Contracts

- Input contracts: `app/contracts/documents.py`
  - `QueryDocumentsRequest`
  - `GetSectionRequest`

- Output/data schemas: `app/schemas/documents.py`
  - `Chunk`, `RetrievedChunk`, `Citation`, `Answer`, `DocumentInfo`, `Section`

## MCP Surface

Implemented MCP tools:

1. `query_documents(question: str, top_k: int | None = None) -> Answer`
2. `list_documents() -> list[DocumentInfo]`
3. `get_document_section(document: str, page: int) -> Section`

## Runtime Characteristics

- Ingestion is idempotent due to deterministic chunk IDs + upsert.
- Retrieval runs across the whole indexed collection (multi-document aware).
- Answers include source attribution (document/page/section/snippet/score).
- Server runs locally via stdio (`python -m app.server` or `make run`).

## Operational Commands

- Install: `make install`
- Ingest: `env -u OPENAI_API_KEY make ingest`
- Run server: `make run`
- Quality checks: `make check`

