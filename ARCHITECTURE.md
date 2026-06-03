# Architecture Guide

This document explains what is implemented in the Nexla take-home MCP server and how data flows through the system. The README gives the quick overview; this file is the deeper implementation map.

## System Goal

Provide a local MCP server that:

- ingests the provided PDFs,
- indexes them into a persistent local vector store,
- answers natural-language questions with source attribution,
- exposes a small, read-only MCP tool surface for document Q&A and source inspection.

## High-Level Flow

```text
PDF files (data/*.pdf)
  -> Ingestion service
      -> PDF loader (page text via PyMuPDF)
      -> Chunker (page-aware chunks + stable chunk IDs)
      -> Embeddings service (OpenAI embeddings)
      -> Document repo upsert (Chroma PersistentClient)

MCP client question
  -> query_documents tool
      -> Handler
      -> Retrieval service
          -> Embed query
          -> Dense Chroma search over candidate_k chunks
          -> BM25-style keyword search over candidate_k chunks
          -> Weighted fusion/rerank to final top_k
      -> Synthesis service
          -> Grounded LLM answer with citations
          -> Fallback ranked passages if synthesis is disabled or fails
      -> Answer + citations
```

## Package Structure and Responsibilities

- `app/config.py`
  - Centralized typed settings (`get_settings()`).
  - Reads `.env` / environment values once and validates constraints such as `CHUNK_OVERLAP < CHUNK_SIZE` and positive retrieval weights.
  - Owns retrieval knobs: `TOP_K`, `RETRIEVAL_CANDIDATE_K`, `ENABLE_HYBRID_RETRIEVAL`, `VECTOR_WEIGHT`, and `LEXICAL_WEIGHT`.

- `app/db/session.py`
  - Creates and caches the Chroma `PersistentClient`.
  - Exposes the single project collection used by all repos.

- `app/repos/base.py`
  - Base repo abstraction over the shared Chroma collection.

- `app/repos/document_repo.py`
  - Owns all document vector-store operations.
  - Implements idempotent `add_chunks`, dense vector `query`, BM25-style `keyword_query`, `list_documents`, and `get_section`.
  - Keeps Chroma-specific query shapes and metadata handling isolated from higher layers.

- `app/repos/repo_container.py`
  - Lazy DI-style repo container (`get_repos()`), so services depend on an app-level interface rather than constructing repos directly.

- `app/services/ingestion/pdf_loader.py`
  - Extracts per-page text from PDFs using PyMuPDF.
  - Emits `PageText(document, page, text)`.

- `app/services/ingestion/chunker.py`
  - Splits page text into bounded chunks with overlap.
  - Keeps chunks page-aware so citations have exact page numbers.
  - Builds deterministic chunk IDs (`document::index`) for idempotent upserts.

- `app/services/ingestion/service.py`
  - Orchestrates parse -> chunk -> embed -> persist.
  - Entrypoint behind `make ingest`.

- `app/services/embeddings_service.py`
  - Single wrapper for the OpenAI embeddings API.
  - Supports batch embedding for ingestion and single-query embedding for retrieval.

- `app/services/retrieval/service.py`
  - Orchestrates retrieval for a question.
  - Embeds the query, over-retrieves candidates, optionally combines dense and lexical results, and returns the final ranked chunks.

- `app/services/retrieval/reranker.py`
  - Deterministic local reranker/fusion layer.
  - Normalizes dense and lexical scores, applies configured weights, merges duplicate chunks, and returns the final top-k.

- `app/services/retrieval/helpers.py`
  - Converts retrieved chunks into citation objects (`to_citations`) and numbered synthesis context blocks (`to_context`).

- `app/services/synthesis/prompts.py`
  - System/user prompt templates for grounded answering.
  - Treats retrieved document text as untrusted evidence, not executable instructions.

- `app/services/synthesis/service.py`
  - Produces the final `Answer` object.
  - Uses LLM synthesis when enabled and a key exists.
  - Falls back to ranked passages when synthesis is disabled or an LLM/network call fails.

- `app/tools/text.py`
  - Pure text cleanup helpers for whitespace normalization and dehyphenation.
  - Keeps low-level text normalization separate from PDF loading and chunking.

- `app/handlers/documents.py`
  - Thin orchestration layer for each tool action.
  - Keeps MCP route functions small and free of heavy business logic.

- `app/routes/documents.py`
  - Defines the FastMCP document tools through a `register(mcp)` function:
    - `query_documents`
    - `list_documents`
    - `get_document_section`
  - Adds parameter constraints, docstrings, and read-only annotations so MCP clients get useful schemas.

- `app/server.py`
  - Creates the FastMCP app instance and calls route registration.
  - Keeps app construction separate from tool definitions to avoid circular imports as the tool surface grows.

## Data Contracts

- Input contracts: `app/contracts/documents.py`
  - `QueryDocumentsRequest`
  - `GetSectionRequest`

- Output/data schemas: `app/schemas/documents.py`
  - `Chunk`
  - `RetrievedChunk`
  - `Citation`
  - `Answer`
  - `DocumentInfo`
  - `Section`

## MCP Surface

Implemented MCP tools:

1. `query_documents(question: str, top_k: int | None = None) -> Answer`
2. `list_documents() -> list[DocumentInfo]`
3. `get_document_section(document: str, page: int) -> Section`

The tool surface is intentionally small and read-only. `query_documents` answers from the indexed corpus, `list_documents` verifies coverage, and `get_document_section` lets a client inspect the raw page text behind a citation.

## Runtime Characteristics

- Ingestion is idempotent due to deterministic chunk IDs plus Chroma upsert.
- Retrieval runs across the whole indexed collection, so multi-document questions work without a separate document-selection step.
- Hybrid retrieval combines semantic similarity with exact keyword matching for names, datasets, affiliations, and paper-specific terminology.
- Answers include source attribution: document, page, best-effort section, snippet, and score. Page number is the reliable attribution anchor.
- Server runs locally over stdio (`python -m app.server` or `make run`).
- API keys are needed at query time because each incoming question is embedded before retrieval.

## Quality Gates

- Unit tests cover chunking, repo behavior, retrieval fusion/toggles, synthesis fallback, eval metrics, and MCP tool schemas.
- `make eval` runs a golden retrieval dataset and reports document recall, page recall, MRR, and exact-source pass rate.
- `make check` runs formatting, linting, typing, and tests.

## Operational Commands

- Install: `make install`
- Ingest: `make ingest`
- Run server: `make run`
- Run tests: `make test`
- Run retrieval eval: `make eval`
- Full quality checks: `make check`

Troubleshooting: if a known-good `.env` key produces OpenAI 401s, check for a stale shell variable. Shell `OPENAI_API_KEY` takes precedence over `.env`, so run `unset OPENAI_API_KEY` when you want the `.env` value to be used.

