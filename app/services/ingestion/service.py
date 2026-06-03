"""Ingestion pipeline — parse → chunk → embed → persist.

Run via `make ingest` (python -m app.services.ingestion.service). Idempotent:
deterministic chunk ids mean re-running upserts rather than duplicating.
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.repos.repo_container import get_repos
from app.services.embeddings_service import get_embeddings_service
from app.services.ingestion.chunker import chunk_pages
from app.services.ingestion.pdf_loader import load_pdfs


def ingest(data_dir: Path | None = None) -> int:
    """Index every PDF in data_dir. Returns the number of chunks written."""
    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    pages = load_pdfs(data_dir)
    chunks = chunk_pages(pages)
    if not chunks:
        print("No chunks produced — are the PDFs text-based, not scanned images?")
        return 0

    embeddings = get_embeddings_service().embed_texts([c.text for c in chunks])
    get_repos().documents.add_chunks(chunks, embeddings)

    docs = {c.document for c in chunks}
    print(
        f"Indexed {len(chunks)} chunks from {len(docs)} document(s) "
        f"across {len(pages)} pages into '{settings.collection_name}'."
    )
    return len(chunks)


if __name__ == "__main__":
    ingest()
