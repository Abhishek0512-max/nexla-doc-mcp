"""Vector store session — the Chroma persistence layer.

The analog of a SQLAlchemy session module, but for an embedded vector store.
One PersistentClient per process, exposing the project's single collection.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings


@lru_cache
def get_client() -> Any:
    """One on-disk Chroma client per process (cached singleton)."""
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection() -> chromadb.Collection:
    """The single collection holding all document chunks.

    We bring our own embeddings (OpenAI), so embedding_function is None — Chroma
    stores and searches vectors but never computes them. Cosine space matches
    OpenAI text-embedding similarity.
    """
    settings = get_settings()
    return cast(
        chromadb.Collection,
        get_client().get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        ),
    )
