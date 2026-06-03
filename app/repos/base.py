"""Base repo for stores backed by the shared Chroma collection.

The honestly-adapted form of a relational BaseRepo[Model, CreateContract]:
a vector store has one collection and no separate create-contract type, so this
is generic over a single record type T and centralizes collection access.
"""

from __future__ import annotations

import chromadb

from app.db.session import get_collection


class BaseRepo[T]:
    def __init__(self) -> None:
        self._collection = get_collection()

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def count(self) -> int:
        return self._collection.count()
