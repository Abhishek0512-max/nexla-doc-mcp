"""Dependency-injection container — the single seam services depend on.

Services call get_repos().documents rather than constructing repos themselves,
so wiring lives in one place and tests can swap the container.
"""

from __future__ import annotations

from functools import lru_cache

from app.repos.document_repo import DocumentRepo


class RepoContainer:
    def __init__(self) -> None:
        self._documents: DocumentRepo | None = None

    @property
    def documents(self) -> DocumentRepo:
        if self._documents is None:
            self._documents = DocumentRepo()
        return self._documents


@lru_cache
def get_repos() -> RepoContainer:
    return RepoContainer()
