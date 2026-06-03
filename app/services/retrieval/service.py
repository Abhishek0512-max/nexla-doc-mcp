"""Retrieval — embed the question, retrieve candidates, and rerank them."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.repos.repo_container import get_repos
from app.schemas.documents import RetrievedChunk
from app.services.embeddings_service import get_embeddings_service
from app.services.retrieval.reranker import fuse_and_rerank


class RetrievalService:
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        settings = get_settings()
        k = top_k or settings.top_k
        candidate_k = max(k, settings.retrieval_candidate_k)
        vector = get_embeddings_service().embed_query(question)
        repo = get_repos().documents
        vector_chunks = repo.query(vector, candidate_k)
        if not settings.enable_hybrid_retrieval:
            return vector_chunks[:k]

        lexical_chunks = repo.keyword_query(question, candidate_k)
        return fuse_and_rerank(
            vector_chunks=vector_chunks,
            lexical_chunks=lexical_chunks,
            top_k=k,
            vector_weight=settings.vector_weight,
            lexical_weight=settings.lexical_weight,
        )


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()
