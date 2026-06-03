"""OpenAI embeddings — the single wrapper around the embedding API.

embed_texts() for ingestion (batched), embed_query() for search. The only module
that touches the embedding API; swapping to a local model means rewriting just
this file, keeping the same two-method surface.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import get_settings

_BATCH = 100  # inputs per request: well under limits, flat memory, fewer round trips


class EmbeddingsService:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.has_openai_key:
            raise RuntimeError("OPENAI_API_KEY is required to embed. Set it in .env.")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            batch = texts[start : start + _BATCH]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)  # defensive
            vectors.extend(item.embedding for item in ordered)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return resp.data[0].embedding


@lru_cache
def get_embeddings_service() -> EmbeddingsService:
    """Cached so the OpenAI client (and its connection pool) is reused."""
    return EmbeddingsService()
