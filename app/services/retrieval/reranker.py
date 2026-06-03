"""Local reranking/fusion for hybrid retrieval candidates."""

from __future__ import annotations

from app.schemas.documents import RetrievedChunk


def fuse_and_rerank(
    *,
    vector_chunks: list[RetrievedChunk],
    lexical_chunks: list[RetrievedChunk],
    top_k: int,
    vector_weight: float,
    lexical_weight: float,
) -> list[RetrievedChunk]:
    """Fuse dense and lexical candidates, then return the final top-k.

    Scores are normalized per retrieval source before weighted fusion. This keeps
    the improvement local and deterministic while leaving room to swap in a true
    cross-encoder/API reranker later.
    """
    candidates: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    rank = 0

    def add(chunks: list[RetrievedChunk], weight: float) -> None:
        nonlocal rank
        if not chunks or weight <= 0:
            return
        max_score = max(max(chunk.score, 0.0) for chunk in chunks) or 1.0
        for chunk in chunks:
            candidates.setdefault(chunk.id, chunk)
            first_seen.setdefault(chunk.id, rank)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight * (
                max(chunk.score, 0.0) / max_score
            )
            rank += 1

    add(vector_chunks, vector_weight)
    add(lexical_chunks, lexical_weight)

    ordered = sorted(
        candidates,
        key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id], chunk_id),
    )
    return [
        candidates[chunk_id].model_copy(update={"score": scores[chunk_id]})
        for chunk_id in ordered[:top_k]
    ]
