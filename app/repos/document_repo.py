"""DocumentRepo — all vector-store access for document chunks lives here."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from app.repos.base import BaseRepo
from app.schemas.documents import Chunk, DocumentInfo, RetrievedChunk, Section

_TOKEN = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9_\-]*\b")
_BM25_K1 = 1.5
_BM25_B = 0.75
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "paper",
    "papers",
    "the",
    "these",
    "this",
    "to",
    "what",
    "which",
}


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def _query_tokens(text: str) -> list[str]:
    tokens = [token for token in _tokens(text) if token not in _QUERY_STOPWORDS]
    return tokens or _tokens(text)


class DocumentRepo(BaseRepo[Chunk]):
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Idempotent bulk insert. upsert (not add) so re-running ingestion
        replaces existing chunks by id instead of erroring on duplicates."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=cast(list[Sequence[float]], embeddings),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document": c.document,
                    "page": c.page,
                    "section": c.section or "",  # Chroma metadata can't hold None
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []

        res = self.collection.query(
            query_embeddings=cast(list[Sequence[float]], [embedding]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids_batch = res.get("ids")
        docs_batch = res.get("documents")
        metas_batch = res.get("metadatas")
        dists_batch = res.get("distances")
        if not ids_batch or not docs_batch or not metas_batch or not dists_batch:
            return []

        ids = ids_batch[0]
        docs = docs_batch[0]
        metas = cast(list[Mapping[str, Any]], metas_batch[0])
        dists = dists_batch[0]

        return [
            RetrievedChunk(
                id=cid,
                text=text,
                document=str(meta.get("document", "")),
                page=int(cast(int | str, meta.get("page", 0))),
                section=str(meta["section"]) if meta.get("section") else None,
                score=1.0 - float(dist),  # cosine distance → similarity
            )
            for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=False)
        ]

    def keyword_query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Small-corpus BM25-style lexical retrieval over stored chunks."""
        if self.count() == 0:
            return []

        query_terms = _query_tokens(question)
        if not query_terms:
            return []

        got = self.collection.get(include=["documents", "metadatas"])
        ids = cast(list[str], got["ids"] or [])
        docs = cast(list[str], got["documents"] or [])
        metas = cast(list[Mapping[str, Any]], got["metadatas"] or [])
        if not ids or not docs or not metas:
            return []

        tokenized_docs = [_tokens(doc) for doc in docs]
        doc_count = len(tokenized_docs)
        avg_len = sum(len(tokens) for tokens in tokenized_docs) / doc_count
        doc_freq = {
            term: sum(1 for doc_tokens in tokenized_docs if term in set(doc_tokens))
            for term in set(query_terms)
        }

        scored: list[tuple[float, str, str, Mapping[str, Any]]] = []
        for cid, text, meta, doc_tokens in zip(ids, docs, metas, tokenized_docs, strict=False):
            term_counts = Counter(doc_tokens)
            doc_len = len(doc_tokens) or 1
            score = 0.0
            for term in query_terms:
                freq = term_counts[term]
                if freq == 0:
                    continue
                df = doc_freq[term]
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                denom = freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg_len)
                score += idf * (freq * (_BM25_K1 + 1)) / denom

            if score > 0:
                scored.append((score, cid, text, meta))

        return [
            RetrievedChunk(
                id=cid,
                text=text,
                document=str(meta.get("document", "")),
                page=int(cast(int | str, meta.get("page", 0))),
                section=str(meta["section"]) if meta.get("section") else None,
                score=score,
            )
            for score, cid, text, meta in sorted(
                scored,
                key=lambda row: (-row[0], row[1]),
            )[:top_k]
        ]

    def list_documents(self) -> list[DocumentInfo]:
        """No GROUP BY in a vector store, so we pull all metadata and aggregate
        in Python — fine at this scale (a handful of PDFs)."""
        got = self.collection.get(include=["metadatas"])
        agg: dict[str, dict[str, set[int] | int]] = {}
        for raw_meta in got["metadatas"] or []:
            meta = cast(Mapping[str, Any], raw_meta)
            document = str(meta.get("document", ""))
            page_raw = meta.get("page")
            if not document or page_raw is None:
                continue

            entry = agg.setdefault(document, {"pages": set(), "chunks": 0})
            pages = cast(set[int], entry["pages"])
            pages.add(int(cast(int | str, page_raw)))
            entry["chunks"] = cast(int, entry["chunks"]) + 1

        return [
            DocumentInfo(
                document=doc,
                pages=len(cast(set[int], v["pages"])),
                chunks=cast(int, v["chunks"]),
            )
            for doc, v in sorted(agg.items())
        ]

    def get_section(self, document: str, page: int) -> Section | None:
        got = self.collection.get(
            where={"$and": [{"document": document}, {"page": page}]},
            include=["documents", "metadatas"],
        )
        docs = cast(list[str], got["documents"] or [])
        metas = cast(list[Mapping[str, Any]], got["metadatas"] or [])
        if not docs or not metas:
            return None

        ordered = sorted(
            zip(docs, metas, strict=False),
            key=lambda r: int(cast(int | str, r[1].get("chunk_index", 0))),
        )
        return Section(
            document=document,
            page=page,
            text="\n".join(text for text, _ in ordered),
        )
