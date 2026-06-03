"""Synthesis — turn retrieved chunks into a grounded, cited Answer.

Hybrid: with synthesis enabled and a key present, an LLM composes prose citing
[n] markers. Otherwise (or on any LLM failure) we return the ranked chunks
themselves — still fully attributed. Citations always come from retrieval.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import get_settings
from app.schemas.documents import Answer, Citation, RetrievedChunk
from app.services.retrieval.helpers import to_citations, to_context
from app.services.synthesis.prompts import SYSTEM_PROMPT, build_user_prompt


class SynthesisService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: OpenAI | None = (
            OpenAI(api_key=self._settings.openai_api_key) if self._settings.has_openai_key else None
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        citations = to_citations(chunks)

        if not chunks:
            return Answer(
                answer="I couldn't find anything relevant in the indexed documents.",
                citations=[],
                synthesized=False,
            )

        if not (self._settings.enable_synthesis and self._client):
            return self._fallback(chunks, citations)

        try:
            text = self._synthesize(question, chunks)
            return Answer(answer=text, citations=citations, synthesized=True)
        except Exception:
            return self._fallback(chunks, citations)  # degrade, never crash

    def _synthesize(self, question: str, chunks: list[RetrievedChunk]) -> str:
        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, to_context(chunks))},
            ],
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()

    def _fallback(self, chunks: list[RetrievedChunk], citations: list[Citation]) -> Answer:
        passages = [
            f"[{i}] {c.document} (p{c.page}): {c.text}" for i, c in enumerate(chunks, start=1)
        ]
        return Answer(
            answer="Synthesis disabled — most relevant passages:\n\n" + "\n\n".join(passages),
            citations=citations,
            synthesized=False,
        )


@lru_cache
def get_synthesis_service() -> SynthesisService:
    return SynthesisService()
