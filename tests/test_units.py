from __future__ import annotations

import re
from itertools import pairwise
from types import SimpleNamespace

import pytest
from app.contracts.documents import GetSectionRequest, QueryDocumentsRequest
from app.handlers import documents as handlers_module
from app.repos.document_repo import DocumentRepo
from app.schemas.documents import Chunk, RetrievedChunk
from app.services import embeddings_service as embeddings_module
from app.services.embeddings_service import EmbeddingsService
from app.services.ingestion.chunker import chunk_pages
from app.services.ingestion.pdf_loader import PageText
from app.services.retrieval import helpers as retrieval_helpers
from app.services.retrieval import service as retrieval_module
from app.services.synthesis import service as synthesis_module
from app.tools.text import clean_pdf_text, dehyphenate, normalize_whitespace
from evals.run_retrieval_eval import (
    ExpectedSource,
    GoldenCase,
    evaluate_case,
    summarize,
)
from fastmcp import Client
from pydantic import ValidationError


def make_chunk(
    *,
    chunk_id: str = "doc.pdf::0000",
    text: str = "supporting text",
    document: str = "doc.pdf",
    page: int = 1,
    section: str | None = "Intro",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        document=document,
        page=page,
        section=section,
        chunk_index=chunk_index,
    )


def make_retrieved(
    *,
    text: str = "supporting retrieved text",
    document: str = "doc.pdf",
    page: int = 1,
    section: str | None = "Intro",
    score: float = 0.98765,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document}::{page:04d}",
        text=text,
        document=document,
        page=page,
        section=section,
        score=score,
    )


def test_text_helpers_dehyphenate_normalize_and_clean_in_order() -> None:
    assert dehyphenate("infor-\nmation") == "information"
    assert dehyphenate("well-known") == "well-known"

    messy = "  a \t\t b\n\n\n\n c  "
    assert normalize_whitespace(messy) == "a b\n\nc"

    assert clean_pdf_text("infor-\nmation   x") == "information x"


def test_chunking_is_deterministic(configure_env) -> None:
    configure_env(chunk_size=45, chunk_overlap=6)
    pages = [
        PageText("doc.pdf", 1, "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."),
        PageText("doc.pdf", 2, "Kappa lambda mu. Nu xi omicron. Pi rho sigma."),
    ]

    first = chunk_pages(pages)
    second = chunk_pages(pages)

    assert [(c.id, c.text) for c in first] == [(c.id, c.text) for c in second]


def test_chunking_keeps_chunks_page_bounded(configure_env) -> None:
    configure_env(chunk_size=35, chunk_overlap=5)
    pages = [
        PageText("doc.pdf", 1, "AAA one. AAA two. AAA three. AAA four."),
        PageText("doc.pdf", 2, "BBB one. BBB two. BBB three. BBB four."),
    ]

    chunks = chunk_pages(pages)

    assert chunks
    for chunk in chunks:
        if chunk.page == 1:
            assert "AAA" in chunk.text
            assert "BBB" not in chunk.text
        else:
            assert chunk.page == 2
            assert "BBB" in chunk.text
            assert "AAA" not in chunk.text


def test_chunking_respects_size_bound_with_overlap(configure_env) -> None:
    chunk_size = 28
    overlap = 7
    configure_env(chunk_size=chunk_size, chunk_overlap=overlap)
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."

    chunks = chunk_pages([PageText("doc.pdf", 1, text)])

    assert len(chunks) > 1
    assert all(len(chunk.text) <= chunk_size + overlap for chunk in chunks)


def test_chunking_carries_overlap_between_neighboring_chunks(configure_env) -> None:
    overlap = 8
    configure_env(chunk_size=32, chunk_overlap=overlap)
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu. Nu xi omicron."

    chunks = chunk_pages([PageText("doc.pdf", 1, text)])

    assert len(chunks) > 1
    for left, right in pairwise(chunks):
        assert right.text.startswith(left.text[-overlap:])


def test_chunking_ids_are_unique_formatted_and_increment_per_doc(configure_env) -> None:
    configure_env(chunk_size=30, chunk_overlap=5)
    pages = [
        PageText("doc.pdf", 1, "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."),
        PageText("doc.pdf", 2, "Kappa lambda mu. Nu xi omicron. Pi rho sigma."),
        PageText("other.pdf", 1, "One two three. Four five six. Seven eight nine."),
    ]

    chunks = chunk_pages(pages)
    ids = [chunk.id for chunk in chunks]
    doc_chunks = [chunk for chunk in chunks if chunk.document == "doc.pdf"]

    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r".+\.pdf::\d{4}", chunk.id) for chunk in chunks)
    assert [chunk.chunk_index for chunk in doc_chunks] == list(range(len(doc_chunks)))
    assert [chunk.id for chunk in doc_chunks] == [
        f"doc.pdf::{index:04d}" for index in range(len(doc_chunks))
    ]


def test_chunking_splits_huge_unbroken_paragraph(configure_env) -> None:
    chunk_size = 25
    overlap = 5
    configure_env(chunk_size=chunk_size, chunk_overlap=overlap)

    chunks = chunk_pages([PageText("doc.pdf", 1, "x" * 90)])

    assert len(chunks) > 1
    assert all(len(chunk.text) <= chunk_size + overlap for chunk in chunks)


def test_chunking_handles_empty_and_short_pages(configure_env) -> None:
    configure_env(chunk_size=40, chunk_overlap=5)

    assert chunk_pages([PageText("doc.pdf", 1, "   \n\t  ")]) == []

    chunks = chunk_pages([PageText("doc.pdf", 1, "Short page.")])
    assert len(chunks) == 1
    assert chunks[0].text == "Short page."


def test_embeddings_batches_inputs_without_network(monkeypatch, configure_env) -> None:
    configure_env()
    calls: list[list[str]] = []

    class FakeOpenAI:
        def __init__(self, api_key: str | None) -> None:
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            calls.append(input)
            data = [
                SimpleNamespace(index=index, embedding=[float(index)])
                for index, _ in enumerate(input)
            ]
            return SimpleNamespace(data=data)

    monkeypatch.setattr(embeddings_module, "OpenAI", FakeOpenAI)

    vectors = EmbeddingsService().embed_texts([f"text {index}" for index in range(250)])

    assert [len(call) for call in calls] == [100, 100, 50]
    assert len(vectors) == 250


def test_embeddings_restore_api_order_from_shuffled_response(
    monkeypatch,
    configure_env,
) -> None:
    configure_env()

    class FakeOpenAI:
        def __init__(self, api_key: str | None) -> None:
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            data = [
                SimpleNamespace(index=2, embedding=[2.0]),
                SimpleNamespace(index=0, embedding=[0.0]),
                SimpleNamespace(index=1, embedding=[1.0]),
            ]
            return SimpleNamespace(data=data)

    monkeypatch.setattr(embeddings_module, "OpenAI", FakeOpenAI)

    assert EmbeddingsService().embed_texts(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]


def test_embeddings_constructor_requires_api_key(configure_env) -> None:
    configure_env(openai_key="")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        EmbeddingsService()


def test_document_repo_round_trip_query_and_score(configure_env) -> None:
    configure_env()
    repo = DocumentRepo()
    chunks = [
        make_chunk(chunk_id="doc.pdf::0000", text="alpha", chunk_index=0),
        make_chunk(chunk_id="doc.pdf::0001", text="beta", chunk_index=1),
    ]

    repo.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = repo.query([1.0, 0.0, 0.0], 2)

    assert repo.count() == 2
    assert result[0].id == "doc.pdf::0000"
    assert 0 <= result[0].score <= 1
    assert result[0].score >= result[1].score


def test_document_repo_keyword_query_matches_exact_terms(configure_env) -> None:
    configure_env()
    repo = DocumentRepo()
    chunks = [
        make_chunk(
            chunk_id="doc.pdf::0000",
            text="This chunk discusses general multilingual translation.",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="doc.pdf::0001",
            text="This chunk names the WinoMT benchmark and gender bias evaluation.",
            chunk_index=1,
        ),
    ]

    repo.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = repo.keyword_query("Where is WinoMT discussed?", 2)

    assert result[0].id == "doc.pdf::0001"
    assert result[0].score > 0


def test_document_repo_keyword_query_ignores_question_stopwords(configure_env) -> None:
    configure_env()
    repo = DocumentRepo()
    chunks = [
        make_chunk(
            chunk_id="doc.pdf::0000",
            text="This unrelated research paper discusses social media.",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="doc.pdf::0001",
            text="The title page lists Facebook AI Research as the affiliation.",
            chunk_index=1,
        ),
    ]

    repo.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = repo.keyword_query("Which paper is from Facebook AI Research?", 2)

    assert result[0].id == "doc.pdf::0001"


def test_document_repo_lists_documents_by_pages_and_chunks(configure_env) -> None:
    configure_env()
    repo = DocumentRepo()
    chunks = [
        make_chunk(chunk_id="a.pdf::0000", document="a.pdf", page=1, chunk_index=0),
        make_chunk(chunk_id="a.pdf::0001", document="a.pdf", page=1, chunk_index=1),
        make_chunk(chunk_id="a.pdf::0002", document="a.pdf", page=2, chunk_index=2),
        make_chunk(chunk_id="b.pdf::0000", document="b.pdf", page=3, chunk_index=0),
    ]

    repo.add_chunks(
        chunks,
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    assert [document.model_dump() for document in repo.list_documents()] == [
        {"document": "a.pdf", "pages": 2, "chunks": 3},
        {"document": "b.pdf", "pages": 1, "chunks": 1},
    ]


def test_document_repo_get_section_orders_chunks_by_index(configure_env) -> None:
    configure_env()
    repo = DocumentRepo()
    chunks = [
        make_chunk(chunk_id="doc.pdf::0001", text="second", chunk_index=1),
        make_chunk(chunk_id="doc.pdf::0000", text="first", chunk_index=0),
    ]

    repo.add_chunks(chunks, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    section = repo.get_section("doc.pdf", 1)
    assert section is not None
    assert section.text == "first\nsecond"
    assert repo.get_section("missing.pdf", 1) is None


def test_document_repo_upsert_is_idempotent_and_updates_text(configure_env) -> None:
    configure_env()
    repo = DocumentRepo()

    repo.add_chunks(
        [make_chunk(chunk_id="doc.pdf::0000", text="old", section=None)],
        [[1.0, 0.0, 0.0]],
    )
    repo.add_chunks(
        [make_chunk(chunk_id="doc.pdf::0000", text="new", section=None)],
        [[1.0, 0.0, 0.0]],
    )

    got = repo.collection.get(ids=["doc.pdf::0000"], include=["documents"])
    assert repo.count() == 1
    assert got["documents"] == ["new"]


def test_document_repo_guards_and_empty_query(configure_env) -> None:
    configure_env()
    repo = DocumentRepo()

    assert repo.query([1.0, 0.0, 0.0], 1) == []

    with pytest.raises(ValueError, match="same length"):
        repo.add_chunks([make_chunk()], [])


def test_document_repo_surfaces_empty_section_metadata_as_none(configure_env) -> None:
    configure_env()
    repo = DocumentRepo()

    repo.add_chunks(
        [make_chunk(chunk_id="doc.pdf::0000", text="alpha", section=None)],
        [[1.0, 0.0, 0.0]],
    )

    got = repo.collection.get(ids=["doc.pdf::0000"], include=["metadatas"])
    result = repo.query([1.0, 0.0, 0.0], 1)
    assert got["metadatas"][0]["section"] == ""
    assert result[0].section is None


def test_retrieval_service_embeds_query_and_uses_requested_top_k(
    monkeypatch,
    configure_env,
) -> None:
    configure_env(top_k=7, retrieval_candidate_k=11, vector_weight=0, lexical_weight=1)
    calls: dict[str, object] = {}

    class FakeEmbeddings:
        def embed_query(self, question: str) -> list[float]:
            calls["question"] = question
            return [0.1, 0.2, 0.3]

    class FakeDocuments:
        def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
            calls["query"] = (vector, top_k)
            return [make_retrieved(document="vector.pdf", score=0.9)]

        def keyword_query(self, question: str, top_k: int) -> list[RetrievedChunk]:
            calls["keyword_query"] = (question, top_k)
            return [make_retrieved(document="lexical.pdf", score=4.0)]

    monkeypatch.setattr(
        retrieval_module,
        "get_embeddings_service",
        lambda: FakeEmbeddings(),
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_repos",
        lambda: SimpleNamespace(documents=FakeDocuments()),
    )

    service = retrieval_module.RetrievalService()

    result = service.retrieve("What happened?", top_k=4)
    assert result[0].document == "lexical.pdf"
    assert calls == {
        "question": "What happened?",
        "query": ([0.1, 0.2, 0.3], 11),
        "keyword_query": ("What happened?", 11),
    }

    service.retrieve("Default top k")
    assert calls["query"] == ([0.1, 0.2, 0.3], 11)


def test_retrieval_service_can_disable_hybrid_retrieval(
    monkeypatch,
    configure_env,
) -> None:
    configure_env(enable_hybrid_retrieval=False, retrieval_candidate_k=10)

    class FakeEmbeddings:
        def embed_query(self, question: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    class FakeDocuments:
        def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
            return [
                make_retrieved(document="first.pdf", score=0.9),
                make_retrieved(document="second.pdf", score=0.8),
            ]

        def keyword_query(self, question: str, top_k: int) -> list[RetrievedChunk]:
            raise AssertionError("keyword retrieval should be disabled")

    monkeypatch.setattr(
        retrieval_module,
        "get_embeddings_service",
        lambda: FakeEmbeddings(),
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_repos",
        lambda: SimpleNamespace(documents=FakeDocuments()),
    )

    result = retrieval_module.RetrievalService().retrieve("What happened?", top_k=1)

    assert [chunk.document for chunk in result] == ["first.pdf"]


def test_retrieval_helpers_format_citations_and_context() -> None:
    long_text = "word " * 80
    chunks = [
        make_retrieved(text=long_text, document="a.pdf", page=2, section="Methods"),
        make_retrieved(text="short", document="b.pdf", page=3, section=None, score=0.1),
    ]

    citations = retrieval_helpers.to_citations(chunks)
    context = retrieval_helpers.to_context(chunks)

    assert len(citations[0].snippet) <= 240
    assert citations[0].snippet.endswith("…")
    assert citations[0].score == 0.9877
    assert citations[0].section == "Methods"
    assert "[1] a.pdf (p2) — Methods" in context
    assert "[2] b.pdf (p3)\nshort" in context


def test_synthesis_returns_empty_result_for_no_chunks(configure_env) -> None:
    configure_env()

    answer = synthesis_module.SynthesisService().answer("question", [])

    assert answer.synthesized is False
    assert "couldn't find" in answer.answer
    assert answer.citations == []


@pytest.mark.parametrize(
    ("openai_key", "enable_synthesis"),
    [("", True), ("test-key", False)],
)
def test_synthesis_falls_back_without_client_or_when_disabled(
    configure_env,
    openai_key: str,
    enable_synthesis: bool,
) -> None:
    configure_env(openai_key=openai_key, enable_synthesis=enable_synthesis)

    answer = synthesis_module.SynthesisService().answer("question", [make_retrieved()])

    assert answer.synthesized is False
    assert "Synthesis disabled" in answer.answer
    assert len(answer.citations) == 1


def test_synthesis_happy_path_uses_mocked_chat_response(
    monkeypatch,
    configure_env,
) -> None:
    configure_env(openai_key="test-key", enable_synthesis=True)

    class FakeOpenAI:
        def __init__(self, api_key: str | None) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs: object) -> SimpleNamespace:
            message = SimpleNamespace(content="mocked answer [1]")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(synthesis_module, "OpenAI", FakeOpenAI)

    answer = synthesis_module.SynthesisService().answer("question", [make_retrieved()])

    assert answer.synthesized is True
    assert answer.answer == "mocked answer [1]"
    assert len(answer.citations) == 1


def test_synthesis_degrades_to_fallback_when_llm_raises(
    monkeypatch,
    configure_env,
) -> None:
    configure_env(openai_key="test-key", enable_synthesis=True)

    class FakeOpenAI:
        def __init__(self, api_key: str | None) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs: object) -> SimpleNamespace:
            raise RuntimeError("boom")

    monkeypatch.setattr(synthesis_module, "OpenAI", FakeOpenAI)

    answer = synthesis_module.SynthesisService().answer("question", [make_retrieved()])

    assert answer.synthesized is False
    assert "Synthesis disabled" in answer.answer
    assert len(answer.citations) == 1


@pytest.mark.anyio
async def test_server_lists_exactly_the_document_tools() -> None:
    from app.server import mcp

    async with Client(mcp) as client:
        tools = await client.list_tools()

    assert sorted(tool.name for tool in tools) == [
        "get_document_section",
        "list_documents",
        "query_documents",
    ]


@pytest.mark.anyio
async def test_server_advertises_tool_schemas_and_annotations() -> None:
    from app.server import mcp

    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    for tool in tools.values():
        assert tool.description
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False
        assert tool.outputSchema is not None

    query_schema = tools["query_documents"].inputSchema
    assert query_schema["properties"]["question"]["minLength"] == 1
    assert "Natural-language question" in query_schema["properties"]["question"]["description"]
    top_k_schema = query_schema["properties"]["top_k"]["anyOf"][0]
    assert top_k_schema["minimum"] == 1
    assert top_k_schema["maximum"] == 20
    assert "Facebook AI Research" in tools["query_documents"].description

    section_schema = tools["get_document_section"].inputSchema
    assert section_schema["properties"]["document"]["minLength"] == 1
    assert section_schema["properties"]["page"]["minimum"] == 1

    assert tools["list_documents"].inputSchema["properties"] == {}


def test_get_document_section_handler_raises_when_repo_returns_none(
    monkeypatch,
) -> None:
    class FakeDocuments:
        def get_section(self, document: str, page: int) -> None:
            return None

    monkeypatch.setattr(
        handlers_module,
        "get_repos",
        lambda: SimpleNamespace(documents=FakeDocuments()),
    )

    with pytest.raises(ValueError, match="No indexed content"):
        handlers_module.get_document_section(GetSectionRequest(document="missing.pdf", page=1))


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "valid", "top_k": 0},
        {"question": "valid", "top_k": 21},
    ],
)
def test_query_contract_validation(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        QueryDocumentsRequest(**payload)


def test_retrieval_eval_scores_document_page_and_rank_hits() -> None:
    case = GoldenCase(
        id="gold",
        question="Where is the answer?",
        expected_sources=[ExpectedSource(document="expected.pdf", pages={2})],
        rationale="unit test",
    )
    chunks = [
        make_retrieved(document="other.pdf", page=1, score=0.9),
        make_retrieved(document="expected.pdf", page=2, score=0.8),
    ]

    result = evaluate_case(case, chunks)

    assert result.document_recall == 1.0
    assert result.page_recall == 1.0
    assert result.reciprocal_rank == 0.5
    assert result.passed is True


def test_retrieval_eval_summary_averages_component_metrics() -> None:
    passing = GoldenCase(
        id="pass",
        question="passing",
        expected_sources=[ExpectedSource(document="expected.pdf", pages={1})],
        rationale="unit test",
    )
    failing = GoldenCase(
        id="fail",
        question="failing",
        expected_sources=[ExpectedSource(document="missing.pdf", pages={3})],
        rationale="unit test",
    )

    summary = summarize(
        [
            evaluate_case(passing, [make_retrieved(document="expected.pdf", page=1)]),
            evaluate_case(failing, [make_retrieved(document="other.pdf", page=1)]),
        ]
    )

    assert summary.cases == 2
    assert summary.document_recall == 0.5
    assert summary.page_recall == 0.5
    assert summary.mrr == 0.5
    assert summary.exact_source_pass_rate == 0.5
