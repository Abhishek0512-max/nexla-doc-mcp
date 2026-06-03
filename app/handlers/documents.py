"""Handlers — per-tool orchestration. Routes call these; these call services.

Each function takes a validated contract, composes the service calls needed to
fulfil that one tool, and returns a schema. No business logic lives here.
"""

from __future__ import annotations

from app.contracts.documents import GetSectionRequest, QueryDocumentsRequest
from app.repos.repo_container import get_repos
from app.schemas.documents import Answer, DocumentInfo, Section
from app.services.retrieval.service import get_retrieval_service
from app.services.synthesis.service import get_synthesis_service


def query_documents(request: QueryDocumentsRequest) -> Answer:
    chunks = get_retrieval_service().retrieve(request.question, request.top_k)
    return get_synthesis_service().answer(request.question, chunks)


def list_documents() -> list[DocumentInfo]:
    return get_repos().documents.list_documents()


def get_document_section(request: GetSectionRequest) -> Section:
    section = get_repos().documents.get_section(request.document, request.page)
    if section is None:
        raise ValueError(f"No indexed content for '{request.document}' page {request.page}.")
    return section
