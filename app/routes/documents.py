"""MCP tool routes for document querying and inspection."""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from app.contracts.documents import GetSectionRequest, QueryDocumentsRequest
from app.handlers.documents import (
    get_document_section as handle_get_document_section,
)
from app.handlers.documents import list_documents as handle_list_documents
from app.handlers.documents import query_documents as handle_query_documents
from app.schemas.documents import Answer, DocumentInfo, Section

_READ_ONLY_TOOL_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


def register_document_routes(server: FastMCP) -> None:
    """Register document-related MCP tools on the server."""

    @server.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
    def query_documents(
        question: Annotated[
            str,
            Field(
                min_length=1,
                description="Natural-language question to answer from the indexed PDFs.",
            ),
        ],
        top_k: Annotated[
            int | None,
            Field(
                ge=1,
                le=20,
                description="Optional number of retrieved chunks to consider before answering.",
            ),
        ] = None,
    ) -> Answer:
        """Answer a natural-language question from indexed PDFs with citations.

        Use this for grounded Q&A across the indexed corpus. Example:
        "Which papers are from Facebook AI Research?"
        """
        request = QueryDocumentsRequest(question=question, top_k=top_k)
        return handle_query_documents(request)

    @server.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
    def list_documents() -> list[DocumentInfo]:
        """List indexed PDF documents with page and chunk counts.

        Use this before querying to verify which documents are available.
        """
        return handle_list_documents()

    @server.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
    def get_document_section(
        document: Annotated[
            str,
            Field(
                min_length=1,
                description="Exact indexed PDF file name, such as 'W18-4401.pdf'.",
            ),
        ],
        page: Annotated[
            int,
            Field(ge=1, description="1-based page number to retrieve."),
        ],
    ) -> Section:
        """Return indexed text for one document page.

        Use this to inspect or expand a cited source. Example:
        document="W18-4401.pdf", page=3.
        """
        request = GetSectionRequest(document=document, page=page)
        return handle_get_document_section(request)
