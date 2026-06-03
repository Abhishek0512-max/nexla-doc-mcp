"""Application entrypoint for the Nexla document MCP server."""

from __future__ import annotations

from fastmcp import FastMCP

from app.routes.documents import register_document_routes

mcp = FastMCP("Nexla Document MCP")
register_document_routes(mcp)


if __name__ == "__main__":
    mcp.run()
