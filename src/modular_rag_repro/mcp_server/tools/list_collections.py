"""MCP tool: list_collections。"""

from __future__ import annotations

import json
from typing import Any, Dict

from mcp import types

from modular_rag_repro.mcp_server.catalog import KnowledgeCatalog
from modular_rag_repro.settings import load_settings

TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = "列出当前本地知识库中的 collection，并可附带统计信息。"
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_stats": {
            "type": "boolean",
            "description": "是否返回 Chroma/BM25 统计。",
            "default": True,
        }
    },
    "required": [],
}


async def handler(include_stats: bool = True) -> types.CallToolResult:
    """列出 collection。"""
    settings = load_settings()
    catalog = KnowledgeCatalog(settings)
    collections = catalog.list_collections(include_stats=include_stats)

    if not collections:
        text = "No collections found in the local knowledge base."
    else:
        lines = [f"## Collections ({len(collections)})", ""]
        for item in collections:
            line = f"- {item.name}"
            if include_stats:
                line += f" | chroma={item.chroma_count} | bm25={item.bm25_count}"
            lines.append(line)
        text = "\n".join(lines)

    structured = [
        {
            "name": item.name,
            "chroma_count": item.chroma_count,
            "bm25_count": item.bm25_count,
            "metadata": item.metadata,
        }
        for item in collections
    ]
    return types.CallToolResult(
        content=[
            types.TextContent(type="text", text=text),
            types.TextContent(type="text", text=json.dumps(structured, ensure_ascii=False, indent=2)),
        ],
        isError=False,
    )


def register_tool(protocol_handler) -> None:
    """注册 tool。"""
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
