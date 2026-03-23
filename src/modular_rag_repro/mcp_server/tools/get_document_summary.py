"""MCP tool: get_document_summary。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mcp import types

from modular_rag_repro.mcp_server.catalog import KnowledgeCatalog
from modular_rag_repro.settings import load_settings

TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = "按 doc_id 返回文档摘要、来源路径和 chunk 数量。"
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string", "description": "文档 ID，例如 doc_xxx。"},
        "collection": {"type": "string", "description": "可选 collection 名称。"},
    },
    "required": ["doc_id"],
}


async def handler(doc_id: str, collection: Optional[str] = None) -> types.CallToolResult:
    """返回单个文档摘要。"""
    if not doc_id or not doc_id.strip():
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="参数错误: doc_id 不能为空")],
            isError=True,
        )

    settings = load_settings()
    catalog = KnowledgeCatalog(settings)
    try:
        summary = catalog.get_document_summary(doc_id=doc_id, collection=collection)
    except Exception as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {exc}")],
            isError=True,
        )

    text = "\n".join(
        [
            f"## Document: {summary.title}",
            "",
            f"doc_id={summary.doc_id}",
            f"collection={summary.collection}",
            f"source_path={summary.source_path}",
            f"chunk_count={summary.chunk_count}",
            f"summary={summary.summary}",
        ]
    )
    structured = {
        "doc_id": summary.doc_id,
        "collection": summary.collection,
        "title": summary.title,
        "source_path": summary.source_path,
        "chunk_count": summary.chunk_count,
        "summary": summary.summary,
        "tags": summary.tags,
        "metadata": summary.metadata,
    }
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
