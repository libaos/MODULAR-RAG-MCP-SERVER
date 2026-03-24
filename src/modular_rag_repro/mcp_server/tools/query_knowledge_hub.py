"""MCP tool: query_knowledge_hub。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mcp import types

from modular_rag_repro.query_engine import QueryWorkflow
from modular_rag_repro.settings import load_settings

TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = "使用复现版知识库执行 hybrid search，并返回格式化结果。"
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "查询文本。"},
        "top_k": {
            "type": "integer",
            "description": "返回结果上限。",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {"type": "string", "description": "限定查询的集合。"},
    },
    "required": ["query"],
}


async def handler(query: str, top_k: int = 5, collection: Optional[str] = None) -> types.CallToolResult:
    """执行知识库查询。"""
    if not query or not query.strip():
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="参数错误: query 不能为空")],
            isError=True,
        )

    settings = load_settings()
    target_collection = collection or settings.vector_store.collection_name
    workflow = QueryWorkflow(settings)
    workflow_result = workflow.run(
        query=query,
        collection=target_collection,
        top_k=top_k,
        no_rerank=not settings.rerank.enabled,
        source="mcp",
    )
    response = workflow_result.formatted_response
    text = workflow.formatter.render_text(response)
    return types.CallToolResult(
        content=[
            types.TextContent(type="text", text=text),
            types.TextContent(type="text", text=json.dumps(response.to_dict(), ensure_ascii=False, indent=2)),
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
