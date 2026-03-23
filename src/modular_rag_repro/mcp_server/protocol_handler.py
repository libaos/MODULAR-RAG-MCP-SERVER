"""MCP 协议处理器。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from mcp import types
from mcp.server.lowlevel import Server

from modular_rag_repro.logging_utils import get_logger


ToolHandler = Callable[..., Awaitable[types.CallToolResult] | types.CallToolResult | str | List[Any]]


@dataclass(slots=True)
class ToolDefinition:
    """MCP tool 定义。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: ToolHandler


@dataclass(slots=True)
class ProtocolHandler:
    """负责注册和执行 MCP tools。"""

    server_name: str
    server_version: str
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)
    logger: Any = field(init=False, repr=False)
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = get_logger(__name__)
        self._lock = asyncio.Lock()

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """注册一个 tool。"""
        if name in self.tools:
            raise ValueError(f"Tool '{name}' already registered")
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )

    def get_tool_schemas(self) -> List[types.Tool]:
        """返回 tools/list 需要的 schema 列表。"""
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in self.tools.values()
        ]

    async def execute_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> types.CallToolResult:
        """执行一个 tool。"""
        if name not in self.tools:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Tool '{name}' not found")],
                isError=True,
            )

        tool = self.tools[name]
        arguments = arguments or {}

        try:
            async with self._lock:
                result = tool.handler(**arguments)
                if asyncio.iscoroutine(result):
                    result = await result

                if isinstance(result, types.CallToolResult):
                    return result
                if isinstance(result, str):
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=result)],
                        isError=False,
                    )
                if isinstance(result, list):
                    return types.CallToolResult(content=result, isError=False)
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(result))],
                    isError=False,
                )
        except TypeError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Invalid parameters - {exc}")],
                isError=True,
            )
        except Exception:
            self.logger.exception("Tool execution failed: %s", name)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Internal server error while executing '{name}'")],
                isError=True,
            )


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: Optional[ProtocolHandler] = None,
    register_tools: bool = True,
) -> Server:
    """创建并配置一个 stdio MCP server。"""
    if protocol_handler is None:
        protocol_handler = ProtocolHandler(server_name=server_name, server_version=server_version)

    if register_tools:
        from modular_rag_repro.mcp_server.tools.get_document_summary import register_tool as register_summary_tool
        from modular_rag_repro.mcp_server.tools.list_collections import register_tool as register_list_tool
        from modular_rag_repro.mcp_server.tools.query_knowledge_hub import register_tool as register_query_tool

        register_query_tool(protocol_handler)
        register_list_tool(protocol_handler)
        register_summary_tool(protocol_handler)

    server = Server(server_name)

    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        return protocol_handler.get_tool_schemas()

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> types.CallToolResult:
        return await protocol_handler.execute_tool(name, arguments)

    server._protocol_handler = protocol_handler  # type: ignore[attr-defined]
    return server
