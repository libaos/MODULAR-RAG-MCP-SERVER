"""MCP Server 启动入口。"""

from __future__ import annotations

import asyncio
import logging
import sys

from modular_rag_repro.logging_utils import get_logger
from modular_rag_repro.mcp_server.protocol_handler import create_mcp_server

SERVER_NAME = "modular-rag-repro"
SERVER_VERSION = "0.1.0"


def _redirect_logs_to_stderr() -> None:
    """把日志统一重定向到 stderr，避免污染 stdio 协议流。"""
    root = logging.getLogger()
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)
    root.setLevel(logging.INFO)


async def run_stdio_server_async() -> int:
    """异步运行 stdio MCP server。"""
    import mcp.server.stdio

    _redirect_logs_to_stderr()
    logger = get_logger(__name__)
    logger.info("Starting repro MCP server over stdio.")

    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

    logger.info("Repro MCP server stopped.")
    return 0


def run_stdio_server() -> int:
    """同步包装器。"""
    return asyncio.run(run_stdio_server_async())


def main() -> int:
    """脚本入口。"""
    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())
