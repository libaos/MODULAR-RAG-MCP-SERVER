"""MCP Server 冒烟测试。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from modular_rag_repro.settings import load_settings
from helpers import SAMPLE_PDF, WITH_IMAGES_PDF, REPRO_ROOT, cleanup_collection, seed_collection, unique_collection_name


INIT_REQUEST = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "clientInfo": {"name": "repro-e2e-client", "version": "1.0.0"},
        "capabilities": {},
    },
}

INITIALIZED_NOTIFICATION = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
}


def _start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [sys.executable, "scripts/mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPRO_ROOT),
        env=env,
    )


def _send_jsonrpc(proc: subprocess.Popen, messages: list[dict], expected_responses: int, timeout: float = 30.0):
    assert proc.stdin is not None
    assert proc.stdout is not None

    for msg in messages:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    responses: list[dict] = []
    stop = threading.Event()

    def _reader() -> None:
        while not stop.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in payload and ("result" in payload or "error" in payload):
                responses.append(payload)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while len(responses) < expected_responses and time.time() < deadline:
        time.sleep(0.1)
    stop.set()
    return responses


def _find(responses: list[dict], req_id: int):
    for response in responses:
        if response.get("id") == req_id:
            return response
    return None


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_mcp_server_smoke() -> None:
    """MCP server 应能完成 initialize/list/query/summary 最小流程。"""
    settings = load_settings()
    collection = unique_collection_name("mcp-smoke")
    cleanup_collection(settings, collection)
    seeded = seed_collection(settings, collection, SAMPLE_PDF)
    proc = _start_server()

    try:
        responses = _send_jsonrpc(
            proc,
            [
                INIT_REQUEST,
                INITIALIZED_NOTIFICATION,
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "query_knowledge_hub",
                        "arguments": {"query": "sample pdf", "top_k": 3, "collection": collection},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "get_document_summary",
                        "arguments": {"doc_id": seeded.document.id, "collection": collection},
                    },
                },
            ],
            expected_responses=4,
            timeout=60.0,
        )

        init_resp = _find(responses, 1)
        tools_resp = _find(responses, 2)
        query_resp = _find(responses, 3)
        summary_resp = _find(responses, 4)

        assert init_resp is not None
        assert tools_resp is not None
        assert query_resp is not None
        assert summary_resp is not None

        tool_names = {item["name"] for item in tools_resp["result"]["tools"]}
        assert "query_knowledge_hub" in tool_names
        assert "list_collections" in tool_names
        assert "get_document_summary" in tool_names

        query_text = " ".join(block.get("text", "") for block in query_resp["result"]["content"])
        assert "FORMATTED RESPONSE" in query_text
        assert collection in query_text

        summary_text = " ".join(block.get("text", "") for block in summary_resp["result"]["content"])
        assert seeded.document.id in summary_text
        assert "Sample Document" in summary_text
    finally:
        _terminate(proc)
        cleanup_collection(settings, collection)


def test_mcp_server_returns_image_blocks_for_multimodal_result() -> None:
    """查询带图文档时，MCP tool 应返回 image content block。"""
    settings = load_settings()
    collection = unique_collection_name("mcp-image")
    cleanup_collection(settings, collection)
    seed_collection(settings, collection, WITH_IMAGES_PDF)
    proc = _start_server()

    try:
        responses = _send_jsonrpc(
            proc,
            [
                INIT_REQUEST,
                INITIALIZED_NOTIFICATION,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "query_knowledge_hub",
                        "arguments": {"query": "image", "top_k": 3, "collection": collection},
                    },
                },
            ],
            expected_responses=2,
            timeout=60.0,
        )

        query_resp = _find(responses, 2)
        assert query_resp is not None
        assert any(block.get("type") == "image" for block in query_resp["result"]["content"])
    finally:
        _terminate(proc)
        cleanup_collection(settings, collection)
