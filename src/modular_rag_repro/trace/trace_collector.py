"""Trace 持久化器。"""

from __future__ import annotations

import json
from pathlib import Path

from modular_rag_repro.settings import resolve_path
from modular_rag_repro.types import TraceContext


class TraceCollector:
    """把 TraceContext 追加写入 JSONL 文件。"""

    def __init__(self, trace_file: str = "./logs/traces.jsonl") -> None:
        self.path = resolve_path(trace_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """把单条 trace 追加到文件。"""
        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
