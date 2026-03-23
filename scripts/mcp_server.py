#!/usr/bin/env python
"""复现版 MCP Server 启动脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPRO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from modular_rag_repro.mcp_server.server import main


if __name__ == "__main__":
    raise SystemExit(main())
