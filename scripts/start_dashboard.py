#!/usr/bin/env python
"""复现版 Dashboard 启动脚本。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    """解析 Dashboard 启动参数。"""
    parser = argparse.ArgumentParser(description="启动复现版 Dashboard。")
    parser.add_argument("--port", type=int, default=8501, help="服务端口。")
    parser.add_argument("--host", type=str, default="localhost", help="绑定地址。")
    return parser.parse_args()


def main() -> int:
    """启动 Streamlit 占位页面。"""
    args = parse_args()
    app_path = REPRO_ROOT / "src" / "modular_rag_repro" / "dashboard" / "app.py"
    if not app_path.exists():
        print(f"未找到 Dashboard 应用入口: {app_path}")
        return 1

    # 先复用最稳的 Streamlit 非交互参数，避免首次启动卡住。
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--server.headless",
        "true",
        "--server.showEmailPrompt",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    print(f"正在启动 Dashboard: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
