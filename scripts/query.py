#!/usr/bin/env python
"""知识库查询命令行骨架。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPRO_ROOT / "src"
# 让脚本可以直接导入 `src/` 下的复现版包。
sys.path.insert(0, str(SRC_ROOT))

from modular_rag_repro.logging_utils import configure_logging
from modular_rag_repro.settings import load_settings


def parse_args() -> argparse.Namespace:
    """解析查询命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从复现版知识库中执行查询。",
    )
    parser.add_argument("--query", "-q", required=True, help="查询文本。")
    parser.add_argument("--collection", "-c", default="default", help="目标集合名称。")
    parser.add_argument("--top-k", type=int, default=10, help="最多返回多少条结果。")
    parser.add_argument(
        "--config",
        default=str(REPRO_ROOT / "config" / "settings.yaml"),
        help="配置文件路径，默认使用 config/settings.yaml。",
    )
    parser.add_argument("--no-rerank", action="store_true", help="后续即使打开 rerank 也在本次查询中关闭。")
    parser.add_argument("--verbose", action="store_true", help="展示中间阶段详情。")
    return parser.parse_args()


def main() -> int:
    """运行查询命令入口。

    当前阶段只验证参数解析、配置加载和日志初始化是否正常。
    """
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging("DEBUG" if args.verbose else settings.observability.log_level)

    print("[Phase 1] 查询命令骨架")
    print(f"config={args.config}")
    print(f"query={args.query}")
    print(f"collection={args.collection}")
    print(f"top_k={args.top_k}")
    print(f"no_rerank={args.no_rerank}")
    print("next_step=实现 QueryProcessor、检索器、融合与响应格式化")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
