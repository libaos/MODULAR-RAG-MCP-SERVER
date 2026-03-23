#!/usr/bin/env python
"""文档摄取命令行骨架。"""

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
    """解析摄取命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将文档摄取到复现版知识库。",
    )
    parser.add_argument("--path", "-p", required=True, help="文件或目录路径。")
    parser.add_argument("--collection", "-c", default="default", help="目标集合名称。")
    parser.add_argument("--force", "-f", action="store_true", help="即使已处理过也重新执行。")
    parser.add_argument(
        "--config",
        default=str(REPRO_ROOT / "config" / "settings.yaml"),
        help="配置文件路径，默认使用 config/settings.yaml。",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细日志。")
    parser.add_argument("--dry-run", action="store_true", help="仅展示计划执行内容。")
    return parser.parse_args()


def main() -> int:
    """运行摄取命令入口。

    当前阶段只验证参数解析、配置加载和日志初始化是否正常。
    """
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging("DEBUG" if args.verbose else settings.observability.log_level)

    print("[Phase 1] 摄取命令骨架")
    print(f"config={args.config}")
    print(f"path={args.path}")
    print(f"collection={args.collection}")
    print(f"force={args.force}")
    print(f"dry_run={args.dry_run}")
    print("next_step=实现 PDF Loader、Chunker、编码器与存储流水线")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
