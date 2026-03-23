#!/usr/bin/env python
"""评估命令行骨架。"""

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
    """解析评估命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基于 Golden Test Set 执行评估。",
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Golden Test Set JSON 文件路径。",
    )
    parser.add_argument("--collection", default=None, help="要评估的集合名称。")
    parser.add_argument("--top-k", type=int, default=10, help="每次查询最多召回多少个 chunk。")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果。")
    parser.add_argument("--no-search", action="store_true", help="跳过检索阶段，只验证评估骨架。")
    parser.add_argument(
        "--config",
        default=str(REPRO_ROOT / "config" / "settings.yaml"),
        help="配置文件路径，默认使用 config/settings.yaml。",
    )
    return parser.parse_args()


def main() -> int:
    """运行评估命令入口。"""
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.observability.log_level)

    print("[Phase 1] 评估命令骨架")
    print(f"config={args.config}")
    print(f"test_set={args.test_set}")
    print(f"collection={args.collection}")
    print(f"top_k={args.top_k}")
    print(f"json={args.json}")
    print(f"no_search={args.no_search}")
    print("next_step=实现评估执行器与报告输出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
