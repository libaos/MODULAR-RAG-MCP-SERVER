#!/usr/bin/env python
"""知识库查询命令行入口。

当前阶段已经接入最小查询链路：

- QueryProcessor
- DenseRetriever
- SparseRetriever
- RRF Fusion
- ResponseFormatter
- 分路结果与最终结果打印

还没有接入：

- rerank
- 生成式回答
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPRO_ROOT / "src"
# 让脚本可以直接导入 `src/` 下的复现版包。
sys.path.insert(0, str(SRC_ROOT))

from modular_rag_repro.logging_utils import configure_logging
from modular_rag_repro.query_engine import QueryWorkflow
from modular_rag_repro.settings import load_settings
from modular_rag_repro.types import RetrievalResult


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

    当前阶段负责把最小 Hybrid 查询链路跑起来：

    query -> QueryProcessor -> DenseRetriever + SparseRetriever -> RRF Fusion -> ResponseFormatter
    """
    args = parse_args()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"[FAIL] 配置加载失败: {exc}")
        return 2

    configure_logging("DEBUG" if args.verbose else settings.observability.log_level)

    workflow = QueryWorkflow(settings)

    try:
        workflow_result = workflow.run(
            query=args.query,
            collection=args.collection,
            top_k=args.top_k,
            no_rerank=args.no_rerank,
            source="cli",
        )
    except Exception as exc:
        print(f"[FAIL] 查询执行失败: {exc}")
        return 2

    print("[*] Modular RAG Repro Query")
    print("=" * 60)
    print(f"config={args.config}")
    print(f"collection={args.collection}")
    print(f"top_k={args.top_k}")

    if args.verbose:
        filters_text = workflow_result.processed_query.filters if workflow_result.processed_query.filters else "(none)"
        print(f"[INFO] ProcessedQuery normalized_text={workflow_result.processed_query.normalized_text}")
        print(f"[INFO] ProcessedQuery keywords={workflow_result.processed_query.keywords} filters={filters_text}")

    print_result_section("DENSE RESULTS", workflow_result.dense_results, top_k=args.top_k)
    print_result_section("SPARSE RESULTS", workflow_result.sparse_results, top_k=args.top_k)
    print_result_section("FUSION RESULTS", workflow_result.fusion_results, top_k=args.top_k)

    if args.no_rerank or not settings.rerank.enabled:
        print("[INFO] Reranking disabled by settings.")
        final_results = workflow_result.final_results
    else:
        # 当前阶段还没实现 rerank，这里先保留接口和回退逻辑。
        print("[INFO] Reranker not implemented yet, fallback to fusion results.")
        final_results = workflow_result.final_results

    print_result_section("RESULTS", final_results, top_k=args.top_k)
    print()
    print(workflow.formatter.render_text(workflow_result.formatted_response))
    return 0


def print_result_section(title: str, results: Iterable[RetrievalResult], top_k: int) -> None:
    """打印某一路查询结果。"""
    results = list(results)
    print("\n" + "=" * 60)
    print(f"{title} (top_k={top_k}, returned={len(results)})")
    print("=" * 60)

    if not results:
        print("[INFO] 没有命中任何结果")
        print("=" * 60)
        return

    for index, item in enumerate(results, start=1):
        source_path = item.metadata.get("source_path", "(unknown)")
        chunk_index = item.metadata.get("chunk_index", "(unknown)")
        preview = build_preview(item.text)

        print(f"#{index:02d}  score={item.score:.4f}  id={item.chunk_id}")
        print(f"     source_path={source_path}")
        print(f"     chunk_index={chunk_index}")
        print(f"     text={preview}")

    print("=" * 60)


def build_preview(text: str, limit: int = 180) -> str:
    """把结果文本压成一行预览。"""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
