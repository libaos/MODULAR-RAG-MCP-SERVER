#!/usr/bin/env python
"""文档摄取命令行入口。

当前阶段已经接入最小摄取链路：

- 发现 PDF 文件
- 调用 `IngestionPipeline`
- 输出每个文件的执行结果和汇总

还没有接入：

- SHA256 去重
- 向量化
- BM25
- Chroma 入库
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPRO_ROOT / "src"
# 让脚本可以直接导入 `src/` 下的复现版包。
sys.path.insert(0, str(SRC_ROOT))

from modular_rag_repro.logging_utils import configure_logging
from modular_rag_repro.ingestion import IngestionPipeline, PipelineResult
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


def discover_files(path_value: str, extensions: List[str] | None = None) -> List[Path]:
    """发现要处理的 PDF 文件。

    支持两种输入：

    - 单个 PDF 文件
    - 一个目录（递归搜索 PDF）
    """
    path = Path(path_value)
    extensions = extensions or [".pdf"]

    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    if path.is_file():
        if path.suffix.lower() not in extensions:
            raise ValueError(f"暂不支持的文件类型: {path.suffix}")
        return [path]

    files: List[Path] = []
    for ext in extensions:
        files.extend(path.rglob(f"*{ext}"))
        files.extend(path.rglob(f"*{ext.upper()}"))

    return sorted(set(files))


def print_summary(results: List[PipelineResult]) -> None:
    """打印执行汇总。"""
    total = len(results)
    success_count = sum(1 for item in results if item.success)
    failed_count = total - success_count
    chunk_total = sum(item.chunk_count for item in results if item.success)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"total_files={total}")
    print(f"successful={success_count}")
    print(f"failed={failed_count}")
    print(f"total_chunks={chunk_total}")
    print("=" * 60)


def main() -> int:
    """运行摄取命令入口。

    当前阶段负责把最小摄取链路真正跑起来：

    文件 -> PdfLoader -> Document -> DocumentChunker -> PipelineResult
    """
    args = parse_args()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"[FAIL] 配置加载失败: {exc}")
        return 2

    configure_logging("DEBUG" if args.verbose else settings.observability.log_level)

    try:
        files = discover_files(args.path)
    except Exception as exc:
        print(f"[FAIL] 文件发现失败: {exc}")
        return 2

    print("[*] Modular RAG Repro Ingestion")
    print("=" * 60)
    print(f"config={args.config}")
    print(f"collection={args.collection}")
    print(f"force={args.force}")
    print(f"dry_run={args.dry_run}")
    print(f"found_files={len(files)}")

    for file_path in files:
        print(f"  - {file_path}")

    if not files:
        print("[WARN] 没有发现可处理的 PDF 文件")
        return 0

    if args.dry_run:
        print("[INFO] dry-run 模式，不执行实际摄取")
        return 0

    if args.force:
        print("[INFO] 当前阶段尚未接入 SHA256 去重，force 参数暂时只保留接口含义")

    pipeline = IngestionPipeline(settings, collection=args.collection)
    results: List[PipelineResult] = []

    print("\n[INFO] 开始执行摄取...\n")
    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path}")
        result = pipeline.run(str(file_path))
        results.append(result)

        if result.success:
            doc_id = result.document.id if result.document else "(none)"
            print(f"  [OK] doc_id={doc_id}")
            print(f"  [OK] chunk_count={result.chunk_count}")
        else:
            print(f"  [FAIL] error={result.error}")

    print_summary(results)

    if all(item.success for item in results):
        return 0
    if any(item.success for item in results):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
