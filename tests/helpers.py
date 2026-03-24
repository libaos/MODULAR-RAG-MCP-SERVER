"""测试辅助函数。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import yaml

from modular_rag_repro.ingestion import BM25Indexer, ChromaUpserter, IngestionPipeline
from modular_rag_repro.settings import load_settings


REPRO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPRO_ROOT.parent
SAMPLE_PDF = WORKSPACE_ROOT / "tests" / "fixtures" / "sample_documents" / "simple.pdf"


def unique_collection_name(prefix: str = "repro-test") -> str:
    """生成一个当前测试专用的 collection 名称。"""
    return f"{prefix}-{uuid4().hex[:8]}"


def make_temp_config(tmp_path: Path) -> Path:
    """基于 repro 默认配置生成一份临时测试配置。"""
    base_path = REPRO_ROOT / "config" / "settings.yaml"
    payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    chroma_dir = (tmp_path / "chroma").resolve()
    trace_file = (tmp_path / "logs" / "traces.jsonl").resolve()
    integrity_db = (tmp_path / "ingestion_history" / "history.sqlite3").resolve()
    payload["vector_store"]["persist_directory"] = str(chroma_dir)
    payload["observability"]["trace_file"] = str(trace_file)
    payload["ingestion"]["integrity_db_path"] = str(integrity_db)

    config_path = tmp_path / "settings.test.yaml"
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def load_test_settings(config_path: Path | str):
    """加载测试配置。"""
    return load_settings(str(config_path))


def cleanup_collection(settings, collection: str) -> None:
    """清理某个 collection 在 Chroma 和 BM25 中的测试数据。"""
    try:
        ChromaUpserter(settings, collection=collection).delete_collection(collection)
    except Exception:
        pass

    try:
        BM25Indexer().delete_index(collection)
    except Exception:
        pass


def seed_collection(settings, collection: str, pdf_path: Path | None = None):
    """把样例 PDF 灌入测试 collection。"""
    target_pdf = pdf_path or SAMPLE_PDF
    pipeline = IngestionPipeline(settings, collection=collection)
    result = pipeline.run(str(target_pdf))
    if not result.success:
        raise AssertionError(f"seed_collection failed: {result.error}")
    return result


def run_script(args: list[str], cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """运行一个 repro 脚本命令。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd or REPRO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def list_evaluation_reports() -> set[Path]:
    """列出当前已有评估报告。"""
    evaluation_dir = REPRO_ROOT / "data" / "evaluation"
    if not evaluation_dir.exists():
        return set()
    return set(evaluation_dir.glob("evaluation-report-*.json"))


def remove_paths(paths: set[Path]) -> None:
    """删除一组测试生成的文件。"""
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


def parse_json_content(text: str) -> dict:
    """从文本中解析 JSON。"""
    return json.loads(text)
