"""评估 CLI 冒烟测试。"""

from __future__ import annotations

import json
from pathlib import Path

from helpers import (
    SAMPLE_PDF,
    cleanup_collection,
    list_evaluation_reports,
    load_test_settings,
    remove_paths,
    run_script,
)


def test_evaluate_cli_smoke(tmp_path: Path, temp_config_path, unique_collection: str) -> None:
    """`scripts/evaluate.py` 应能跑出最小 Golden Set 报告。"""
    settings = load_test_settings(temp_config_path)
    cleanup_collection(settings, unique_collection)

    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "sample pdf",
                        "expected_chunk_ids": [],
                        "expected_sources": [str(SAMPLE_PDF)],
                        "collection": unique_collection,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    reports_before = list_evaluation_reports()
    try:
        ingest = run_script(
            [
                "scripts/ingest.py",
                "--path",
                str(SAMPLE_PDF),
                "--collection",
                unique_collection,
                "--config",
                str(temp_config_path),
            ]
        )
        assert ingest.returncode == 0, ingest.stdout + "\n" + ingest.stderr

        evaluate = run_script(
            [
                "scripts/evaluate.py",
                "--test-set",
                str(golden_path),
                "--collection",
                unique_collection,
                "--config",
                str(temp_config_path),
                "--top-k",
                "3",
            ]
        )
        assert evaluate.returncode == 0, evaluate.stdout + "\n" + evaluate.stderr
        assert "aggregate_metrics=" in evaluate.stdout
        assert "query_count=1" in evaluate.stdout
    finally:
        cleanup_collection(settings, unique_collection)
        reports_after = list_evaluation_reports()
        remove_paths(reports_after - reports_before)
