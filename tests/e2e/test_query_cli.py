"""查询 CLI 冒烟测试。"""

from __future__ import annotations

from helpers import SAMPLE_PDF, cleanup_collection, load_test_settings, run_script


def test_query_cli_smoke(temp_config_path, unique_collection: str) -> None:
    """`scripts/query.py` 应输出完整查询结果段落。"""
    settings = load_test_settings(temp_config_path)
    cleanup_collection(settings, unique_collection)

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

        query = run_script(
            [
                "scripts/query.py",
                "--query",
                "sample pdf",
                "--collection",
                unique_collection,
                "--config",
                str(temp_config_path),
                "--top-k",
                "3",
                "--verbose",
            ]
        )

        assert query.returncode == 0, query.stdout + "\n" + query.stderr
        assert "DENSE RESULTS" in query.stdout
        assert "SPARSE RESULTS" in query.stdout
        assert "FUSION RESULTS" in query.stdout
        assert "FORMATTED RESPONSE" in query.stdout
    finally:
        cleanup_collection(settings, unique_collection)
