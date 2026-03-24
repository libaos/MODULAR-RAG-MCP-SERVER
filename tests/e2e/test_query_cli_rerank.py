"""查询 CLI 的 rerank 冒烟测试。"""

from __future__ import annotations

from helpers import SAMPLE_PDF, cleanup_collection, load_test_settings, run_script, update_test_config


def test_query_cli_with_simple_rerank(temp_config_path, unique_collection: str) -> None:
    """打开 simple rerank 后，CLI 应显示 rerank 已应用。"""
    update_test_config(
        temp_config_path,
        lambda payload: payload["rerank"].update({"enabled": True, "provider": "simple", "top_k": 3}),
    )
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
        assert "Reranking applied by provider=simple." in query.stdout
        assert "FORMATTED RESPONSE" in query.stdout
    finally:
        cleanup_collection(settings, unique_collection)
