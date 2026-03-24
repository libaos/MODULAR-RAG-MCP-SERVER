"""查询 CLI 生成式回答冒烟测试。"""

from __future__ import annotations

from helpers import SAMPLE_PDF, cleanup_collection, load_test_settings, run_script, update_test_config


def test_query_cli_prints_generated_answer(temp_config_path, unique_collection: str) -> None:
    """打开本地生成式回答后，CLI 应打印 GENERATED ANSWER 段。"""
    update_test_config(
        temp_config_path,
        lambda payload: payload.update(
            {
                "answer_generation": {
                    "enabled": True,
                    "provider": "local",
                    "model": "local-template",
                    "max_context_chunks": 2,
                    "timeout_seconds": 30,
                }
            }
        ),
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
            ]
        )

        assert query.returncode == 0, query.stdout + "\n" + query.stderr
        assert "GENERATED ANSWER" in query.stdout
    finally:
        cleanup_collection(settings, unique_collection)
