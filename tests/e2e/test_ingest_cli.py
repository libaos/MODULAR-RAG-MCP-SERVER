"""摄取 CLI 冒烟测试。"""

from __future__ import annotations

from helpers import SAMPLE_PDF, cleanup_collection, run_script


def test_ingest_cli_smoke(temp_config_path, unique_collection: str) -> None:
    """`scripts/ingest.py` 应成功完成一次最小摄取。"""
    from helpers import load_test_settings

    settings = load_test_settings(temp_config_path)
    cleanup_collection(settings, unique_collection)
    try:
        completed = run_script(
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

        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
        assert "[OK] doc_id=" in completed.stdout
        assert "total_upserted=" in completed.stdout
    finally:
        cleanup_collection(settings, unique_collection)
