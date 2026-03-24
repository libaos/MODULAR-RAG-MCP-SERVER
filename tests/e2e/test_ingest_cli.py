"""摄取 CLI 冒烟测试。"""

from __future__ import annotations

from helpers import SAMPLE_PDF, cleanup_collection, run_script


def test_ingest_cli_smoke(temp_config_path, unique_collection: str) -> None:
    """`scripts/ingest.py` 应支持首次摄取、skip 和 force 重跑。"""
    from helpers import load_test_settings

    settings = load_test_settings(temp_config_path)
    cleanup_collection(settings, unique_collection)
    try:
        first = run_script(
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

        assert first.returncode == 0, first.stdout + "\n" + first.stderr
        assert "[OK] doc_id=" in first.stdout
        assert "skipped=0" in first.stdout

        second = run_script(
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
        assert second.returncode == 0, second.stdout + "\n" + second.stderr
        assert "[SKIP] reason=already_processed" in second.stdout
        assert "skipped=1" in second.stdout

        forced = run_script(
            [
                "scripts/ingest.py",
                "--path",
                str(SAMPLE_PDF),
                "--collection",
                unique_collection,
                "--config",
                str(temp_config_path),
                "--force",
            ]
        )
        assert forced.returncode == 0, forced.stdout + "\n" + forced.stderr
        assert "[OK] doc_id=" in forced.stdout
        assert "[SKIP]" not in forced.stdout
    finally:
        cleanup_collection(settings, unique_collection)
