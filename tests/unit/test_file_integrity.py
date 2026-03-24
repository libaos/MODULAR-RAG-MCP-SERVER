"""`SQLiteIntegrityChecker` 的最小单元测试。"""

from __future__ import annotations

from pathlib import Path

from modular_rag_repro.ingestion import SQLiteIntegrityChecker


def test_compute_sha256_is_stable(tmp_path: Path) -> None:
    """同一个文件重复计算哈希应得到相同结果。"""
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"same-content")
    checker = SQLiteIntegrityChecker(str(tmp_path / "history.sqlite3"))

    first = checker.compute_sha256(file_path)
    second = checker.compute_sha256(file_path)

    assert first == second
    assert len(first) == 64


def test_success_record_only_skips_same_collection(tmp_path: Path) -> None:
    """去重应按 `file_hash + collection` 生效，而不是全局生效。"""
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"same-content")
    checker = SQLiteIntegrityChecker(str(tmp_path / "history.sqlite3"))
    file_hash = checker.compute_sha256(file_path)

    checker.mark_success(file_hash, file_path, "alpha")

    assert checker.should_skip(file_hash, "alpha") is True
    assert checker.should_skip(file_hash, "beta") is False


def test_failed_record_does_not_skip_and_remove_record_clears_state(tmp_path: Path) -> None:
    """失败记录不应触发 skip，删除记录后也不应继续 skip。"""
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"same-content")
    checker = SQLiteIntegrityChecker(str(tmp_path / "history.sqlite3"))
    file_hash = checker.compute_sha256(file_path)

    checker.mark_failed(file_hash, file_path, "alpha", "boom")
    assert checker.should_skip(file_hash, "alpha") is False

    checker.mark_success(file_hash, file_path, "alpha")
    assert checker.should_skip(file_hash, "alpha") is True

    removed = checker.remove_record(file_hash, "alpha")
    assert removed is True
    assert checker.should_skip(file_hash, "alpha") is False
