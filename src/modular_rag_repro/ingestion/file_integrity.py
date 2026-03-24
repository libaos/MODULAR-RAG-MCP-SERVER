"""基于 SHA256 的最小摄取去重器。"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from modular_rag_repro.settings import resolve_path


class SQLiteIntegrityChecker:
    """按 `file_hash + collection` 记录摄取状态。"""

    def __init__(self, db_path: str = "data/db/ingestion_history/history.sqlite3") -> None:
        self.db_path = resolve_path(db_path)
        self._ensure_database()

    def compute_sha256(self, file_path: str | Path) -> str:
        """计算文件 SHA256。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"输入路径不是文件: {path}")

        sha256 = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def should_skip(self, file_hash: str, collection: str) -> bool:
        """判断当前 collection 下是否可以跳过。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT status
                FROM ingestion_history
                WHERE file_hash = ? AND collection = ?
                """,
                (file_hash, collection),
            )
            row = cursor.fetchone()
        return bool(row and row[0] == "success")

    def mark_success(self, file_hash: str, file_path: str | Path, collection: str) -> None:
        """记录成功处理。"""
        self._upsert(file_hash, file_path, collection, "success", None)

    def mark_failed(self, file_hash: str, file_path: str | Path, collection: str, error_msg: str) -> None:
        """记录失败处理。"""
        self._upsert(file_hash, file_path, collection, "failed", error_msg)

    def remove_record(self, file_hash: str, collection: str) -> bool:
        """删除某个 collection 下的去重记录。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM ingestion_history
                WHERE file_hash = ? AND collection = ?
                """,
                (file_hash, collection),
            )
            conn.commit()
            return cursor.rowcount > 0

    def _ensure_database(self) -> None:
        """初始化 SQLite 表结构。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_msg TEXT,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (file_hash, collection)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_history_status
                ON ingestion_history(status)
                """
            )
            conn.commit()

    def _upsert(
        self,
        file_hash: str,
        file_path: str | Path,
        collection: str,
        status: str,
        error_msg: str | None,
    ) -> None:
        """插入或更新一条记录。"""
        now = datetime.now(timezone.utc).isoformat()
        file_path_str = str(Path(file_path).resolve())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT processed_at
                FROM ingestion_history
                WHERE file_hash = ? AND collection = ?
                """,
                (file_hash, collection),
            )
            existing = cursor.fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE ingestion_history
                    SET file_path = ?,
                        status = ?,
                        error_msg = ?,
                        updated_at = ?
                    WHERE file_hash = ? AND collection = ?
                    """,
                    (file_path_str, status, error_msg, now, file_hash, collection),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO ingestion_history
                    (file_hash, collection, file_path, status, error_msg, processed_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (file_hash, collection, file_path_str, status, error_msg, now, now),
                )
            conn.commit()
