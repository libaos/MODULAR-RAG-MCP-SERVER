"""ImageStorage：把 PDF 里的图片落到本地并建立 SQLite 索引。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

from modular_rag_repro.settings import resolve_path


class ImageStorage:
    """最小可用的图片存储器。

    当前只做两件事：

    - 把 PDF 中的图片真正写到磁盘
    - 用 SQLite 记录图片路径和所属 collection/doc_hash
    """

    def __init__(
        self,
        db_path: str = "./data/db/image_index/images.sqlite3",
        images_root: str = "./data/images",
    ) -> None:
        self.db_path = resolve_path(db_path)
        self.images_root = resolve_path(images_root)
        self._ensure_schema()

    def store_pdf_images(
        self,
        file_path: str | Path,
        images: List[Dict[str, Any]],
        collection: str,
        doc_hash: str,
    ) -> List[Dict[str, Any]]:
        """从 PDF 中抽取并保存图片。

        入参 `images` 来自 `PdfLoader` 的元数据，占位信息里至少包含：

        - `id`
        - `xref`
        - `page`
        - `index`
        """
        if not images:
            return []

        path = Path(file_path).resolve()
        stored_images: List[Dict[str, Any]] = []

        pdf = fitz.open(path)
        try:
            for item in images:
                xref = int(item["xref"])
                extracted = pdf.extract_image(xref)
                image_bytes = extracted.get("image")
                extension = str(extracted.get("ext") or "png").lower()
                if not image_bytes:
                    continue

                image_path = self._build_image_path(collection, doc_hash, str(item["id"]), extension)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_bytes)

                record = {
                    **item,
                    "collection": collection,
                    "doc_hash": doc_hash,
                    "file_path": str(image_path.resolve()),
                    "extension": extension,
                    "size_bytes": len(image_bytes),
                }
                self._upsert_record(record)
                stored_images.append(record)
        finally:
            pdf.close()

        return stored_images

    def list_images(self, collection: Optional[str] = None, doc_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出已索引图片。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            sql = "SELECT * FROM image_index WHERE 1=1"
            params: list[Any] = []
            if collection:
                sql += " AND collection = ?"
                params.append(collection)
            if doc_hash:
                sql += " AND doc_hash = ?"
                params.append(doc_hash)
            sql += " ORDER BY collection, doc_hash, page_num, image_index"
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def count_images(self, collection: Optional[str] = None) -> int:
        """统计某个 collection 的图片数。"""
        conn = sqlite3.connect(self.db_path)
        try:
            if collection:
                row = conn.execute("SELECT COUNT(*) FROM image_index WHERE collection = ?", (collection,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM image_index").fetchone()
            return int(row[0] if row else 0)
        finally:
            conn.close()

    def delete_document_images(self, doc_hash: str, collection: Optional[str] = None) -> int:
        """删除某个文档关联的图片文件和索引记录。"""
        records = self.list_images(collection=collection, doc_hash=doc_hash)
        deleted = 0
        for record in records:
            path = Path(str(record["file_path"]))
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            deleted += 1

        conn = sqlite3.connect(self.db_path)
        try:
            if collection:
                conn.execute("DELETE FROM image_index WHERE doc_hash = ? AND collection = ?", (doc_hash, collection))
            else:
                conn.execute("DELETE FROM image_index WHERE doc_hash = ?", (doc_hash,))
            conn.commit()
        finally:
            conn.close()

        # 顺手清空空目录，避免测试目录越积越多。
        self._cleanup_empty_dirs(self.images_root)
        return deleted

    def delete_collection(self, collection: str) -> int:
        """删除整个 collection 下的图片和索引。"""
        records = self.list_images(collection=collection)
        for record in records:
            path = Path(str(record["file_path"]))
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM image_index WHERE collection = ?", (collection,))
            conn.commit()
        finally:
            conn.close()

        collection_dir = self.images_root / collection
        if collection_dir.exists():
            for child in sorted(collection_dir.rglob("*"), reverse=True):
                try:
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                except Exception:
                    pass
            try:
                collection_dir.rmdir()
            except Exception:
                pass
        return len(records)

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_root.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    doc_hash TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    image_index INTEGER NOT NULL,
                    xref INTEGER NOT NULL,
                    extension TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_collection ON image_index(collection)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_doc_hash ON image_index(doc_hash)")
            conn.commit()
        finally:
            conn.close()

    def _build_image_path(self, collection: str, doc_hash: str, image_id: str, extension: str) -> Path:
        return self.images_root / collection / doc_hash / f"{image_id}.{extension}"

    def _upsert_record(self, record: Dict[str, Any]) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO image_index
                (image_id, collection, doc_hash, page_num, image_index, xref, extension, file_path, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["collection"],
                    record["doc_hash"],
                    int(record["page"]),
                    int(record["index"]),
                    int(record["xref"]),
                    record["extension"],
                    record["file_path"],
                    int(record["size_bytes"]),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _cleanup_empty_dirs(self, root: Path) -> None:
        if not root.exists():
            return
        for child in sorted(root.rglob("*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
