"""`ImageStorage` 的最小单元测试。"""

from __future__ import annotations

from modular_rag_repro.ingestion import ImageStorage, PdfLoader
from helpers import WITH_IMAGES_PDF


def test_image_storage_stores_pdf_images_and_indexes_paths(test_settings) -> None:
    """应能从带图 PDF 落盘图片并建立索引。"""
    storage = ImageStorage(
        db_path=test_settings.ingestion.image_index_db_path,
        images_root=test_settings.ingestion.images_root_dir,
    )
    document = PdfLoader(extract_images=True).load(WITH_IMAGES_PDF)
    fake_images = document.metadata["images"]

    stored = storage.store_pdf_images(
        file_path=str(WITH_IMAGES_PDF),
        images=fake_images,
        collection="unit-image-test",
        doc_hash="abc123",
    )

    assert len(stored) == 1
    assert stored[0]["collection"] == "unit-image-test"
    assert stored[0]["doc_hash"] == "abc123"
    assert stored[0]["size_bytes"] > 0
    assert storage.count_images("unit-image-test") == 1


def test_image_storage_delete_document_images_removes_records(test_settings) -> None:
    """按文档删除应同时清掉索引和文件。"""
    storage = ImageStorage(
        db_path=test_settings.ingestion.image_index_db_path,
        images_root=test_settings.ingestion.images_root_dir,
    )
    document = PdfLoader(extract_images=True).load(WITH_IMAGES_PDF)
    fake_images = document.metadata["images"]
    stored = storage.store_pdf_images(
        file_path=str(WITH_IMAGES_PDF),
        images=fake_images,
        collection="unit-image-delete",
        doc_hash="dochash-delete",
    )

    deleted = storage.delete_document_images("dochash-delete", "unit-image-delete")

    assert deleted == 1
    assert storage.count_images("unit-image-delete") == 0
    assert stored[0]["file_path"]
