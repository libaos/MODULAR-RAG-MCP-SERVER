"""摄取去重集成测试。"""

from __future__ import annotations

from modular_rag_repro.ingestion import IngestionPipeline
from modular_rag_repro.management import DocumentManager
from helpers import SAMPLE_PDF, cleanup_collection


def test_ingestion_pipeline_supports_skip_force_and_reingest_after_delete(test_settings, unique_collection: str) -> None:
    """同文件再次摄取应 skip，force 可重跑，删除后可再次正常摄取。"""
    cleanup_collection(test_settings, unique_collection)
    pipeline = IngestionPipeline(test_settings, collection=unique_collection)

    try:
        first = pipeline.run(str(SAMPLE_PDF))
        assert first.success is True
        assert first.skipped is False
        assert first.file_hash is not None

        second = pipeline.run(str(SAMPLE_PDF))
        assert second.success is True
        assert second.skipped is True
        assert second.skip_reason == "already_processed"
        assert second.file_hash == first.file_hash

        forced = pipeline.run(str(SAMPLE_PDF), force=True)
        assert forced.success is True
        assert forced.skipped is False
        assert forced.file_hash == first.file_hash

        manager = DocumentManager(test_settings)
        delete_result = manager.delete_document(first.document.id, collection=unique_collection)
        assert delete_result.rebuilt is True

        after_delete = pipeline.run(str(SAMPLE_PDF))
        assert after_delete.success is True
        assert after_delete.skipped is False
    finally:
        cleanup_collection(test_settings, unique_collection)
