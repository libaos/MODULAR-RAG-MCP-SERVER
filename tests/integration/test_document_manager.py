"""文档管理集成测试。"""

from __future__ import annotations

from modular_rag_repro.ingestion import BM25Indexer, ChromaUpserter
from modular_rag_repro.management import DocumentManager
from helpers import SAMPLE_PDF, cleanup_collection, seed_collection


def test_document_manager_delete_document_cleans_bm25_and_chroma(test_settings, unique_collection: str) -> None:
    """删除文档后，BM25 与 Chroma 都应同步清理。"""
    cleanup_collection(test_settings, unique_collection)
    seeded = seed_collection(test_settings, unique_collection, SAMPLE_PDF)
    manager = DocumentManager(test_settings)

    try:
        summary = manager.get_document_summary(seeded.document.id, collection=unique_collection)
        assert summary.doc_id == seeded.document.id
        assert summary.chunk_count >= 1

        delete_result = manager.delete_document(seeded.document.id, collection=unique_collection)

        assert delete_result.doc_id == seeded.document.id
        assert delete_result.chroma_deleted_count >= 1
        assert delete_result.bm25_remaining_count == 0
        assert delete_result.rebuilt is True

        bm25 = BM25Indexer()
        assert bm25.load(unique_collection) is False

        chroma = ChromaUpserter(test_settings, collection=unique_collection)
        assert chroma.get_collection_count(unique_collection) == 0
    finally:
        cleanup_collection(test_settings, unique_collection)
