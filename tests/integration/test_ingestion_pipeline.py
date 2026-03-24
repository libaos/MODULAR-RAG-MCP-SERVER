"""摄取链路集成测试。"""

from __future__ import annotations

from pathlib import Path

from modular_rag_repro.ingestion import BM25Indexer, ChromaUpserter, IngestionPipeline
from helpers import SAMPLE_PDF, cleanup_collection


def test_ingestion_pipeline_persists_bm25_and_chroma(test_settings, unique_collection: str) -> None:
    """摄取链路应同时落下 BM25 和 Chroma 数据。"""
    cleanup_collection(test_settings, unique_collection)
    pipeline = IngestionPipeline(test_settings, collection=unique_collection)

    try:
        result = pipeline.run(str(SAMPLE_PDF))

        assert result.success is True
        assert result.document is not None
        assert result.chunk_count >= 1
        assert result.vector_count == result.chunk_count
        assert result.upserted_count == result.chunk_count

        bm25 = BM25Indexer()
        assert bm25.load(unique_collection) is True
        assert len(bm25.query("sample pdf", top_k=3)) >= 1

        chroma = ChromaUpserter(test_settings, collection=unique_collection)
        assert chroma.get_collection_count(unique_collection) == result.chunk_count
    finally:
        cleanup_collection(test_settings, unique_collection)
