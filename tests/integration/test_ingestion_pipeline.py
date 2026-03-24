"""摄取链路集成测试。"""

from __future__ import annotations

import pytest

from modular_rag_repro.ingestion import BM25Indexer, ChromaUpserter, IngestionPipeline
from modular_rag_repro.types import Document
from helpers import SAMPLE_PDF, WITH_IMAGES_PDF, cleanup_collection


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


class DirtyTextLoader:
    """返回带脏文本的假 loader，用来验证 `ChunkRefiner` 是否接进 pipeline。"""

    def load(self, file_path: str) -> Document:
        return Document(
            id="doc_dirty_case",
            text="Hello\xa0\xa0world <!-- hidden -->\n\n\n<p>next</p>",
            source_path=file_path,
            metadata={"title": "dirty-case", "page_count": 1},
        )


def test_ingestion_pipeline_applies_chunk_refiner(test_settings, unique_collection: str) -> None:
    """摄取链路应在 embedding 前先执行规则版 chunk 清洗。"""
    cleanup_collection(test_settings, unique_collection)
    pipeline = IngestionPipeline(
        test_settings,
        collection=unique_collection,
        loader=DirtyTextLoader(),
    )

    try:
        result = pipeline.run(str(SAMPLE_PDF))

        assert result.success is True
        assert result.chunk_count == 1
        assert result.chunks[0].text == "Hello world\n\nnext"
        assert result.chunks[0].metadata["refined_by"] == "rule"
        assert result.chunks[0].metadata["refinement_changed"] is True
        assert result.stages["refine"]["changed_chunk_count"] == 1
    finally:
        cleanup_collection(test_settings, unique_collection)


def test_ingestion_pipeline_applies_metadata_enricher(test_settings, unique_collection: str) -> None:
    """摄取链路应在 embedding 前补齐规则元数据。"""
    cleanup_collection(test_settings, unique_collection)
    pipeline = IngestionPipeline(
        test_settings,
        collection=unique_collection,
        loader=DirtyTextLoader(),
    )

    try:
        result = pipeline.run(str(SAMPLE_PDF))

        assert result.success is True
        assert result.chunk_count == 1
        assert result.chunks[0].metadata["enriched_by"] == "rule"
        assert result.chunks[0].metadata["title"] == "dirty-case"
        assert result.chunks[0].metadata["summary"] == "Hello world next"
        assert result.chunks[0].metadata["has_images"] is False
        assert "hello" in result.chunks[0].metadata["tags"]
        assert result.stages["enrich"]["sample_title"] == "dirty-case"
    finally:
        cleanup_collection(test_settings, unique_collection)


def test_ingestion_pipeline_stores_images_for_pdf_with_images(test_settings, unique_collection: str) -> None:
    """带图 PDF 进入 pipeline 后应真正落盘图片并建立索引。"""
    cleanup_collection(test_settings, unique_collection)
    pipeline = IngestionPipeline(test_settings, collection=unique_collection)

    try:
        result = pipeline.run(str(WITH_IMAGES_PDF))

        assert result.success is True
        assert result.image_count >= 1
        assert result.stages["images"]["image_count"] >= 1
        assert result.stages["caption"]["captioned_images"] >= 1
        assert result.document is not None
        assert result.document.metadata["images"][0]["file_path"]
        assert "Image description:" in result.chunks[0].text
        assert result.chunks[0].metadata["captioned_image_count"] >= 1
    finally:
        cleanup_collection(test_settings, unique_collection)


def test_ingestion_pipeline_records_fallback_modes(
    test_settings,
    unique_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """增强器失败时，pipeline 应保留 fallback metadata。"""
    cleanup_collection(test_settings, unique_collection)
    test_settings.ingestion.chunk_refiner.use_llm = True
    test_settings.ingestion.chunk_refiner.provider = "llm"
    test_settings.ingestion.metadata_enricher.use_llm = True
    test_settings.ingestion.metadata_enricher.provider = "llm"
    test_settings.vision_llm.enabled = True
    test_settings.vision_llm.provider = "ollama"

    pipeline = IngestionPipeline(test_settings, collection=unique_collection)
    monkeypatch.setattr(pipeline.chunk_refiner, "_llm_refine", lambda text: (_ for _ in ()).throw(RuntimeError("llm down")))
    monkeypatch.setattr(pipeline.metadata_enricher, "_llm_enrich", lambda chunk: (_ for _ in ()).throw(RuntimeError("llm down")))
    monkeypatch.setattr(pipeline.image_captioner, "_build_vision_caption", lambda image_meta: (_ for _ in ()).throw(RuntimeError("vision down")))

    try:
        result = pipeline.run(str(WITH_IMAGES_PDF))

        assert result.success is True
        assert result.chunks[0].metadata["refiner_fallback"] == "llm"
        assert result.chunks[0].metadata["enricher_fallback"] == "llm"
        assert result.chunks[0].metadata["caption_fallback"] == "vision_llm"
    finally:
        cleanup_collection(test_settings, unique_collection)
