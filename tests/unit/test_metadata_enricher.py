"""`MetadataEnricher` 的最小单元测试。"""

from __future__ import annotations

from modular_rag_repro.ingestion import MetadataEnricher
from modular_rag_repro.types import Chunk


def make_chunk(text: str, metadata: dict | None = None) -> Chunk:
    """构造测试 chunk。"""
    return Chunk(
        id="chunk-meta-001",
        document_id="doc-meta-001",
        text=text,
        metadata=metadata or {"chunk_index": 0},
    )


def test_metadata_enricher_adds_summary_and_tags(test_settings) -> None:
    """规则增强应补上摘要和关键词。"""
    enricher = MetadataEnricher(test_settings)
    chunk = make_chunk("Sample PDF document about vector retrieval and hybrid search.")

    enriched = enricher.enrich_chunk(chunk)

    assert enriched.metadata["enriched_by"] == "rule"
    assert enriched.metadata["summary"].startswith("Sample PDF document")
    assert "vector" in enriched.metadata["tags"]
    assert enriched.metadata["chunk_char_length"] == len(chunk.text)


def test_metadata_enricher_preserves_existing_title_and_sets_image_flags(test_settings) -> None:
    """已有标题应保留，同时补图片相关标记。"""
    enricher = MetadataEnricher(test_settings)
    chunk = make_chunk(
        "content body",
        metadata={
            "chunk_index": 0,
            "title": "Existing Title",
            "images": [{"id": "img1"}, {"id": "img2"}],
        },
    )

    enriched = enricher.enrich_chunk(chunk)

    assert enriched.metadata["title"] == "Existing Title"
    assert enriched.metadata["has_images"] is True
    assert enriched.metadata["image_count"] == 2
