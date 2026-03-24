"""`DocumentChunker` 的最小单元测试。"""

from __future__ import annotations

import pytest

from modular_rag_repro.settings import Settings
from modular_rag_repro.types import Document
from modular_rag_repro.ingestion import DocumentChunker


def build_settings(chunk_size: int = 10, chunk_overlap: int = 2) -> Settings:
    """构造只关心 chunk 参数的测试配置。"""
    settings = Settings()
    settings.ingestion.chunk_size = chunk_size
    settings.ingestion.chunk_overlap = chunk_overlap
    return settings


def test_split_document_short_text_keeps_single_chunk_and_metadata() -> None:
    """短文本不应该被过度切分，并且要继承文档元数据。"""
    chunker = DocumentChunker(build_settings(chunk_size=50, chunk_overlap=10))
    document = Document(
        id="doc_demo",
        text="这是一段很短的文本，不需要被切成多个 chunk。",
        source_path="demo.pdf",
        metadata={"title": "短文档", "doc_type": "pdf"},
    )

    chunks = chunker.split_document(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc_demo"
    assert chunks[0].metadata["title"] == "短文档"
    assert chunks[0].metadata["doc_type"] == "pdf"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["source_ref"] == "doc_demo"
    assert chunks[0].metadata["char_length"] == len(chunks[0].text)


def test_window_split_respects_chunk_size_and_overlap() -> None:
    """固定窗口切分时，`chunk_size` 和 `chunk_overlap` 都必须生效。"""
    chunker = DocumentChunker(build_settings(chunk_size=4, chunk_overlap=1))
    document = Document(
        id="doc_alpha",
        text="ABCDEFGHIJ",
        source_path="alpha.pdf",
    )

    chunks = chunker.split_document(document)

    assert [chunk.text for chunk in chunks] == ["ABCD", "DEFG", "GHIJ"]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]


def test_chunk_ids_are_stable_for_same_input() -> None:
    """同一文档重复切分时，chunk_id 应保持稳定。"""
    chunker = DocumentChunker(build_settings(chunk_size=4, chunk_overlap=1))
    document = Document(
        id="doc_stable",
        text="ABCDEFGHIJ",
        source_path="stable.pdf",
    )

    first = chunker.split_document(document)
    second = chunker.split_document(document)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_invalid_chunk_overlap_raises_error() -> None:
    """`chunk_overlap >= chunk_size` 应直接报错。"""
    with pytest.raises(ValueError, match="chunk_overlap 必须小于 chunk_size"):
        DocumentChunker(build_settings(chunk_size=8, chunk_overlap=8))
