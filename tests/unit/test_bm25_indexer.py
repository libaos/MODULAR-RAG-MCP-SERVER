"""`BM25Indexer` 的最小单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from modular_rag_repro.ingestion import BM25Indexer
from modular_rag_repro.types import Chunk


def build_chunks() -> list[Chunk]:
    """构造一组稳定的测试 chunk。"""
    return [
        Chunk(
            id="doc_alpha_0000_aaaa1111",
            document_id="doc_alpha",
            text="sample pdf document for testing retrieval",
            metadata={"source_ref": "doc_alpha", "chunk_index": 0, "source_path": "alpha.pdf"},
        ),
        Chunk(
            id="doc_beta_0000_bbbb2222",
            document_id="doc_beta",
            text="another document about bm25 ranking",
            metadata={"source_ref": "doc_beta", "chunk_index": 0, "source_path": "beta.pdf"},
        ),
    ]


def test_build_and_query_returns_expected_chunk(tmp_path: Path) -> None:
    """构建索引后，关键词查询应该能命中对应 chunk。"""
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = build_chunks()

    indexer.build(chunks, collection="unit-bm25")
    results = indexer.query("sample pdf", top_k=3)

    assert len(results) >= 1
    assert results[0].chunk_id == "doc_alpha_0000_aaaa1111"
    assert results[0].metadata["source_ref"] == "doc_alpha"


def test_load_chunks_roundtrip_restores_chunk_metadata(tmp_path: Path) -> None:
    """落盘后的 BM25 文档区应该能恢复成 Chunk 列表。"""
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = build_chunks()

    indexer.build(chunks, collection="unit-roundtrip")
    restored = indexer.load_chunks("unit-roundtrip")

    assert len(restored) == 2
    assert restored[0].id == chunks[0].id
    assert restored[0].metadata["chunk_index"] == 0
    assert restored[1].metadata["source_ref"] == "doc_beta"


def test_rebuild_from_chunks_overwrites_previous_index(tmp_path: Path) -> None:
    """重建索引后，旧文档不应该继续留在查询结果里。"""
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = build_chunks()

    indexer.build(chunks, collection="unit-rebuild")
    remaining = [chunks[1]]
    count = indexer.rebuild_from_chunks(remaining, collection="unit-rebuild")

    assert count == 1
    assert indexer.load("unit-rebuild") is True
    results = indexer.query("sample pdf", top_k=3)
    assert results == []

    results = indexer.query("bm25 ranking", top_k=3)
    assert len(results) == 1
    assert results[0].chunk_id == "doc_beta_0000_bbbb2222"


def test_delete_index_removes_index_file(tmp_path: Path) -> None:
    """删除索引后，磁盘文件应该消失。"""
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = build_chunks()

    indexer.build(chunks, collection="unit-delete")
    index_path = indexer.get_index_path("unit-delete")
    assert index_path.exists() is True

    deleted = indexer.delete_index("unit-delete")

    assert deleted is True
    assert index_path.exists() is False
    assert indexer.load("unit-delete") is False


def test_query_without_loading_index_raises_error(tmp_path: Path) -> None:
    """未 build/load 前直接查询应报错。"""
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))

    with pytest.raises(ValueError, match="索引尚未加载"):
        indexer.query("sample pdf", top_k=3)
