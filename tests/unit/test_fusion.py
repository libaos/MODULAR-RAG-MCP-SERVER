"""`RRFFusion` 的最小单元测试。"""

from __future__ import annotations

import pytest

from modular_rag_repro.query_engine import RRFFusion
from modular_rag_repro.types import RetrievalResult


def make_result(chunk_id: str, score: float, source_path: str) -> RetrievalResult:
    """构造检索结果。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text for {chunk_id}",
        metadata={"source_path": source_path},
    )


def test_fuse_merges_duplicate_chunk_ids_and_ranks_them_first() -> None:
    """同一个 chunk 同时出现在两路结果里时，融合后应排得更靠前。"""
    fusion = RRFFusion(k=60)
    dense = [
        make_result("chunk_a", 0.9, "a.pdf"),
        make_result("chunk_b", 0.8, "b.pdf"),
    ]
    sparse = [
        make_result("chunk_b", 2.0, "b.pdf"),
        make_result("chunk_c", 1.5, "c.pdf"),
    ]

    fused = fusion.fuse([dense, sparse], top_k=10)

    assert [item.chunk_id for item in fused] == ["chunk_b", "chunk_a", "chunk_c"]


def test_fuse_respects_top_k() -> None:
    """融合结果应支持按 top_k 截断。"""
    fusion = RRFFusion(k=60)
    dense = [
        make_result("chunk_a", 0.9, "a.pdf"),
        make_result("chunk_b", 0.8, "b.pdf"),
        make_result("chunk_c", 0.7, "c.pdf"),
    ]

    fused = fusion.fuse([dense], top_k=2)

    assert len(fused) == 2
    assert [item.chunk_id for item in fused] == ["chunk_a", "chunk_b"]


def test_fuse_empty_lists_returns_empty_results() -> None:
    """没有输入结果时应直接返回空列表。"""
    fusion = RRFFusion(k=60)
    assert fusion.fuse([], top_k=5) == []
    assert fusion.fuse([[]], top_k=5) == []


def test_invalid_k_raises_error() -> None:
    """`k` 必须是正整数。"""
    with pytest.raises(ValueError, match="k 必须是正整数"):
        RRFFusion(k=0)
