"""`Reranker` 的最小单元测试。"""

from __future__ import annotations

import pytest

from modular_rag_repro.query_engine import QueryProcessor, SimpleReranker
from modular_rag_repro.types import RetrievalResult


def make_result(chunk_id: str, score: float, text: str) -> RetrievalResult:
    """构造检索结果。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": f"{chunk_id}.pdf", "title": chunk_id},
    )


def test_simple_reranker_promotes_more_keyword_relevant_result() -> None:
    """关键词覆盖率更高的结果应被提到前面。"""
    processor = QueryProcessor()
    reranker = SimpleReranker(provider="simple", top_k=5)
    processed = processor.process("sample pdf")

    results = [
        make_result("chunk_a", 0.9, "generic content with weak relevance"),
        make_result("chunk_b", 0.5, "sample pdf document with strong relevance"),
    ]

    reranked = reranker.rerank(processed, results, top_k=5)

    assert [item.chunk_id for item in reranked] == ["chunk_b", "chunk_a"]
    assert reranked[0].metadata["reranked"] is True
    assert reranked[0].metadata["matched_terms"] == ["sample", "pdf"]


def test_none_provider_keeps_original_order() -> None:
    """provider=none 时应保持原始顺序。"""
    processor = QueryProcessor()
    reranker = SimpleReranker(provider="none", top_k=5)
    processed = processor.process("sample pdf")

    results = [
        make_result("chunk_a", 0.9, "a"),
        make_result("chunk_b", 0.5, "b"),
    ]

    reranked = reranker.rerank(processed, results, top_k=5)

    assert [item.chunk_id for item in reranked] == ["chunk_a", "chunk_b"]


def test_invalid_provider_raises_error() -> None:
    """未知 provider 应报错。"""
    processor = QueryProcessor()
    reranker = SimpleReranker(provider="mystery", top_k=5)
    processed = processor.process("sample pdf")
    results = [make_result("chunk_a", 0.9, "sample pdf")]

    with pytest.raises(ValueError, match="不支持的 rerank provider"):
        reranker.rerank(processed, results, top_k=5)


def test_llm_provider_falls_back_to_simple(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm provider 失败时应退回 simple 逻辑。"""
    processor = QueryProcessor()
    reranker = SimpleReranker(provider="llm", top_k=5)
    processed = processor.process("sample pdf")

    results = [
        make_result("chunk_a", 0.9, "generic content with weak relevance"),
        make_result("chunk_b", 0.5, "sample pdf document with strong relevance"),
    ]

    monkeypatch.setattr(reranker, "_rerank_with_llm", lambda processed_query, results, top_k: (_ for _ in ()).throw(RuntimeError("llm down")))

    reranked = reranker.rerank(processed, results, top_k=5)

    assert [item.chunk_id for item in reranked] == ["chunk_b", "chunk_a"]
    assert reranked[0].metadata["rerank_provider"] == "simple"
    assert reranked[0].metadata["rerank_fallback"] == "llm"


def test_cross_encoder_provider_falls_back_to_simple(monkeypatch: pytest.MonkeyPatch) -> None:
    """cross_encoder provider 失败时也应退回 simple。"""
    processor = QueryProcessor()
    reranker = SimpleReranker(provider="cross_encoder", top_k=5)
    processed = processor.process("sample pdf")

    results = [
        make_result("chunk_a", 0.9, "generic content with weak relevance"),
        make_result("chunk_b", 0.5, "sample pdf document with strong relevance"),
    ]

    monkeypatch.setattr(
        reranker,
        "_rerank_with_cross_encoder",
        lambda processed_query, results, top_k: (_ for _ in ()).throw(RuntimeError("ce down")),
    )

    reranked = reranker.rerank(processed, results, top_k=5)

    assert [item.chunk_id for item in reranked] == ["chunk_b", "chunk_a"]
    assert reranked[0].metadata["rerank_provider"] == "simple"
    assert reranked[0].metadata["rerank_fallback"] == "cross_encoder"
