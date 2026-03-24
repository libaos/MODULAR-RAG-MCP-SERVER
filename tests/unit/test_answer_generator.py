"""`AnswerGenerator` 的最小单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modular_rag_repro.types import RetrievalResult


def make_settings(provider: str = "local", enabled: bool = True):
    """构造最小可用的 AnswerGenerator 配置。"""
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="ollama",
            model="qwen2.5:3b",
            base_url="http://localhost:11434",
            temperature=0.0,
            max_tokens=2048,
            enabled=True,
            timeout_seconds=30.0,
        ),
        answer_generation=SimpleNamespace(
            enabled=enabled,
            provider=provider,
            model="qwen2.5:3b",
            max_context_chunks=2,
            timeout_seconds=30.0,
        ),
    )


def make_results() -> list[RetrievalResult]:
    """构造两条稳定检索结果。"""
    return [
        RetrievalResult(
            chunk_id="chunk_a",
            score=0.9,
            text="Sample PDF document for testing hybrid search.",
            metadata={"source_path": "sample.pdf", "chunk_index": 0},
        ),
        RetrievalResult(
            chunk_id="chunk_b",
            score=0.8,
            text="The document explains the PDF loader and retrieval pipeline.",
            metadata={"source_path": "sample.pdf", "chunk_index": 1},
        ),
    ]


def test_local_answer_generator_returns_generated_answer() -> None:
    """provider=local 时应返回本地生成答案。"""
    from modular_rag_repro.query_engine.answer_generator import AnswerGenerator

    generator = AnswerGenerator(make_settings(provider="local"))
    answer, mode, metadata = generator.generate("sample pdf", make_results())

    assert answer is not None
    assert "sample.pdf" in answer
    assert mode == "generated"
    assert metadata["provider"] == "local"


def test_ollama_answer_generator_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """ollama 失败时应退回 local answer。"""
    from modular_rag_repro.query_engine.answer_generator import AnswerGenerator

    generator = AnswerGenerator(make_settings(provider="ollama"))
    monkeypatch.setattr(generator, "_generate_with_ollama", lambda query, results: (_ for _ in ()).throw(RuntimeError("boom")))

    answer, mode, metadata = generator.generate("sample pdf", make_results())

    assert answer is not None
    assert mode == "fallback_local"
    assert metadata["provider"] == "ollama"
    assert metadata["fallback_provider"] == "local"
    assert "boom" in metadata["error"]
