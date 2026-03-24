"""`QueryProcessor` 的最小单元测试。"""

from __future__ import annotations

import pytest

from modular_rag_repro.query_engine import QueryProcessor


def test_process_extracts_keywords_and_filters() -> None:
    """应同时提取关键词并解析 `key:value` 过滤项。"""
    processor = QueryProcessor()

    processed = processor.process("What is sample pdf doc_type:pdf page_count:3 enabled:true score:1.5")

    assert processed.normalized_text == "What is sample pdf"
    assert processed.keywords == ["sample", "pdf"]
    assert processed.filters == {
        "doc_type": "pdf",
        "page_count": 3,
        "enabled": True,
        "score": 1.5,
    }


def test_process_chinese_query_removes_stopwords() -> None:
    """中文停用词不应留在关键词里。"""
    processor = QueryProcessor()

    processed = processor.process("这个文档讲了什么")

    assert "这个" not in processed.keywords
    assert "文档" not in processed.keywords
    assert "什么" not in processed.keywords
    assert processed.keywords == []


def test_process_deduplicates_keywords() -> None:
    """重复关键词只保留一次。"""
    processor = QueryProcessor()

    processed = processor.process("sample sample pdf pdf")

    assert processed.keywords == ["sample", "pdf"]


def test_empty_query_raises_error() -> None:
    """空查询应报错。"""
    processor = QueryProcessor()

    with pytest.raises(ValueError, match="query 不能为空"):
        processor.process("   ")
