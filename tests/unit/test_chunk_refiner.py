"""`ChunkRefiner` 的最小单元测试。"""

from __future__ import annotations

from modular_rag_repro.ingestion import ChunkRefiner
from modular_rag_repro.types import Chunk


def make_chunk(text: str) -> Chunk:
    """构造测试 chunk。"""
    return Chunk(
        id="chunk-001",
        document_id="doc-001",
        text=text,
        metadata={"chunk_index": 0},
    )


def test_chunk_refiner_cleans_whitespace_and_comments(test_settings) -> None:
    """规则版清洗应去掉无效空白和 HTML 注释。"""
    refiner = ChunkRefiner(test_settings)
    chunk = make_chunk("Hello\xa0\xa0world <!-- hidden -->\n\n\n<p>next</p>")

    refined = refiner.refine_chunk(chunk)

    assert refined.text == "Hello world\n\nnext"
    assert refined.metadata["refined_by"] == "rule"
    assert refined.metadata["refinement_changed"] is True


def test_chunk_refiner_preserves_meaningful_text_when_no_change_needed(test_settings) -> None:
    """干净文本不应被过度修改。"""
    refiner = ChunkRefiner(test_settings)
    chunk = make_chunk("sample pdf document")

    refined = refiner.refine_chunk(chunk)

    assert refined.text == "sample pdf document"
    assert refined.metadata["refinement_changed"] is False
    assert refined.metadata["original_char_length"] == len(chunk.text)
    assert refined.metadata["refined_char_length"] == len(chunk.text)
