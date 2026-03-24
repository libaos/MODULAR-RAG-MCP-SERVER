"""`MultimodalAssembler` 的最小单元测试。"""

from __future__ import annotations

from pathlib import Path

from modular_rag_repro.response import MultimodalAssembler
from modular_rag_repro.types import RetrievalResult
from helpers import WORKSPACE_ROOT


def test_multimodal_assembler_extracts_images_and_builds_blocks(test_settings) -> None:
    """应能从结果 metadata 中抽出图片并生成 MCP image block。"""
    image_path = WORKSPACE_ROOT / "tests" / "fixtures" / "sample_documents" / "test_vision_llm.jpg"
    assembler = MultimodalAssembler(test_settings)
    result = RetrievalResult(
        chunk_id="chunk-mm-001",
        score=0.9,
        text="captioned chunk",
        metadata={
            "images": [{"id": "img_1", "file_path": str(image_path), "page": 1, "index": 1}],
            "image_captions": [{"id": "img_1", "caption": "Document image on page 1."}],
        },
    )

    images = assembler.extract_images(result)
    blocks = assembler.build_mcp_blocks([result])

    assert len(images) == 1
    assert images[0]["id"] == "img_1"
    assert any(getattr(block, "type", None) == "image" for block in blocks)
