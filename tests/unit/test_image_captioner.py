"""`ImageCaptioner` 的最小单元测试。"""

from __future__ import annotations

import pytest

from modular_rag_repro.ingestion import ImageCaptioner
from modular_rag_repro.types import Chunk


def test_image_captioner_replaces_placeholder_and_sets_metadata(test_settings) -> None:
    """占位符应被替换成规则 caption，并写入 metadata。"""
    captioner = ImageCaptioner(test_settings)
    chunk = Chunk(
        id="chunk-img-001",
        document_id="doc-img-001",
        text="Some text\n\n[IMAGE: img_1]",
        metadata={
            "images": [
                {
                    "id": "img_1",
                    "page": 1,
                    "index": 1,
                    "file_path": "C:/tmp/fake.png",
                    "size_bytes": 1234,
                    "extension": "png",
                }
            ]
        },
    )

    captioned = captioner.caption_chunk(chunk)

    assert "Image description:" in captioned.text
    assert captioned.metadata["captioned_by"] == "rule"
    assert captioned.metadata["captioned_image_count"] == 1
    assert captioned.metadata["image_captions"][0]["id"] == "img_1"


def test_image_captioner_vision_mode_uses_vision_caption(test_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """打开 vision 模式后，应优先使用视觉 caption。"""
    test_settings.vision_llm.enabled = True
    test_settings.vision_llm.provider = "ollama"
    captioner = ImageCaptioner(test_settings)
    chunk = Chunk(
        id="chunk-img-002",
        document_id="doc-img-002",
        text="[IMAGE: img_1]",
        metadata={
            "images": [
                {
                    "id": "img_1",
                    "page": 1,
                    "index": 1,
                    "file_path": "C:/tmp/fake.png",
                    "size_bytes": 1234,
                    "extension": "png",
                }
            ]
        },
    )

    monkeypatch.setattr(captioner, "_build_vision_caption", lambda image_meta: "vision caption")

    captioned = captioner.caption_chunk(chunk)

    assert "vision caption" in captioned.text
    assert captioned.metadata["captioned_by"] == "vision_llm"
    assert captioned.metadata["image_captioner_mode"] == "vision_llm"


def test_image_captioner_vision_failure_falls_back_to_rule(test_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """视觉增强失败时，应退回规则 caption。"""
    test_settings.vision_llm.enabled = True
    test_settings.vision_llm.provider = "ollama"
    captioner = ImageCaptioner(test_settings)
    chunk = Chunk(
        id="chunk-img-003",
        document_id="doc-img-003",
        text="[IMAGE: img_1]",
        metadata={
            "images": [
                {
                    "id": "img_1",
                    "page": 1,
                    "index": 1,
                    "file_path": "C:/tmp/fake.png",
                    "size_bytes": 1234,
                    "extension": "png",
                }
            ]
        },
    )

    monkeypatch.setattr(captioner, "_build_vision_caption", lambda image_meta: (_ for _ in ()).throw(RuntimeError("vision down")))

    captioned = captioner.caption_chunk(chunk)

    assert "Image description:" in captioned.text
    assert captioned.metadata["captioned_by"] == "rule"
    assert captioned.metadata["image_captioner_mode"] == "fallback_rule"
    assert captioned.metadata["caption_fallback"] == "vision_llm"
