"""`ImageCaptioner` 的最小单元测试。"""

from __future__ import annotations

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
