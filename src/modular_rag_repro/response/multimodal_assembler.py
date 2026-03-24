"""把检索结果里的图片信息组装成统一结构和 MCP 内容块。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import types

from modular_rag_repro.ingestion import ImageStorage
from modular_rag_repro.settings import Settings
from modular_rag_repro.types import RetrievalResult


class MultimodalAssembler:
    """最小可用的多模态组装器。"""

    MIME_MAP = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }

    def __init__(self, settings: Settings, max_images_per_result: int = 3) -> None:
        self.settings = settings
        self.max_images_per_result = max_images_per_result
        self.image_storage = ImageStorage(
            db_path=settings.ingestion.image_index_db_path,
            images_root=settings.ingestion.images_root_dir,
        )

    def extract_images(self, result: RetrievalResult) -> List[Dict[str, Any]]:
        """从检索结果 metadata 里抽取图片信息。"""
        images = result.metadata.get("images", [])
        captions = result.metadata.get("image_captions", [])
        caption_map = {
            str(item.get("id")): str(item.get("caption"))
            for item in captions
            if isinstance(item, dict) and item.get("id") and item.get("caption")
        }

        normalized: List[Dict[str, Any]] = []
        if not isinstance(images, list):
            return normalized

        for item in images[: self.max_images_per_result]:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            image_id = str(item["id"])
            file_path = str(item.get("file_path") or self.image_storage.get_image_path(image_id) or "")
            normalized.append(
                {
                    "id": image_id,
                    "page": item.get("page"),
                    "index": item.get("index"),
                    "file_path": file_path,
                    "caption": caption_map.get(image_id),
                }
            )
        return normalized

    def build_mcp_blocks(self, results: List[RetrievalResult]) -> List[types.ContentBlock]:
        """把检索结果中的图片转成 MCP 内容块。"""
        blocks: List[types.ContentBlock] = []
        seen: set[str] = set()

        for result in results:
            for image in self.extract_images(result):
                image_id = str(image["id"])
                if image_id in seen:
                    continue
                seen.add(image_id)

                file_path = str(image.get("file_path") or "")
                if not file_path:
                    continue
                path = Path(file_path)
                if not path.exists():
                    continue

                mime_type = self.MIME_MAP.get(path.suffix.lower(), "image/png")
                encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
                blocks.append(types.ImageContent(type="image", data=encoded, mimeType=mime_type))

                caption = image.get("caption")
                if caption:
                    blocks.append(types.TextContent(type="text", text=f"[Image {image_id}] {caption}"))

        return blocks
