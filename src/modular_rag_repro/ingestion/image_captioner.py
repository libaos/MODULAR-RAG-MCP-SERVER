"""ImageCaptioner：把图片占位符替换成可检索的文字描述。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from modular_rag_repro.settings import Settings
from modular_rag_repro.types import Chunk


class ImageCaptioner:
    """最小可用的规则版图片描述器。"""

    PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vision_enabled = settings.vision_llm.enabled

    def caption_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """批量给 chunk 图片占位符加描述。"""
        return [self.caption_chunk(chunk) for chunk in chunks]

    def caption_chunk(self, chunk: Chunk) -> Chunk:
        """处理单个 chunk。"""
        image_lookup = self._build_image_lookup(chunk.metadata.get("images", []))
        matches = self.PLACEHOLDER_RE.findall(chunk.text)
        if not matches:
            return chunk

        text = chunk.text
        captions: List[Dict[str, str]] = []

        for raw_image_id in matches:
            image_id = raw_image_id.strip()
            image_meta = image_lookup.get(image_id)
            if image_meta is None:
                continue

            caption = self._build_rule_caption(image_meta)
            placeholder = f"[IMAGE: {image_id}]"
            replacement = f"{placeholder}\n(Image description: {caption})"
            text = text.replace(placeholder, replacement)
            captions.append(
                {
                    "id": image_id,
                    "caption": caption,
                    "file_path": str(image_meta.get("file_path") or ""),
                }
            )

        if not captions:
            return chunk

        metadata = dict(chunk.metadata)
        metadata["image_captions"] = captions
        metadata["captioned_by"] = "rule"
        metadata["image_captioner_mode"] = "rule_only"
        metadata["captioned_image_count"] = len(captions)

        return Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            text=text,
            metadata=metadata,
        )

    def _build_image_lookup(self, images: object) -> Dict[str, Dict[str, object]]:
        lookup: Dict[str, Dict[str, object]] = {}
        if not isinstance(images, list):
            return lookup
        for item in images:
            if isinstance(item, dict) and item.get("id"):
                lookup[str(item["id"])] = item
        return lookup

    def _build_rule_caption(self, image_meta: Dict[str, object]) -> str:
        page = image_meta.get("page")
        index = image_meta.get("index")
        file_path = str(image_meta.get("file_path") or "")
        extension = Path(file_path).suffix.lstrip(".").lower() if file_path else str(image_meta.get("extension") or "image")
        size_bytes = image_meta.get("size_bytes")

        parts = [f"Document image on page {page}, figure {index}."]
        if extension:
            parts.append(f"Format: {extension}.")
        if size_bytes:
            parts.append(f"Size: {size_bytes} bytes.")
        return " ".join(parts)
