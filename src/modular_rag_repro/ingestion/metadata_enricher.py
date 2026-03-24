"""MetadataEnricher：给 chunk 补一层稳定的规则元数据。"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from modular_rag_repro.settings import Settings
from modular_rag_repro.types import Chunk


class MetadataEnricher:
    """最小可用的规则版元数据增强器。

    当前只做规则提取，不接 LLM。
    先补这几个最稳定的字段：

    - `title`
    - `summary`
    - `tags`
    - `has_images`
    - `image_count`
    - `chunk_char_length`
    """

    _WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
    _CN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
    _STOPWORDS = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "your",
        "about",
        "there",
        "their",
        "sample",
        "document",
        "testing",
        "loader",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.use_llm = settings.ingestion.metadata_enricher.use_llm

    def enrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """批量增强 chunk metadata。"""
        return [self.enrich_chunk(chunk) for chunk in chunks]

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        """增强单个 chunk 的 metadata。"""
        metadata = dict(chunk.metadata)
        images = metadata.get("images", [])

        metadata["title"] = self._build_title(chunk)
        metadata["summary"] = self._build_summary(chunk.text)
        metadata["tags"] = self._extract_tags(chunk.text)
        metadata["has_images"] = bool(images)
        metadata["image_count"] = len(images) if isinstance(images, list) else 0
        metadata["chunk_char_length"] = len(chunk.text)
        metadata["enriched_by"] = "rule"
        metadata["metadata_enricher_mode"] = "rule_only"

        return Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            metadata=metadata,
        )

    def _build_title(self, chunk: Chunk) -> str:
        """生成稳定标题。

        优先保留已有 `title`，否则取 chunk 第一条非空行。
        """
        existing_title = str(chunk.metadata.get("title", "")).strip()
        if existing_title:
            return existing_title[:120]

        for line in chunk.text.splitlines():
            candidate = line.strip()
            if candidate:
                return candidate[:120]
        return "Untitled Chunk"

    def _build_summary(self, text: str) -> str:
        """生成简短摘要。"""
        normalized = " ".join(part.strip() for part in text.splitlines() if part.strip())
        if not normalized:
            return ""
        return normalized[:180]

    def _extract_tags(self, text: str, top_k: int = 5) -> List[str]:
        """抽取一组低成本关键词。"""
        tokens: List[str] = []
        for word in self._WORD_RE.findall(text.lower()):
            if word in self._STOPWORDS:
                continue
            tokens.append(word)
        tokens.extend(self._CN_RE.findall(text))

        if not tokens:
            return []

        counter = Counter(tokens)
        return [item for item, _ in counter.most_common(top_k)]
