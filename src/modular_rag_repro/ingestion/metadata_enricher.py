"""MetadataEnricher：给 chunk 补一层稳定的规则元数据。"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from modular_rag_repro.llm import OllamaClient
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
        self.provider = settings.ingestion.metadata_enricher.provider.lower()
        self.model = settings.ingestion.metadata_enricher.model or settings.llm.model
        self.timeout = settings.ingestion.metadata_enricher.timeout_seconds
        self.client = OllamaClient(
            base_url=settings.llm.base_url,
            model=self.model,
            temperature=0.0,
            max_tokens=min(settings.llm.max_tokens, 512),
            timeout=self.timeout,
        )

    def enrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """批量增强 chunk metadata。"""
        return [self.enrich_chunk(chunk) for chunk in chunks]

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        """增强单个 chunk 的 metadata。"""
        metadata = dict(chunk.metadata)
        images = metadata.get("images", [])

        enriched_by = "rule"
        enricher_mode = "rule_only"
        enricher_fallback = None
        payload = {
            "title": self._build_title(chunk),
            "summary": self._build_summary(chunk.text),
            "tags": self._extract_tags(chunk.text),
        }
        if self.use_llm and self.provider == "llm":
            try:
                llm_payload = self._llm_enrich(chunk)
                payload.update({key: value for key, value in llm_payload.items() if value})
                enriched_by = "llm"
                enricher_mode = "llm"
            except Exception:
                enricher_mode = "fallback_rule"
                enricher_fallback = "llm"

        metadata["title"] = payload["title"]
        metadata["summary"] = payload["summary"]
        metadata["tags"] = payload["tags"]
        metadata["has_images"] = bool(images)
        metadata["image_count"] = len(images) if isinstance(images, list) else 0
        metadata["chunk_char_length"] = len(chunk.text)
        metadata["enriched_by"] = enriched_by
        metadata["metadata_enricher_mode"] = enricher_mode
        if enricher_fallback:
            metadata["enricher_fallback"] = enricher_fallback

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

    def _llm_enrich(self, chunk: Chunk) -> dict:
        """使用 LLM 生成标题、摘要和标签。"""
        prompt = (
            "请为下面的文档片段提取标题、摘要和标签。\n"
            "输出格式必须严格为三行：\n"
            "TITLE: ...\nSUMMARY: ...\nTAGS: a, b, c\n\n"
            f"TEXT:\n{chunk.text}\n"
        )
        raw = self.client.generate(prompt)

        title = self._extract_prefixed_line(raw, "TITLE") or self._build_title(chunk)
        summary = self._extract_prefixed_line(raw, "SUMMARY") or self._build_summary(chunk.text)
        tags_line = self._extract_prefixed_line(raw, "TAGS")
        tags = [item.strip() for item in (tags_line or "").split(",") if item.strip()]
        if not tags:
            tags = self._extract_tags(chunk.text)
        return {"title": title, "summary": summary, "tags": tags}

    def _extract_prefixed_line(self, text: str, prefix: str) -> str:
        """提取类似 `PREFIX: ...` 的单行值。"""
        pattern = re.compile(rf"^{prefix}\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""
