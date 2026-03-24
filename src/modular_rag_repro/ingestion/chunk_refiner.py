"""ChunkRefiner：在 embedding 前做一层轻量文本清洗。

当前实现只做规则版，不接 LLM。

它解决的核心问题很简单：

- PDF 抽出来的文本里常有多余空白
- 有些 chunk 会带 HTML 注释或零宽字符
- 行尾和段落间距可能很乱

所以这里先做一层“低风险清洗”：

- 统一换行符
- 去掉零宽字符 / NBSP
- 去掉 HTML 注释
- 压缩连续空白和空行

这样可以让后面的 embedding 和 BM25 输入更稳定。
"""

from __future__ import annotations

import re
from typing import List

from modular_rag_repro.llm import OllamaClient
from modular_rag_repro.settings import Settings
from modular_rag_repro.types import Chunk


class ChunkRefiner:
    """最小可用的规则版 chunk 清洗器。"""

    _HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
    _TAG_BREAK_RE = re.compile(r"<\s*(br|/p|p)\s*/?\s*>", re.IGNORECASE)
    _GENERIC_TAG_RE = re.compile(r"<[^>\n]+>")
    _ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
    _MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.use_llm = settings.ingestion.chunk_refiner.use_llm
        self.provider = settings.ingestion.chunk_refiner.provider.lower()
        self.model = settings.ingestion.chunk_refiner.model or settings.llm.model
        self.timeout = settings.ingestion.chunk_refiner.timeout_seconds
        self.client = OllamaClient(
            base_url=settings.llm.base_url,
            model=self.model,
            temperature=0.0,
            max_tokens=min(settings.llm.max_tokens, 512),
            timeout=self.timeout,
        )

    def refine_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """批量清洗 chunk。

        当前不会因为 `use_llm=true` 就真的调用 LLM。
        这个字段先保留，只是为了和配置接口对齐，方便后面扩展。
        """
        refined_chunks: List[Chunk] = []
        for chunk in chunks:
            refined_chunks.append(self.refine_chunk(chunk))
        return refined_chunks

    def refine_chunk(self, chunk: Chunk) -> Chunk:
        """清洗单个 chunk，并在 metadata 里标记清洗结果。"""
        refined_by = "rule"
        refiner_mode = "rule_only"
        refiner_fallback = None
        if self.use_llm and self.provider == "llm":
            try:
                refined_text = self._llm_refine(chunk.text)
                refined_by = "llm"
                refiner_mode = "llm"
            except Exception:
                refined_text = self._rule_based_refine(chunk.text)
                refiner_mode = "fallback_rule"
                refiner_fallback = "llm"
        else:
            refined_text = self._rule_based_refine(chunk.text)
        changed = refined_text != chunk.text

        metadata = dict(chunk.metadata)
        metadata["refined_by"] = refined_by
        metadata["refiner_mode"] = refiner_mode
        if refiner_fallback:
            metadata["refiner_fallback"] = refiner_fallback
        metadata["refinement_changed"] = changed
        metadata["original_char_length"] = len(chunk.text)
        metadata["refined_char_length"] = len(refined_text)

        return Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            text=refined_text,
            metadata=metadata,
        )

    def _rule_based_refine(self, text: str) -> str:
        """执行最小规则清洗。"""
        refined = text.replace("\r\n", "\n").replace("\r", "\n")
        refined = refined.replace("\xa0", " ")
        refined = self._ZERO_WIDTH_RE.sub("", refined)
        refined = self._HTML_COMMENT_RE.sub("", refined)
        refined = self._TAG_BREAK_RE.sub("\n", refined)
        refined = self._GENERIC_TAG_RE.sub(" ", refined)

        lines = []
        for line in refined.split("\n"):
            normalized = self._MULTI_SPACE_RE.sub(" ", line).strip()
            lines.append(normalized)
        refined = "\n".join(lines)
        refined = self._MULTI_NEWLINE_RE.sub("\n\n", refined)
        refined = refined.strip()

        # 如果规则清洗意外把内容清空，就退回原文，避免把有效文本误删。
        return refined if refined else text.strip()

    def _llm_refine(self, text: str) -> str:
        """调用 LLM 进行最小文本精炼。"""
        prompt = (
            "请对下面文本做轻量清洗，只做格式整理，不改变事实内容。\n"
            "要求：去掉明显噪声、合并多余空白、保留原语言。\n\n"
            f"文本：\n{text}\n\n"
            "输出清洗后的正文："
        )
        refined = self.client.generate(prompt).strip()
        return refined or self._rule_based_refine(text)
