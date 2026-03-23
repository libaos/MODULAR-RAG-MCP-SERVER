"""文档分块器。

当前阶段只解决一件事：

- 把 `Document` 切成 `Chunk` 列表

实现目标保持简单：

- 使用配置里的 `chunk_size` 和 `chunk_overlap`
- 优先按段落边界切分，边界不合适时再按字符窗口兜底
- 为每个 chunk 生成稳定 ID
- 继承文档元数据，并补充 `chunk_index` / `source_ref`

这里最容易混淆的两个概念：

- chunk:
  指“切出来的一段文本”本身。
  例如一篇长文被切成 4 段，这 4 段每一段都是一个 chunk。

- overlap:
  指“相邻两个 chunk 之间重复保留的那一小段内容”。
  它的作用是避免语义刚好断在边界上，导致后一个 chunk 缺上下文。

例子：

- 原文：`ABCDEFGHIJ`
- `chunk_size = 4`
- `chunk_overlap = 1`

切分结果可以是：

- chunk1 = `ABCD`
- chunk2 = `DEFG`
- chunk3 = `GHIJ`

这里：

- `ABCD` / `DEFG` / `GHIJ` 是 3 个 chunk
- `D` 和 `G` 是 overlap 带来的重复部分
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from modular_rag_repro.settings import Settings
from modular_rag_repro.types import Chunk, Document


class DocumentChunker:
    """最小可用的文档分块器。

    参数说明：

    - `chunk_size` 决定一个 chunk 最多装多少文本
    - `chunk_overlap` 决定下一个 chunk 要从上一个 chunk 末尾重复带多少文本

    所以：

    - `chunk_size` 控制“块有多大”
    - `chunk_overlap` 控制“块和块之间重复多少”
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunk_size = settings.ingestion.chunk_size
        self.chunk_overlap = settings.ingestion.chunk_overlap

        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

    def split_document(self, document: Document) -> List[Chunk]:
        """把一个 `Document` 切成多个 `Chunk`。

        输出结果里的每一项都是真正的 chunk。
        overlap 不会单独变成一项，它只是会出现在相邻两个 chunk 的边界中。
        """
        text = document.text.strip()
        if not text:
            raise ValueError(f"文档没有可切分文本: {document.id}")

        fragments = self._split_text(text)
        chunks: List[Chunk] = []

        for index, fragment in enumerate(fragments):
            chunk_text = fragment.strip()
            if not chunk_text:
                continue

            chunk_id = self._build_chunk_id(document.id, index, chunk_text)
            metadata = self._build_chunk_metadata(document, index, chunk_text)

            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        if not chunks:
            raise ValueError(f"分块结果为空: {document.id}")

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """执行最小分块策略。

        规则：

        1. 先按段落拆开
        2. 尽量把多个小段拼进一个 chunk
        3. 如果某段本身超过 chunk_size，就退化为固定窗口切分

        这里返回的是“chunk 文本列表”。
        overlap 只是这些文本之间的重复片段，不是单独返回的列表项。
        """
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        if not paragraphs:
            return self._window_split(text)

        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._window_split(paragraph))
                continue

            if not current:
                current = paragraph
                continue

            candidate = f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current.strip())
                current = self._merge_overlap(current, paragraph)

        if current:
            chunks.append(current.strip())

        return chunks

    def _window_split(self, text: str) -> List[str]:
        """按固定字符窗口切分，作为兜底策略。

        这是最能直接看出 chunk 和 overlap 区别的地方：

        - `end = start + chunk_size` 决定当前 chunk 截到哪里
        - `start = end - chunk_overlap` 决定下一个 chunk 从哪里重新开始

        也就是说：

        - `chunk_size` 决定本块长度
        - `chunk_overlap` 决定下块要从上块尾巴往回退多少字符再开始
        """
        chunks: List[str] = []
        start = 0
        text = text.strip()

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)

        return chunks

    def _merge_overlap(self, previous_chunk: str, next_paragraph: str) -> str:
        """构造带 overlap 的新 chunk 起始内容。

        这里可以把 overlap 理解成：

        - 不是新建一个“overlap 块”
        - 而是把“前一个 chunk 的尾巴”复制一点到下一个 chunk 的开头

        当前阶段的 overlap 只保留前一个 chunk 末尾的一小段字符，
        这样实现简单，也足够验证链路。
        """
        if self.chunk_overlap == 0:
            return next_paragraph

        overlap_text = previous_chunk[-self.chunk_overlap :].strip()
        if not overlap_text:
            return next_paragraph
        return f"{overlap_text}\n\n{next_paragraph}"

    def _build_chunk_id(self, document_id: str, index: int, text: str) -> str:
        """生成稳定的 chunk ID。"""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{document_id}_{index:04d}_{content_hash}"

    def _build_chunk_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> Dict[str, object]:
        """继承文档元数据并补充 chunk 级字段。"""
        metadata: Dict[str, object] = dict(document.metadata)
        metadata["chunk_index"] = chunk_index
        metadata["source_ref"] = document.id
        metadata["char_length"] = len(chunk_text)
        return metadata
