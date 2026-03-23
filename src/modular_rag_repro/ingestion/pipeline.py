"""摄取链路编排器。

这个文件当前只做一件事：

- 把 `PdfLoader` 和 `DocumentChunker` 串起来

也就是说，它不负责：

- 向量化
- BM25
- Chroma 入库
- 图片落盘

先把最短链路打通，再逐步把后续步骤接进来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modular_rag_repro.ingestion.chunker import DocumentChunker
from modular_rag_repro.ingestion.pdf_loader import PdfLoader
from modular_rag_repro.settings import Settings
from modular_rag_repro.types import Chunk, Document, TraceContext


@dataclass(slots=True)
class PipelineResult:
    """一次摄取执行的结果。

    当前阶段先保留最基础的字段：

    - 是否成功
    - 原始文档对象
    - 切分后的 chunk 列表
    - 统计信息
    - 错误信息
    """

    success: bool
    file_path: str
    document: Optional[Document] = None
    chunks: List[Chunk] = field(default_factory=list)
    error: Optional[str] = None
    stages: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        """返回 chunk 数量。"""
        return len(self.chunks)


class IngestionPipeline:
    """最小可用的摄取编排器。"""

    def __init__(
        self,
        settings: Settings,
        collection: str = "default",
        loader: Optional[PdfLoader] = None,
        chunker: Optional[DocumentChunker] = None,
    ) -> None:
        self.settings = settings
        self.collection = collection
        self.loader = loader or PdfLoader(extract_images=True)
        self.chunker = chunker or DocumentChunker(settings)

    def run(self, file_path: str, trace: Optional[TraceContext] = None) -> PipelineResult:
        """执行最小摄取链路。

        当前顺序固定为：

        1. 读取 PDF
        2. 切分 chunk
        """
        stages: Dict[str, Any] = {}

        try:
            document = self.loader.load(file_path)
            stages["load"] = {
                "doc_id": document.id,
                "text_length": len(document.text),
                "page_count": document.metadata.get("page_count", 0),
                "image_count": len(document.metadata.get("images", [])),
            }

            if trace is not None:
                trace.record_stage("load", stages["load"])

            chunks = self.chunker.split_document(document)
            stages["chunk"] = {
                "chunk_count": len(chunks),
                "first_chunk_id": chunks[0].id if chunks else None,
            }

            if trace is not None:
                trace.record_stage("chunk", stages["chunk"])

            return PipelineResult(
                success=True,
                file_path=file_path,
                document=document,
                chunks=chunks,
                stages=stages,
            )
        except Exception as exc:
            return PipelineResult(
                success=False,
                file_path=file_path,
                error=str(exc),
                stages=stages,
            )
