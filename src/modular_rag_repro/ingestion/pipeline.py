"""摄取链路编排器。

这个文件现在负责把最小可用的离线摄取链路串起来：

- `PdfLoader`
- `DocumentChunker`
- `EmbeddingEncoder`
- `BM25Indexer`
- `ChromaUpserter`

也就是说，当前最小闭环已经是：

- `PDF -> Document -> Chunks -> Embeddings -> BM25 -> Chroma`

还没有接入的能力依然包括：

- 图片落盘
- 元数据增强
- 多阶段 Trace 持久化
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional

from modular_rag_repro.ingestion.bm25_indexer import BM25Indexer
from modular_rag_repro.ingestion.chunker import DocumentChunker
from modular_rag_repro.ingestion.chroma_upserter import ChromaUpserter
from modular_rag_repro.ingestion.embedding_encoder import EmbeddingEncoder
from modular_rag_repro.ingestion.file_integrity import SQLiteIntegrityChecker
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
    - 生成的向量数量
    - 写入 Chroma 的数量
    - 统计信息
    - 错误信息
    """

    success: bool
    file_path: str
    file_hash: Optional[str] = None
    document: Optional[Document] = None
    chunks: List[Chunk] = field(default_factory=list)
    vector_count: int = 0
    upserted_count: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None
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
        embedding_encoder: Optional[EmbeddingEncoder] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        chroma_upserter: Optional[ChromaUpserter] = None,
        integrity_checker: Optional[SQLiteIntegrityChecker] = None,
    ) -> None:
        self.settings = settings
        self.collection = collection
        self.loader = loader or PdfLoader(extract_images=True)
        self.chunker = chunker or DocumentChunker(settings)
        self.embedding_encoder = embedding_encoder or EmbeddingEncoder(settings)
        self.bm25_indexer = bm25_indexer or BM25Indexer(index_dir="data/db/bm25")
        self.chroma_upserter = chroma_upserter or ChromaUpserter(settings, collection=collection)
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker(settings.ingestion.integrity_db_path)

    def run(
        self,
        file_path: str,
        trace: Optional[TraceContext] = None,
        force: bool = False,
    ) -> PipelineResult:
        """执行最小摄取链路。

        当前顺序固定为：

        1. 做 SHA256 幂等检查
        2. 读取 PDF
        3. 切分 chunk
        4. 生成 embedding
        5. 构建 BM25 索引
        6. 写入 Chroma
        """
        stages: Dict[str, Any] = {}
        file_hash: str | None = None

        try:
            stage_t0 = perf_counter()
            file_hash = self.integrity_checker.compute_sha256(file_path)
            should_skip = (not force) and self.integrity_checker.should_skip(file_hash, self.collection)
            stages["integrity"] = {
                "file_hash": file_hash,
                "collection": self.collection,
                "force": force,
                "should_skip": should_skip,
            }

            if trace is not None:
                trace.record_stage("integrity", stages["integrity"], elapsed_ms=(perf_counter() - stage_t0) * 1000.0)

            if should_skip:
                return PipelineResult(
                    success=True,
                    file_path=file_path,
                    file_hash=file_hash,
                    skipped=True,
                    skip_reason="already_processed",
                    stages=stages,
                )

            stage_t0 = perf_counter()
            document = self.loader.load(file_path)
            stages["load"] = {
                "doc_id": document.id,
                "text_length": len(document.text),
                "page_count": document.metadata.get("page_count", 0),
                "image_count": len(document.metadata.get("images", [])),
            }

            if trace is not None:
                trace.record_stage("load", stages["load"], elapsed_ms=(perf_counter() - stage_t0) * 1000.0)

            stage_t0 = perf_counter()
            chunks = self.chunker.split_document(document)
            stages["chunk"] = {
                "chunk_count": len(chunks),
                "first_chunk_id": chunks[0].id if chunks else None,
            }

            if trace is not None:
                trace.record_stage("chunk", stages["chunk"], elapsed_ms=(perf_counter() - stage_t0) * 1000.0)

            stage_t0 = perf_counter()
            vectors = self.embedding_encoder.encode_chunks(chunks)
            stages["embed"] = {
                "vector_count": len(vectors),
                "vector_dim": len(vectors[0]) if vectors else 0,
            }

            if trace is not None:
                trace.record_stage("embed", stages["embed"], elapsed_ms=(perf_counter() - stage_t0) * 1000.0)

            stage_t0 = perf_counter()
            self.bm25_indexer.build(chunks, collection=self.collection)
            stages["bm25"] = {
                "indexed_chunks": len(chunks),
                "index_path": str(self.bm25_indexer.get_index_path(self.collection)),
            }

            if trace is not None:
                trace.record_stage("bm25", stages["bm25"], elapsed_ms=(perf_counter() - stage_t0) * 1000.0)

            stage_t0 = perf_counter()
            upserted_count = self.chroma_upserter.upsert_chunks(chunks, vectors, collection=self.collection)
            stages["chroma"] = {
                "upserted_count": upserted_count,
                "collection_count": self.chroma_upserter.get_collection_count(self.collection),
                "persist_directory": str(self.chroma_upserter.get_persist_directory()),
            }

            if trace is not None:
                trace.record_stage("chroma", stages["chroma"], elapsed_ms=(perf_counter() - stage_t0) * 1000.0)

            if file_hash is not None:
                self.integrity_checker.mark_success(file_hash, file_path, self.collection)

            return PipelineResult(
                success=True,
                file_path=file_path,
                file_hash=file_hash,
                document=document,
                chunks=chunks,
                vector_count=len(vectors),
                upserted_count=upserted_count,
                stages=stages,
            )
        except Exception as exc:
            if file_hash is not None:
                self.integrity_checker.mark_failed(file_hash, file_path, self.collection, str(exc))
            return PipelineResult(
                success=False,
                file_path=file_path,
                file_hash=file_hash,
                error=str(exc),
                stages=stages,
            )
