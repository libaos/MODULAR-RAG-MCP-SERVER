"""文档生命周期管理器。

Phase 7 从这里开始补：

- 集合统计：直接复用现有 catalog
- 文档摘要：直接复用现有 catalog
- 文档删除：同时清理 Chroma 与 BM25
- 索引重建钩子：基于当前 Chroma 内容重建 BM25
"""

from __future__ import annotations

from modular_rag_repro.ingestion import BM25Indexer, ChromaUpserter, ImageStorage, SQLiteIntegrityChecker
from modular_rag_repro.mcp_server.catalog import CollectionSummary, DocumentSummary, KnowledgeCatalog
from modular_rag_repro.settings import Settings
from modular_rag_repro.types import DeleteDocumentResult


class DocumentManager:
    """统一管理 collection / document 的生命周期。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = KnowledgeCatalog(settings)
        self.bm25 = BM25Indexer()
        self.chroma = ChromaUpserter(settings)
        self.image_storage = ImageStorage(
            db_path=settings.ingestion.image_index_db_path,
            images_root=settings.ingestion.images_root_dir,
        )
        self.integrity = SQLiteIntegrityChecker(settings.ingestion.integrity_db_path)

    def list_collections(self, include_stats: bool = True) -> list[CollectionSummary]:
        """列出集合统计。"""
        return self.catalog.list_collections(include_stats=include_stats)

    def list_documents(self, collection: str) -> list[DocumentSummary]:
        """列出某个 collection 下的文档。"""
        return self.catalog.list_documents(collection)

    def get_document_summary(self, doc_id: str, collection: str | None = None) -> DocumentSummary:
        """按文档 ID 读取摘要。"""
        return self.catalog.get_document_summary(doc_id=doc_id, collection=collection)

    def rebuild_collection_indexes(self, collection: str) -> int:
        """从当前 Chroma 内容重建 BM25。

        这是最小可用的“索引重建钩子”：

        - 如果 Chroma 里还有 chunk，就完整重建 BM25
        - 如果 Chroma 已空，就删除 BM25 索引文件
        """
        current_chunks = self.chroma.export_chunks(collection=collection)
        return self.bm25.rebuild_from_chunks(current_chunks, collection=collection)

    def delete_document(self, doc_id: str, collection: str | None = None) -> DeleteDocumentResult:
        """删除一个文档，并同步清理 Chroma 与 BM25。"""
        summary = self.get_document_summary(doc_id=doc_id, collection=collection)
        target_collection = summary.collection

        existing_chunks = self.chroma.export_chunks(collection=target_collection)
        if not existing_chunks:
            existing_chunks = self.bm25.load_chunks(collection=target_collection)

        target_chunks = [chunk for chunk in existing_chunks if self._match_doc_id(chunk.metadata, chunk.id, doc_id)]
        if not target_chunks:
            raise ValueError(f"Document '{doc_id}' not found in collection '{target_collection}'")

        deleted_chunk_ids = [chunk.id for chunk in target_chunks]
        chroma_deleted_count = self.chroma.delete_chunks(deleted_chunk_ids, collection=target_collection)

        remaining_count = self.rebuild_collection_indexes(target_collection)
        collection_removed = False
        if self.chroma.get_collection_count(target_collection) == 0:
            collection_removed = self.chroma.delete_collection(target_collection)

        doc_hash = summary.metadata.get("doc_hash")
        if doc_hash:
            self.image_storage.delete_document_images(str(doc_hash), target_collection)
        if doc_hash:
            self.integrity.remove_record(str(doc_hash), target_collection)

        return DeleteDocumentResult(
            doc_id=summary.doc_id,
            collection=target_collection,
            deleted_chunk_ids=deleted_chunk_ids,
            chroma_deleted_count=chroma_deleted_count,
            bm25_remaining_count=remaining_count,
            collection_removed=collection_removed,
            rebuilt=True,
        )

    def _match_doc_id(self, metadata: dict, chunk_id: str, doc_id: str) -> bool:
        """判断某个 chunk 是否属于目标文档。"""
        source_ref = str(metadata.get("source_ref") or "")
        if source_ref == doc_id:
            return True
        return str(chunk_id).startswith(doc_id)
