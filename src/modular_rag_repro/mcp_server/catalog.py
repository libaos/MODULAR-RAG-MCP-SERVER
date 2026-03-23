"""本地知识库目录与文档聚合辅助。

这个模块给 MCP tools 提供最小的“集合信息”和“文档聚合”能力：

- 枚举已有 collection
- 汇总 Chroma/BM25 统计
- 基于 BM25 索引按文档聚合 chunk
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from modular_rag_repro.settings import Settings, resolve_path


@dataclass(slots=True)
class CollectionSummary:
    """集合概览信息。"""

    name: str
    chroma_count: Optional[int] = None
    bm25_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentSummary:
    """文档级摘要信息。"""

    doc_id: str
    collection: str
    title: str
    source_path: str
    chunk_count: int
    summary: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeCatalog:
    """最小可用的本地知识库目录器。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chroma_dir = resolve_path(settings.vector_store.persist_directory)
        self.bm25_dir = resolve_path("data/db/bm25")

    def list_collections(self, include_stats: bool = True) -> List[CollectionSummary]:
        """枚举本地已有 collection。"""
        collections: Dict[str, CollectionSummary] = {}

        for collection in self._list_chroma_collections():
            collections[collection.name] = collection

        for collection_name, bm25_count in self._list_bm25_collections().items():
            summary = collections.get(collection_name)
            if summary is None:
                summary = CollectionSummary(name=collection_name)
                collections[collection_name] = summary
            if include_stats:
                summary.bm25_count = bm25_count

        return sorted(collections.values(), key=lambda item: item.name)

    def get_document_summary(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> DocumentSummary:
        """按文档 ID 聚合 chunk，返回文档摘要。"""
        target_collections = [collection] if collection else list(self._list_bm25_collections().keys())
        for collection_name in target_collections:
            index_payload = self._load_bm25_payload(collection_name)
            if index_payload is None:
                continue

            documents = index_payload.get("documents", {})
            matched_chunks = []
            for chunk_id, doc_info in documents.items():
                metadata = doc_info.get("metadata", {}) or {}
                source_ref = metadata.get("source_ref")
                if source_ref == doc_id or str(chunk_id).startswith(doc_id):
                    matched_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "text": doc_info.get("text", ""),
                            "metadata": metadata,
                        }
                    )

            if matched_chunks:
                matched_chunks.sort(key=lambda item: item["metadata"].get("chunk_index", 0))
                first = matched_chunks[0]
                first_metadata = first["metadata"]
                title = str(first_metadata.get("title") or Path(first_metadata.get("source_path", "Untitled")).stem or "Untitled")
                source_path = str(first_metadata.get("source_path", "(unknown)"))
                tags = self._extract_tags(first_metadata)
                summary = self._build_summary(matched_chunks)
                metadata = {
                    "collection": collection_name,
                    "doc_type": first_metadata.get("doc_type"),
                    "page_count": first_metadata.get("page_count"),
                }
                return DocumentSummary(
                    doc_id=str(first_metadata.get("source_ref", doc_id)),
                    collection=collection_name,
                    title=title,
                    source_path=source_path,
                    chunk_count=len(matched_chunks),
                    summary=summary,
                    tags=tags,
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )

        raise ValueError(f"Document '{doc_id}' not found")

    def _list_chroma_collections(self) -> List[CollectionSummary]:
        """读取 Chroma 中的 collection。"""
        if not self.chroma_dir.exists():
            return []

        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        collections: List[CollectionSummary] = []
        for collection in client.list_collections():
            summary = CollectionSummary(
                name=collection.name,
                chroma_count=collection.count(),
                metadata=dict(collection.metadata or {}),
            )
            collections.append(summary)
        return collections

    def _list_bm25_collections(self) -> Dict[str, int]:
        """读取 BM25 目录中的 collection。"""
        if not self.bm25_dir.exists():
            return {}

        results: Dict[str, int] = {}
        for child in self.bm25_dir.iterdir():
            if not child.is_dir():
                continue
            index_path = child / "bm25_index.json"
            if not index_path.exists():
                continue
            payload = self._load_json(index_path)
            num_docs = int(payload.get("metadata", {}).get("num_docs", 0))
            results[child.name] = num_docs
        return results

    def _load_bm25_payload(self, collection: str) -> Optional[Dict[str, Any]]:
        """加载某个 collection 的 BM25 索引 JSON。"""
        path = self.bm25_dir / collection / "bm25_index.json"
        if not path.exists():
            return None
        return self._load_json(path)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """读取 JSON 文件。"""
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _extract_tags(self, metadata: Dict[str, Any]) -> List[str]:
        """从 metadata 中抽取标签。"""
        tags: List[str] = []
        doc_type = metadata.get("doc_type")
        if doc_type:
            tags.append(str(doc_type))
        return tags

    def _build_summary(self, chunks: List[Dict[str, Any]], max_length: int = 240) -> str:
        """把首批 chunk 压缩成文档摘要。"""
        text = " ".join(chunk.get("text", "") for chunk in chunks[:2]).strip()
        compact = " ".join(text.split())
        if len(compact) <= max_length:
            return compact
        return compact[: max_length - 3] + "..."
