"""Chroma 向量入库器。

当前阶段只解决一件事：

- 把 `Chunk` 和它对应的 embedding 写进本地 Chroma collection

实现目标保持简单：

- 只支持 `chroma` provider
- 使用本地 `PersistentClient`
- 使用 `upsert` 保证重复写入时按 `chunk_id` 覆盖
- 对 metadata 做最小清洗，避免复杂对象直接写入 Chroma 失败
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from modular_rag_repro.settings import Settings, resolve_path
from modular_rag_repro.types import Chunk


class ChromaUpserter:
    """最小可用的 Chroma 向量入库器。"""

    def __init__(self, settings: Settings, collection: str | None = None) -> None:
        self.settings = settings
        self.provider = settings.vector_store.provider.lower()
        self.persist_directory = resolve_path(settings.vector_store.persist_directory)
        self.default_collection = collection or settings.vector_store.collection_name

        if self.provider != "chroma":
            raise ValueError(f"当前阶段只支持 chroma provider，收到 provider={self.provider}")

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
        collection: str | None = None,
    ) -> int:
        """把 chunk 与向量写入指定 collection。"""
        if not chunks:
            raise ValueError("不能向 Chroma 写入空 chunk 列表")
        if not vectors:
            raise ValueError("不能向 Chroma 写入空向量列表")
        if len(chunks) != len(vectors):
            raise ValueError(f"chunk 数量和向量数量不一致: chunks={len(chunks)} vectors={len(vectors)}")

        chroma_collection = self._get_collection(collection)
        chroma_collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._sanitize_metadata(chunk.metadata) for chunk in chunks],
            embeddings=vectors,
        )
        return len(chunks)

    def get_collection_count(self, collection: str | None = None) -> int:
        """返回当前 collection 中的向量条数。"""
        return self._get_collection(collection).count()

    def get_persist_directory(self) -> Path:
        """返回 Chroma 数据目录。"""
        return self.persist_directory

    def _get_collection(self, collection: str | None = None):
        """获取或创建一个 collection。"""
        collection_name = collection or self.default_collection
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """把复杂 metadata 转成 Chroma 可接受的简单标量。"""
        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue

            clean_key = str(key)
            if isinstance(value, (str, int, float, bool)):
                sanitized[clean_key] = value
                continue

            # 对列表、字典等复杂对象统一转成 JSON 字符串，先保证链路能写通。
            sanitized[clean_key] = json.dumps(value, ensure_ascii=False, sort_keys=True)

        return sanitized
