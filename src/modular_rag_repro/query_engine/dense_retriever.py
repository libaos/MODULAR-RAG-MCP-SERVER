"""Dense Retriever。

当前阶段先做最小能力：

- 用 `EmbeddingEncoder` 把查询文本编码成向量
- 从本地 Chroma collection 中做向量检索
- 转换成统一的 `RetrievalResult`

后面再补：

- 多路检索融合
- rerank
- 更复杂的 metadata filter
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb

from modular_rag_repro.ingestion.embedding_encoder import EmbeddingEncoder
from modular_rag_repro.settings import Settings, resolve_path
from modular_rag_repro.types import ProcessedQuery, RetrievalResult


class DenseRetriever:
    """最小可用的向量检索器。"""

    def __init__(self, settings: Settings, default_top_k: int | None = None) -> None:
        self.settings = settings
        self.default_top_k = default_top_k or settings.retrieval.dense_top_k
        self.persist_directory = resolve_path(settings.vector_store.persist_directory)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.embedding_encoder = EmbeddingEncoder(settings, batch_size=1)

    def retrieve(
        self,
        processed_query: ProcessedQuery,
        collection: str = "default",
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """执行一次 Dense 检索。"""
        query_text = processed_query.normalized_text.strip()
        if not query_text:
            return []

        query_vector = self.embedding_encoder.encode_texts([query_text])[0]
        collection_obj = self._get_collection(collection)
        if collection_obj is None:
            return []

        total_count = collection_obj.count()
        if total_count == 0:
            return []

        where = processed_query.filters or None
        result = collection_obj.query(
            query_embeddings=[query_vector],
            n_results=min(top_k or self.default_top_k, total_count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_retrieval_results(result)

    def _get_collection(self, collection: str):
        """获取 collection，不存在时直接返回 None。"""
        try:
            return self.client.get_collection(name=collection)
        except Exception:
            return None

    def _to_retrieval_results(self, payload: Dict[str, Any]) -> List[RetrievalResult]:
        """把 Chroma 查询结果转换成统一格式。"""
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]
        ids = payload.get("ids", [[]])[0]

        results: List[RetrievalResult] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            score = self._distance_to_score(distance)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=text or "",
                    metadata=dict(metadata),
                )
            )
        return results

    def _distance_to_score(self, distance: float) -> float:
        """把 Chroma 返回的距离转成更直观的分数。"""
        if distance is None:
            return 0.0
        return max(0.0, 1.0 - float(distance))
