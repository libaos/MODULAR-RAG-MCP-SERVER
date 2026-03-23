"""Sparse Retriever。

当前阶段直接复用已经实现的 `BM25Indexer`：

- 从磁盘加载指定 collection 的 BM25 索引
- 用处理后的 query 做关键词检索
- 返回统一的 `RetrievalResult` 列表

这样做的目的，是先把 Query 链路里的“稀疏检索”补上，
后面再在它上面叠加融合和 rerank。
"""

from __future__ import annotations

from typing import List, Optional

from modular_rag_repro.ingestion.bm25_indexer import BM25Indexer
from modular_rag_repro.settings import Settings
from modular_rag_repro.types import ProcessedQuery, RetrievalResult


class SparseRetriever:
    """最小可用的 BM25 检索器。"""

    def __init__(self, settings: Settings, default_top_k: int | None = None) -> None:
        self.settings = settings
        self.default_top_k = default_top_k or settings.retrieval.sparse_top_k
        self.indexer = BM25Indexer(index_dir="data/db/bm25")

    def retrieve(
        self,
        processed_query: ProcessedQuery,
        collection: str = "default",
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """执行一次 BM25 检索。"""
        if not self.indexer.load(collection):
            return []

        query_text = self._build_query_text(processed_query)
        if not query_text.strip():
            return []

        return self.indexer.query(query_text=query_text, top_k=top_k or self.default_top_k)

    def _build_query_text(self, processed_query: ProcessedQuery) -> str:
        """构造给 BM25 用的查询文本。

        优先用提取后的关键词，避免把无意义口语词带进稀疏检索；
        如果关键词为空，再回退到标准化后的原 query。
        """
        if processed_query.keywords:
            return " ".join(processed_query.keywords)
        return processed_query.normalized_text
