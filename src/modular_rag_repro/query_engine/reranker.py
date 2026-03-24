"""最小可用的本地 reranker。"""

from __future__ import annotations

import re
from typing import List, Sequence

from modular_rag_repro.types import ProcessedQuery, RetrievalResult


class SimpleReranker:
    """基于关键词覆盖率的轻量重排器。

    设计目标：

    - 不引入额外模型依赖
    - 保留原始召回分数
    - 当 query 关键词能明显区分结果时，尽量把更相关的 chunk 往前提
    """

    def __init__(self, provider: str = "simple", top_k: int = 5) -> None:
        self.provider = provider.lower()
        self.top_k = top_k

    def rerank(
        self,
        processed_query: ProcessedQuery,
        results: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """对融合后的结果做轻量重排。"""
        if not results:
            return []

        effective_top_k = top_k or self.top_k
        if self.provider == "none":
            return list(results[:effective_top_k])
        if self.provider != "simple":
            raise ValueError(f"不支持的 rerank provider: {self.provider}")

        query_terms = processed_query.keywords or self._tokenize(processed_query.normalized_text)
        if not query_terms:
            return list(results[:effective_top_k])

        ranked = []
        for item in results:
            rerank_score, matched_terms, exact_match = self._score_item(query_terms, processed_query.normalized_text, item)
            ranked.append(
                (
                    rerank_score,
                    item.chunk_id,
                    RetrievalResult(
                        chunk_id=item.chunk_id,
                        score=rerank_score,
                        text=item.text,
                        metadata={
                            **item.metadata,
                            "original_score": item.score,
                            "rerank_score": rerank_score,
                            "reranked": True,
                            "rerank_provider": self.provider,
                            "matched_terms": matched_terms,
                            "exact_match": exact_match,
                        },
                    ),
                )
            )

        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [item for _, _, item in ranked[:effective_top_k]]

    def _score_item(
        self,
        query_terms: Sequence[str],
        normalized_query: str,
        item: RetrievalResult,
    ) -> tuple[float, List[str], bool]:
        """计算单条结果的 rerank 分数。"""
        haystack = self._build_haystack(item)
        matched_terms = [term for term in query_terms if term and term in haystack]
        coverage = len(matched_terms) / max(len(query_terms), 1)
        exact_match = bool(normalized_query and normalized_query.lower() in haystack)

        base_score = float(item.score)
        rerank_score = (coverage * 2.0) + (0.5 if exact_match else 0.0) + (base_score * 0.25)
        return rerank_score, matched_terms, exact_match

    def _build_haystack(self, item: RetrievalResult) -> str:
        """构造参与匹配的文本池。"""
        parts = [
            item.text or "",
            str(item.metadata.get("title", "")),
            str(item.metadata.get("source_path", "")),
        ]
        return " ".join(parts).lower()

    def _tokenize(self, text: str) -> List[str]:
        """对没有关键词的 query 做最小切词。"""
        return re.findall(r"[a-z0-9_\u4e00-\u9fff]+", text.lower())
