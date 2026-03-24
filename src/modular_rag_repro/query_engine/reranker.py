"""最小可用的 provider 化 reranker。"""

from __future__ import annotations

import re
from typing import List, Sequence

from modular_rag_repro.llm import OllamaClient
from modular_rag_repro.settings import Settings
from modular_rag_repro.types import ProcessedQuery, RetrievalResult


class SimpleReranker:
    """基于 provider 的轻量重排器。

    设计目标：

    - 不引入额外模型依赖
    - 保留原始召回分数
    - 当 query 关键词能明显区分结果时，尽量把更相关的 chunk 往前提
    """

    def __init__(
        self,
        provider: str = "simple",
        top_k: int = 5,
        settings: Settings | None = None,
        fallback_provider: str = "simple",
        timeout: float = 30.0,
    ) -> None:
        self.provider = provider.lower()
        self.top_k = top_k
        self.settings = settings
        self.fallback_provider = fallback_provider.lower()
        self.timeout = timeout
        self.client = None
        if settings is not None:
            self.client = OllamaClient(
                base_url=settings.llm.base_url,
                model=settings.rerank.model or settings.llm.model,
                temperature=0.0,
                max_tokens=64,
                timeout=timeout,
            )

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
        if self.provider == "simple":
            return self._rerank_with_simple(processed_query, results, effective_top_k)
        if self.provider == "llm":
            try:
                return self._rerank_with_llm(processed_query, results, effective_top_k)
            except Exception:
                return self._rerank_with_fallback(processed_query, results, effective_top_k, fallback_from="llm")
        if self.provider == "cross_encoder":
            try:
                return self._rerank_with_cross_encoder(processed_query, results, effective_top_k)
            except Exception:
                return self._rerank_with_fallback(processed_query, results, effective_top_k, fallback_from="cross_encoder")
        raise ValueError(f"不支持的 rerank provider: {self.provider}")

    def _rerank_with_simple(
        self,
        processed_query: ProcessedQuery,
        results: Sequence[RetrievalResult],
        effective_top_k: int,
    ) -> List[RetrievalResult]:
        """执行 simple 规则重排。"""
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
                            "rerank_provider": "simple",
                            "matched_terms": matched_terms,
                            "exact_match": exact_match,
                        },
                    ),
                )
            )

        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [item for _, _, item in ranked[:effective_top_k]]

    def _rerank_with_fallback(
        self,
        processed_query: ProcessedQuery,
        results: Sequence[RetrievalResult],
        effective_top_k: int,
        fallback_from: str,
    ) -> List[RetrievalResult]:
        """增强 provider 失败后的 fallback。"""
        fallback_provider = self.fallback_provider if self.fallback_provider != "none" else "simple"
        if fallback_provider != "simple":
            raise RuntimeError(f"当前仅支持 fallback 到 simple，收到 {fallback_provider}")

        reranked = self._rerank_with_simple(processed_query, results, effective_top_k)
        patched = []
        for item in reranked:
            patched.append(
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    score=item.score,
                    text=item.text,
                    metadata={
                        **item.metadata,
                        "rerank_fallback": fallback_from,
                    },
                )
            )
        return patched

    def _rerank_with_llm(
        self,
        processed_query: ProcessedQuery,
        results: Sequence[RetrievalResult],
        effective_top_k: int,
    ) -> List[RetrievalResult]:
        """使用 LLM 对结果打分。"""
        if self.client is None:
            raise RuntimeError("llm rerank 需要 settings/ollama client")

        ranked = []
        for item in results:
            prompt = (
                "你是检索重排器。请只返回 0 到 100 的分数，数字越高表示越相关。\n"
                f"query: {processed_query.normalized_text}\n"
                f"title: {item.metadata.get('title', '')}\n"
                f"text: {item.text}\n"
                "score:"
            )
            raw = self.client.generate(prompt)
            match = re.search(r"-?\d+(?:\.\d+)?", raw)
            if not match:
                raise RuntimeError(f"llm rerank 无法解析分数: {raw}")
            llm_score = float(match.group(0))
            final_score = llm_score + (float(item.score) * 0.1)
            ranked.append(
                (
                    final_score,
                    item.chunk_id,
                    RetrievalResult(
                        chunk_id=item.chunk_id,
                        score=final_score,
                        text=item.text,
                        metadata={
                            **item.metadata,
                            "original_score": item.score,
                            "rerank_score": final_score,
                            "reranked": True,
                            "rerank_provider": "llm",
                        },
                    ),
                )
            )
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [item for _, _, item in ranked[:effective_top_k]]

    def _rerank_with_cross_encoder(
        self,
        processed_query: ProcessedQuery,
        results: Sequence[RetrievalResult],
        effective_top_k: int,
    ) -> List[RetrievalResult]:
        """尝试使用本地 cross-encoder。"""
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError("cross_encoder 依赖不可用") from exc

        model_name = self.client.model if self.client is not None else "cross-encoder/ms-marco-MiniLM-L-6-v2"
        model = CrossEncoder(model_name)
        pairs = [(processed_query.normalized_text, item.text) for item in results]
        scores = model.predict(pairs)
        ranked = []
        for item, score in zip(results, scores):
            score_value = float(score)
            ranked.append(
                (
                    score_value,
                    item.chunk_id,
                    RetrievalResult(
                        chunk_id=item.chunk_id,
                        score=score_value,
                        text=item.text,
                        metadata={
                            **item.metadata,
                            "original_score": item.score,
                            "rerank_score": score_value,
                            "reranked": True,
                            "rerank_provider": "cross_encoder",
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
