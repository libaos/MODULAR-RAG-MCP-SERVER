"""统一的查询执行工作流。

把 QueryProcessor / Dense / Sparse / Fusion / ResponseFormatter 串起来，
并在一次执行中生成可持久化的 query trace。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

from modular_rag_repro.query_engine.dense_retriever import DenseRetriever
from modular_rag_repro.query_engine.fusion import RRFFusion
from modular_rag_repro.query_engine.query_processor import QueryProcessor
from modular_rag_repro.query_engine.reranker import SimpleReranker
from modular_rag_repro.query_engine.sparse_retriever import SparseRetriever
from modular_rag_repro.response import ResponseFormatter
from modular_rag_repro.settings import Settings
from modular_rag_repro.trace import TraceCollector
from modular_rag_repro.types import ProcessedQuery, QueryResponse, RetrievalResult, TraceContext


@dataclass(slots=True)
class QueryWorkflowResult:
    """一次完整查询工作流的输出。"""

    processed_query: ProcessedQuery
    dense_results: List[RetrievalResult]
    sparse_results: List[RetrievalResult]
    fusion_results: List[RetrievalResult]
    final_results: List[RetrievalResult]
    formatted_response: QueryResponse
    trace: TraceContext


class QueryWorkflow:
    """最小可用的 hybrid 查询工作流。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.processor = QueryProcessor()
        self.dense_retriever = DenseRetriever(settings)
        self.sparse_retriever = SparseRetriever(settings)
        self.fusion = RRFFusion(k=settings.retrieval.rrf_k)
        self.reranker = SimpleReranker(
            provider=settings.rerank.provider,
            top_k=settings.rerank.top_k,
        )
        self.formatter = ResponseFormatter()
        self.trace_collector = TraceCollector(settings.observability.trace_file)

    def run(
        self,
        query: str,
        collection: str,
        top_k: int,
        no_rerank: bool = False,
        source: str = "cli",
    ) -> QueryWorkflowResult:
        """执行完整查询链路，并把 trace 落盘。"""
        trace = TraceContext(
            trace_type="query",
            metadata={
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "source": source,
            },
        )

        stage_t0 = perf_counter()
        processed_query = self.processor.process(query)
        trace.record_stage(
            "process_query",
            payload={
                "normalized_text": processed_query.normalized_text,
                "keywords": processed_query.keywords,
                "filters": processed_query.filters,
            },
            elapsed_ms=(perf_counter() - stage_t0) * 1000.0,
        )

        stage_t0 = perf_counter()
        dense_results = self.dense_retriever.retrieve(
            processed_query=processed_query,
            collection=collection,
            top_k=top_k,
        )
        trace.record_stage(
            "dense",
            payload={
                "result_count": len(dense_results),
                "top_chunk_id": dense_results[0].chunk_id if dense_results else None,
            },
            elapsed_ms=(perf_counter() - stage_t0) * 1000.0,
        )

        stage_t0 = perf_counter()
        sparse_results = self.sparse_retriever.retrieve(
            processed_query=processed_query,
            collection=collection,
            top_k=top_k,
        )
        trace.record_stage(
            "sparse",
            payload={
                "result_count": len(sparse_results),
                "top_chunk_id": sparse_results[0].chunk_id if sparse_results else None,
            },
            elapsed_ms=(perf_counter() - stage_t0) * 1000.0,
        )

        stage_t0 = perf_counter()
        fusion_results = self.fusion.fuse(
            ranking_lists=[dense_results, sparse_results],
            top_k=top_k,
        )
        trace.record_stage(
            "fusion",
            payload={
                "result_count": len(fusion_results),
                "top_chunk_id": fusion_results[0].chunk_id if fusion_results else None,
            },
            elapsed_ms=(perf_counter() - stage_t0) * 1000.0,
        )

        stage_t0 = perf_counter()
        if no_rerank or not self.settings.rerank.enabled:
            final_results = fusion_results
            rerank_mode = "disabled"
            rerank_provider = "none"
        else:
            try:
                final_results = self.reranker.rerank(
                    processed_query=processed_query,
                    results=fusion_results,
                    top_k=min(top_k, self.settings.rerank.top_k),
                )
                rerank_mode = "applied"
                rerank_provider = self.reranker.provider
            except Exception as exc:
                final_results = fusion_results
                rerank_mode = "fallback_to_fusion"
                rerank_provider = self.settings.rerank.provider
                trace.metadata["rerank_error"] = str(exc)
        trace.record_stage(
            "rerank",
            payload={
                "mode": rerank_mode,
                "provider": rerank_provider,
                "result_count": len(final_results),
                "top_chunk_id": final_results[0].chunk_id if final_results else None,
            },
            elapsed_ms=(perf_counter() - stage_t0) * 1000.0,
        )

        stage_t0 = perf_counter()
        formatted_response = self.formatter.format(
            query=query,
            collection=collection,
            results=final_results,
        )
        trace.record_stage(
            "format_response",
            payload={
                "summary": formatted_response.summary,
                "result_count": formatted_response.result_count,
            },
            elapsed_ms=(perf_counter() - stage_t0) * 1000.0,
        )

        trace.metadata["result_count"] = len(final_results)
        trace.metadata["top_chunk_id"] = final_results[0].chunk_id if final_results else None

        if self.settings.observability.trace_enabled:
            self.trace_collector.collect(trace)

        return QueryWorkflowResult(
            processed_query=processed_query,
            dense_results=dense_results,
            sparse_results=sparse_results,
            fusion_results=fusion_results,
            final_results=final_results,
            formatted_response=formatted_response,
            trace=trace,
        )
