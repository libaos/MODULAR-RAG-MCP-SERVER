"""查询工作流集成测试。"""

from __future__ import annotations

from pathlib import Path

from modular_rag_repro.query_engine import QueryWorkflow
from helpers import SAMPLE_PDF, cleanup_collection, seed_collection


def test_query_workflow_returns_dense_sparse_and_fusion_results(test_settings, unique_collection: str) -> None:
    """完整查询链路应返回稳定结果，并写出 query trace。"""
    cleanup_collection(test_settings, unique_collection)
    seed_collection(test_settings, unique_collection, SAMPLE_PDF)
    workflow = QueryWorkflow(test_settings)

    try:
        result = workflow.run(
            query="sample pdf",
            collection=unique_collection,
            top_k=3,
            no_rerank=True,
            source="integration-test",
        )

        assert result.formatted_response.result_count >= 1
        assert len(result.dense_results) >= 1
        assert len(result.sparse_results) >= 1
        assert len(result.fusion_results) >= 1
        assert result.final_results[0].metadata["source_path"].endswith("simple.pdf")
        assert result.trace.trace_type == "query"
        assert len(result.trace.stages) >= 5
    finally:
        cleanup_collection(test_settings, unique_collection)
