"""查询工作流集成测试。"""

from __future__ import annotations

from pathlib import Path

from modular_rag_repro.query_engine import QueryWorkflow
from helpers import SAMPLE_PDF, cleanup_collection, seed_collection, update_test_config


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


def test_query_workflow_applies_simple_rerank(temp_config_path: Path, unique_collection: str) -> None:
    """打开 simple rerank 后，工作流应产出 rerank 元数据。"""
    update_test_config(
        temp_config_path,
        lambda payload: payload["rerank"].update({"enabled": True, "provider": "simple", "top_k": 3}),
    )
    from helpers import load_test_settings

    settings = load_test_settings(temp_config_path)
    cleanup_collection(settings, unique_collection)
    seed_collection(settings, unique_collection, SAMPLE_PDF)
    workflow = QueryWorkflow(settings)

    try:
        result = workflow.run(
            query="sample pdf",
            collection=unique_collection,
            top_k=3,
            no_rerank=False,
            source="integration-test",
        )

        rerank_stage = next(stage for stage in result.trace.stages if stage.name == "rerank")
        assert rerank_stage.payload["mode"] == "applied"
        assert rerank_stage.payload["provider"] == "simple"
        assert result.final_results[0].metadata["reranked"] is True
        assert "rerank_score" in result.final_results[0].metadata
    finally:
        cleanup_collection(settings, unique_collection)
