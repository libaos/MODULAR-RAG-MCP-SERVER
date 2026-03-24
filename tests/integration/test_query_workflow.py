"""查询工作流集成测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from modular_rag_repro.query_engine import QueryWorkflow
from helpers import SAMPLE_PDF, WITH_IMAGES_PDF, cleanup_collection, seed_collection, update_test_config


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


def test_query_workflow_returns_multimodal_metadata_for_image_docs(test_settings, unique_collection: str) -> None:
    """带图文档命中后，格式化结果应携带图片信息。"""
    cleanup_collection(test_settings, unique_collection)
    seed_collection(test_settings, unique_collection, WITH_IMAGES_PDF)
    workflow = QueryWorkflow(test_settings)

    try:
        result = workflow.run(
            query="image",
            collection=unique_collection,
            top_k=3,
            no_rerank=True,
            source="integration-test",
        )

        assert result.formatted_response.result_count >= 1
        assert result.formatted_response.items[0].image_count >= 1
        assert result.formatted_response.items[0].images[0]["file_path"]
        assert result.formatted_response.metadata["total_image_count"] >= 1
    finally:
        cleanup_collection(test_settings, unique_collection)


def test_query_workflow_generates_local_answer(temp_config_path: Path, unique_collection: str) -> None:
    """打开本地生成式回答后，工作流应产出 generated answer 和 trace。"""
    update_test_config(
        temp_config_path,
        lambda payload: payload.update(
            {
                "answer_generation": {
                    "enabled": True,
                    "provider": "local",
                    "model": "local-template",
                    "max_context_chunks": 2,
                    "timeout_seconds": 30,
                }
            }
        ),
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
            no_rerank=True,
            source="integration-test",
        )

        assert result.formatted_response.generated_answer
        assert result.formatted_response.answer_mode == "generated"
        answer_stage = next(stage for stage in result.trace.stages if stage.name == "answer_generation")
        assert answer_stage.payload["mode"] == "generated"
        assert answer_stage.payload["provider"] == "local"
    finally:
        cleanup_collection(settings, unique_collection)


def test_query_workflow_llm_rerank_falls_back_to_simple(
    temp_config_path: Path,
    unique_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llm rerank 失败时，工作流应退回 simple rerank。"""
    update_test_config(
        temp_config_path,
        lambda payload: payload["rerank"].update(
            {"enabled": True, "provider": "llm", "top_k": 3, "fallback_provider": "simple"}
        ),
    )
    from helpers import load_test_settings

    settings = load_test_settings(temp_config_path)
    cleanup_collection(settings, unique_collection)
    seed_collection(settings, unique_collection, SAMPLE_PDF)
    workflow = QueryWorkflow(settings)
    monkeypatch.setattr(workflow.reranker, "_rerank_with_llm", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("llm down")))

    try:
        result = workflow.run(
            query="sample pdf",
            collection=unique_collection,
            top_k=3,
            no_rerank=False,
            source="integration-test",
        )

        rerank_stage = next(stage for stage in result.trace.stages if stage.name == "rerank")
        assert rerank_stage.payload["mode"] == "fallback_to_simple"
        assert rerank_stage.payload["provider"] == "simple"
        assert result.final_results[0].metadata["rerank_fallback"] == "llm"
    finally:
        cleanup_collection(settings, unique_collection)
