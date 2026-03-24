"""Dashboard 页面渲染函数。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from modular_rag_repro.dashboard.services import DashboardService


PAGE_OPTIONS = [
    "Overview",
    "Data Browser",
    "Ingestion Manager",
    "Ingestion Traces",
    "Query Traces",
    "Evaluation Panel",
]


def render_page(page_name: str, service: DashboardService) -> None:
    """按名字渲染页面。"""
    if page_name == "Overview":
        render_overview(service)
    elif page_name == "Data Browser":
        render_data_browser(service)
    elif page_name == "Ingestion Manager":
        render_ingestion_manager(service)
    elif page_name == "Ingestion Traces":
        render_trace_page(service, trace_type="ingestion", title="Ingestion Traces")
    elif page_name == "Query Traces":
        render_trace_page(service, trace_type="query", title="Query Traces")
    elif page_name == "Evaluation Panel":
        render_evaluation_panel(service)
    else:
        st.error(f"未知页面: {page_name}")


def render_overview(service: DashboardService) -> None:
    """渲染概览页。"""
    overview = service.get_overview()
    st.title("Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Collections", overview["total_collections"])
    col2.metric("Documents", overview["total_documents"])
    col3.metric("Chroma Vectors", overview["total_chroma_vectors"])
    col4.metric("BM25 Docs", overview["total_bm25_docs"])
    col5.metric("Trace Records", overview["trace_count"])

    st.subheader("Current Config")
    settings = service.settings
    st.json(
        {
            "llm": {"provider": settings.llm.provider, "model": settings.llm.model},
            "answer_generation": {
                "enabled": settings.answer_generation.enabled,
                "provider": settings.answer_generation.provider,
                "model": settings.answer_generation.model,
            },
            "vision_llm": {
                "enabled": settings.vision_llm.enabled,
                "provider": settings.vision_llm.provider,
                "model": settings.vision_llm.model,
            },
            "rerank": {
                "enabled": settings.rerank.enabled,
                "provider": settings.rerank.provider,
                "fallback_provider": settings.rerank.fallback_provider,
            },
            "embedding": {"provider": settings.embedding.provider, "model": settings.embedding.model},
            "vector_store": {
                "provider": settings.vector_store.provider,
                "persist_directory": settings.vector_store.persist_directory,
            },
            "retrieval": {
                "dense_top_k": settings.retrieval.dense_top_k,
                "sparse_top_k": settings.retrieval.sparse_top_k,
                "fusion_top_k": settings.retrieval.fusion_top_k,
                "rrf_k": settings.retrieval.rrf_k,
            },
        }
    )

    st.subheader("Collections")
    collections = service.list_collections()
    if collections:
        st.dataframe(collections, width="stretch")
    else:
        st.info("当前还没有 collection。")


def render_data_browser(service: DashboardService) -> None:
    """渲染数据浏览页。"""
    st.title("Data Browser")
    collections = service.list_collections()
    collection_names = ["(all)"] + [item["name"] for item in collections]
    selected = st.selectbox("Collection", collection_names, index=0)
    selected_collection = None if selected == "(all)" else selected

    documents = service.list_documents(selected_collection)
    st.caption(f"当前展示 {len(documents)} 个文档")
    if not documents:
        st.info("没有可展示的文档。")
        return

    st.dataframe(documents, width="stretch")

    with st.expander("查看单个文档摘要", expanded=False):
        doc_ids = [item["doc_id"] for item in documents]
        selected_doc = st.selectbox("Document ID", doc_ids)
        if selected_doc:
            doc = next(item for item in documents if item["doc_id"] == selected_doc)
            st.markdown(f"**标题**: {doc['title']}")
            st.markdown(f"**来源**: `{doc['source_path']}`")
            st.markdown(f"**图片数**: {doc.get('image_count', 0)}")
            st.markdown(f"**摘要**: {doc['summary']}")
            images = service.get_document_images(doc["doc_id"], doc["collection"])
            if images:
                st.markdown("**图片预览**")
                preview_paths = [item["file_path"] for item in images[:3] if item.get("file_path")]
                if preview_paths:
                    st.image(preview_paths, width=220)
                st.caption(f"共 {len(images)} 张图片")
                st.dataframe(
                    [
                        {
                            "image_id": item.get("image_id"),
                            "page": item.get("page"),
                            "index": item.get("image_index"),
                            "path": item.get("file_path"),
                        }
                        for item in images
                    ],
                    width="stretch",
                )

    with st.expander("删除文档", expanded=False):
        doc_labels = [f"{item['title']} ({item['collection']})" for item in documents]
        selected_label = st.selectbox("选择要删除的文档", doc_labels, key="delete_doc_label")
        selected_index = doc_labels.index(selected_label)
        selected_doc = documents[selected_index]

        st.caption("删除会同步清理当前文档在 Chroma 和 BM25 中的数据。")
        if st.button("Delete Document", type="secondary"):
            result = service.delete_document(
                doc_id=selected_doc["doc_id"],
                collection=selected_doc["collection"],
            )
            st.session_state["last_delete_result"] = result
            st.rerun()

    delete_result = st.session_state.get("last_delete_result")
    if delete_result:
        st.subheader("Last Delete Result")
        st.json(delete_result)


def render_ingestion_manager(service: DashboardService) -> None:
    """渲染摄取管理页。"""
    st.title("Ingestion Manager")
    st.caption("上传 PDF 后直接调用复现版摄取链路。")

    default_collection = service.settings.vector_store.collection_name
    collection = st.text_input("Target Collection", value=default_collection)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Run Ingestion", type="primary", disabled=uploaded_file is None):
        if uploaded_file is None:
            st.warning("请先选择一个 PDF 文件。")
        else:
            with st.spinner("正在执行摄取..."):
                result = service.ingest_uploaded_file(
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    collection=collection,
                )
            st.session_state["last_ingestion_result"] = result

    result = st.session_state.get("last_ingestion_result")
    if result:
        st.subheader("Last Ingestion Result")
        if result["success"]:
            st.success("摄取成功")
        else:
            st.error("摄取失败")
        st.json(result)


def render_trace_page(service: DashboardService, trace_type: str, title: str) -> None:
    """渲染 trace 页面。"""
    st.title(title)
    traces = service.read_traces_by_type(trace_type)
    if not traces:
        st.info(f"当前没有 `{trace_type}` trace 数据。Phase 6 接入后这里会显示真实链路记录。")
        return

    st.caption(f"共 {len(traces)} 条 trace")
    st.dataframe(traces, width="stretch")
    st.subheader("Latest Trace")
    st.json(traces[-1])
    latest_stages = traces[-1].get("stages", [])
    if latest_stages:
        st.subheader("Latest Trace Stages")
        st.dataframe(latest_stages, width="stretch")


def render_evaluation_panel(service: DashboardService) -> None:
    """渲染评估面板。"""
    st.title("Evaluation Panel")
    reports = service.list_evaluation_reports()
    if not reports:
        st.info("当前没有评估报告。Phase 6 接入后这里会展示 Golden Set 与聚合指标。")
        return

    st.caption(f"共 {len(reports)} 份评估报告")
    st.dataframe(reports, width="stretch")
