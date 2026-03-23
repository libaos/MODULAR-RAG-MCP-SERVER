"""复现版 Dashboard 应用入口。"""

from __future__ import annotations

import streamlit as st

from modular_rag_repro.dashboard.pages import PAGE_OPTIONS, render_page
from modular_rag_repro.dashboard.services import DashboardService


def main() -> None:
    """渲染最小六页 Dashboard。"""
    st.set_page_config(page_title="Modular RAG Repro Dashboard", page_icon="📚", layout="wide")
    service = DashboardService()

    st.sidebar.title("Modular RAG Repro")
    page_name = st.sidebar.radio("Pages", PAGE_OPTIONS, index=0)
    st.sidebar.caption(f"当前页面: {page_name}")
    st.sidebar.caption(f"默认 collection: {service.settings.vector_store.collection_name}")

    render_page(page_name, service)


if __name__ == "__main__":
    main()
