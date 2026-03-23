"""Phase 1 的 Streamlit 占位页面。"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """渲染最小 Dashboard 页面。

    当前阶段先验证 Streamlit 启动链路，真正的六页结构后面再补。
    """
    st.set_page_config(page_title="Modular RAG Repro", page_icon="📚", layout="wide")
    st.title("Modular RAG Repro")
    st.info("Phase 1 骨架已经就绪，后续阶段再补完整 Dashboard 页面。")


if __name__ == "__main__":
    main()
