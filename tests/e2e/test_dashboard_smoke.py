"""Dashboard 六页冒烟测试。"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from helpers import REPRO_ROOT


def test_dashboard_pages_render_without_exception() -> None:
    """六页 Dashboard 都应能正常切换并渲染。"""
    app_path = REPRO_ROOT / "src" / "modular_rag_repro" / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=20)
    at.run()
    assert len(at.exception) == 0

    for page in [
        "Overview",
        "Data Browser",
        "Ingestion Manager",
        "Ingestion Traces",
        "Query Traces",
        "Evaluation Panel",
    ]:
        at.radio[0].set_value(page).run()
        assert len(at.exception) == 0
