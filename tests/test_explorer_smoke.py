"""Smoke-test the real optional Streamlit Explorer interface."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

from milgrau.explorer import streamlit_app


def test_streamlit_explorer_imports_and_starts_without_products() -> None:
    app_path = Path(streamlit_app.__file__)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception, [str(exc.message) for exc in app.exception]
