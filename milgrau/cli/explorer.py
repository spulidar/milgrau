"""Command-line entry point for the MILGRAU Streamlit explorer."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit app with Streamlit's CLI module."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parents[1] / "explorer" / "streamlit_app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(stcli.main())
