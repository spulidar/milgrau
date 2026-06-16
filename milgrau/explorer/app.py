"""Compatibility wrapper for the runnable MILGRAU NetCDF Streamlit app."""

from __future__ import annotations

from milgrau.explorer.streamlit_app import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
