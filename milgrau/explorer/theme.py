"""MILGRAU Hybrid Scientific design, embedded in the Explorer.

The Explorer intentionally owns its visual identity: installing or running it
does not mutate the user's global Streamlit theme or require config.toml.
"""
from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go
import streamlit as st


def escape(value: Any) -> str:
    """Protect user- and file-derived text displayed in HTML visual elements."""
    return html.escape(str(value), quote=True)


_BASE_CSS = """
<style>
:root {
    --mg-ink: #173447;
    --mg-muted: #577184;
    --mg-teal: #087f89;
    --mg-sidebar: #142d3d;
    --mg-border: #dce5ec;
    --mg-surface: #ffffff;
    --mg-bg: #f4f7fa;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--mg-bg) !important;
    color: var(--mg-ink) !important;
}
[data-testid="stHeader"] { background: rgba(244,247,250,.94) !important; }
[data-testid="stAppViewContainer"] .block-container {
    max-width: 1460px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] p,
[data-testid="stMain"] label, [data-testid="stMain"] span {
    color: var(--mg-ink);
}
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: var(--mg-muted) !important;
}
[data-testid="stSidebar"] {
    background: var(--mg-sidebar) !important;
    border-right: 1px solid #345465;
}
[data-testid="stSidebar"] > div:first-child {
    background: var(--mg-sidebar) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: #edf7fa !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #acc6d1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] input {
    background: #203d4f !important;
    color: #f3f9fb !important;
    border-color: #3d5b6a !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #1b3647;
    border: 1px solid #35576a;
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #edf7fa !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .26rem .35rem;
    border-radius: .45rem;
    margin-bottom: .08rem;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: transparent !important;
}
[data-testid="stMain"] [data-baseweb="select"] > div,
[data-testid="stMain"] [data-baseweb="input"] > div,
[data-testid="stMain"] input {
    background: #fff !important;
    color: var(--mg-ink) !important;
    border-color: var(--mg-border) !important;
}
[data-testid="stMain"] [data-testid="stMetric"] {
    background: #fff;
    border: 1px solid var(--mg-border);
    border-radius: 12px;
    padding: .85rem 1rem;
    min-height: 105px;
    box-shadow: 0 2px 8px rgba(14,42,60,.035);
}
[data-testid="stMain"] [data-testid="stMetricLabel"] p {
    color: var(--mg-muted) !important;
    font-size: .8rem !important;
}
[data-testid="stMain"] [data-testid="stMetricValue"] div {
    font-size: 1.55rem !important;
    font-weight: 650 !important;
    color: #173447 !important;
}
[data-testid="stMain"] [data-testid="stExpander"] {
    border-color: var(--mg-border);
    background: #fff;
    border-radius: 10px;
}
[data-testid="stMain"] [data-testid="stDataFrame"] {
    border: 1px solid var(--mg-border);
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stMain"] [data-testid="stButton"] button {
    border-radius: 9px;
    border-color: #b6ccd6;
}
[data-testid="stMain"] [data-testid="stButton"] button:hover {
    border-color: var(--mg-teal);
    color: var(--mg-teal);
}
.mg-brand {
    display: flex;
    align-items: center;
    gap: .65rem;
    font-size: 1.23rem;
    font-weight: 750;
    letter-spacing: .035em;
    color: #f2fbfd;
    padding: .15rem 0 .25rem;
}
.mg-brand-dot {
    display: inline-block;
    width: 12px; height: 12px;
    border: 3px solid #67c7cd;
    border-radius: 50%;
    box-shadow: 0 0 0 3px rgba(103,199,205,.14);
}
.mg-brand-small {
    color: #a8c7d1;
    font-size: .72rem;
    font-weight: 550;
    letter-spacing: .09em;
    margin-bottom: 1.25rem;
}
.mg-eyebrow {
    color: #0e8490;
    font-size: .72rem;
    font-weight: 750;
    letter-spacing: .11em;
    text-transform: uppercase;
    margin-bottom: .25rem;
}
.mg-title {
    color: #173447;
    font-size: clamp(1.7rem, 2.6vw, 2.5rem);
    font-weight: 750;
    letter-spacing: -.035em;
    line-height: 1.2;
    margin: .15rem 0 .4rem;
}
.mg-subtitle { color: #5b7181; font-size: .91rem; line-height: 1.65; }
.mg-hero {
    background: #fff;
    border: 1px solid var(--mg-border);
    border-left: 4px solid #158893;
    border-radius: 12px;
    padding: 1.2rem 1.45rem 1.25rem;
    margin: .45rem 0 1.25rem;
}
.mg-session { color: #19394c; font-size: 1.38rem; font-weight: 700; }
.mg-hero-meta { color: #607b8c; font-size: .82rem; margin-top: .4rem; }
.mg-pills { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .85rem; }
.mg-pill {
    display: inline-block;
    color: #17616b !important;
    background: #e7f4f5;
    border: 1px solid #bfe2e4;
    border-radius: 999px;
    padding: .19rem .63rem;
    font-size: .73rem;
    font-weight: 650;
}
.mg-pill.mg-muted {
    color: #657985 !important;
    background: #eef2f5;
    border-color: #e1e7eb;
}
.mg-section {
    margin: 1.45rem 0 .55rem;
    color: #193b4e;
    font-size: 1.04rem;
    font-weight: 725;
}
.mg-note { color: #687f8e; font-size: .79rem; line-height: 1.5; }
</style>
"""


_SITE_CSS = """
<style>
/* Compact, site-inspired horizontal layout. No sidebar or theme configuration. */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #edf1f5 !important;
}
[data-testid="stMain"] .block-container {
    max-width: 1550px !important;
    width: 100%;
    padding-top: .75rem !important;
    padding-left: 2.1rem !important;
    padding-right: 2.1rem !important;
}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.mg-site-brand) {
    background: #191c20;
    border-radius: 8px 8px 0 0;
    padding: .55rem 1rem;
    align-items: center;
    min-height: 70px;
}
.mg-site-brand {
    color: #f3f6f9 !important;
    letter-spacing: .045em;
    font-size: 1.18rem;
    font-weight: 700;
    white-space: nowrap;
}
.mg-site-divider { color: #648497; padding: 0 .55rem; }
.mg-site-subbrand { color: #43bde4; font-weight: 650; font-size: .92rem; }
.mg-site-location {
    color: #a8b7c0 !important;
    text-align: right;
    font-size: .72rem;
    letter-spacing: .015em;
    white-space: nowrap;
}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.mg-site-brand)
[data-testid="stCaptionContainer"] p {
    color: #9cc6d3 !important;
    font-size: .68rem;
    margin: 0;
}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.mg-site-brand)
[data-baseweb="select"] > div {
    color: #eff7fb !important;
    background: #303942 !important;
    border: 1px solid #445460 !important;
    min-height: 38px;
}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.mg-site-brand)
[data-baseweb="select"] * {
    color: #eff7fb !important;
}
[data-testid="stMain"] [data-testid="stRadio"]:has([role="radiogroup"]) {
    background: #fff;
    border-bottom: 1px solid #cdd8df;
    padding: .35rem .65rem;
}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] {
    flex-wrap: wrap;
    gap: .35rem !important;
}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label {
    border-radius: 5px;
    padding: .3rem .6rem;
    white-space: nowrap;
}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background: #e1f2f7;
    color: #006ea1 !important;
    box-shadow: inset 0 -2px #1172b6;
}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label p {
    font-size: .83rem !important;
    font-weight: 630;
}
.mg-site-session {
    display: flex;
    flex-wrap: wrap;
    column-gap: 1.2rem;
    row-gap: .3rem;
    padding: .7rem 1rem;
    background: #f7fafc;
    border: 1px solid #dce5eb;
    border-top: 0;
    margin-bottom: 1.1rem;
    color: #607788;
    font-size: .76rem;
}
.mg-site-session strong { color: #1b3e52; font-weight: 750; }
.mg-site-session span { overflow-wrap: anywhere; }
[data-testid="stMain"] [data-testid="stPlotlyChart"] {
    border: 1px solid #dce5eb;
    background: #ffffff;
    padding: .5rem;
    border-radius: 8px;
}
@media (max-width: 1100px) {
    .mg-site-brand { font-size: .96rem; white-space: normal; }
    .mg-site-subbrand { font-size: .8rem; }
    .mg-site-location { white-space: normal; }
    [data-testid="stMain"] .block-container {
        padding-left: .8rem !important;
        padding-right: .8rem !important;
    }
}
</style>
"""

_FOCUS_CSS = """
<style>
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stMain"] .block-container { max-width: 100%; padding-left: 2rem; padding-right: 2rem; }
</style>
"""


def apply_theme(*, focus: bool = False) -> None:
    """Apply Hybrid Scientific only inside the running Explorer."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    st.markdown(_SITE_CSS, unsafe_allow_html=True)
    if focus:
        st.markdown(_FOCUS_CSS, unsafe_allow_html=True)


def plot_scientific(fig: go.Figure, **kwargs: Any) -> None:
    """Match Plotly's plotting surface to the Explorer's light canvas."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#173447"},
        margin={"l": 55, "r": 24, "t": 80, "b": 55},
    )
    st.plotly_chart(fig, theme=None, **kwargs)


__all__ = ["apply_theme", "escape", "plot_scientific"]
