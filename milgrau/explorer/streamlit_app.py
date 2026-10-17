"""MILGRAU Explorer: site-style UTC calendar, active-only scientific pages.

The application's only source is the processed-data root resolved in
config.yaml. Neither Streamlit nor the Explorer modifies scientific products.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from milgrau.explorer.catalog import (
    LEVEL_TO_PATH,
    discover_products,
    selected_session_context,
    source_context_from_config,
)
from milgrau.explorer.data import open_dataset
from milgrau.explorer.l2_products import render_l2_products
from milgrau.explorer.navigation import available_utc_dates, sessions_on_utc_date
from milgrau.explorer.theme import apply_theme, escape
from milgrau.explorer.views import (
    render_atmospheric_profiles,
    render_diagnostics,
    render_level0,
    render_level1,
    render_level2,
    render_metadata,
)
from milgrau.version import __version__

PAGES = (
    "RCS Maps",
    "Atmospheric Profiles",
    "L2 Products",
    "QA",
    "Metadata",
)


@st.cache_data(ttl=120, show_spinner="Localizando sessões NetCDF...")
def cached_inventory(processed_root: str, timezone_name: str) -> pd.DataFrame:
    return discover_products(processed_root, timezone_name)


def _selected_session(inv: pd.DataFrame, date_utc: Any) -> dict[str, Any]:
    """Choose among all sessions overlapping the date, without slicing them."""
    dated = sessions_on_utc_date(inv, date_utc)
    if dated.empty:
        st.info("Não há sessões científicas nesta data UTC.")
        st.stop()
    if len(dated) == 1:
        return dated.iloc[0].to_dict()
    indexes = dated.sort_values("session_id", ascending=False).index.to_list()
    selected_index = st.selectbox(
        "Sessão científica (UTC)",
        indexes,
        format_func=lambda index: dated.loc[index, "human_label"],
        key="mg_session_choice",
    )
    return dated.loc[selected_index].to_dict()


def _product_choice(row: dict[str, Any], level: str) -> dict[str, Any]:
    """Preserve canonical-vs-derived product choice as an explicit action."""
    digit = level[-1]
    options = row.get(f"level{digit}_products", []) or []
    if not options:
        return row
    updated = dict(row)
    if len(options) > 1:
        path = st.selectbox(
            f"Arquivo {level}",
            options,
            format_func=lambda file: Path(file).name,
            key=f"mg_product_{digit}_{row['session_id']}",
        )
    else:
        path = options[0]
    updated[f"level{digit}_path"] = path
    return updated


def _figure_gallery(row: dict[str, Any]) -> None:
    images = [
        name for name in row.get("figures", [])
        if Path(name).suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"}
    ]
    if not images:
        st.info("Nenhuma figura pré-renderizada disponível para esta sessão.")
        return
    choice = st.selectbox("Figura produzida", images, key="mg_figure")
    image_path = Path(row["product_dir"]) / "figures" / choice
    if image_path.is_file():
        st.image(str(image_path), caption=choice, use_container_width=True)


def _source_banner(source: dict[str, Any], date_utc: Any) -> None:
    """Station data comes from station.yaml through the configured loader."""
    left, date_col, right = st.columns([2.5, 1.1, 2.4], gap="medium")
    with left:
        st.markdown(
            '<div class="mg-site-brand">'
            f'{escape(source["station"])}'
            '<span class="mg-site-divider">|</span>'
            '<span class="mg-site-subbrand">MILGRAU Explorer</span></div>',
            unsafe_allow_html=True,
        )
    with date_col:
        st.caption("DATA UTC")
        st.markdown(
            f'<div class="mg-site-date">{escape(date_utc.strftime("%d %b %Y"))}</div>',
            unsafe_allow_html=True,
        )
    with right:
        location = []
        for key, label in (("latitude", "LAT"), ("longitude", "LON")):
            value = source.get(key)
            if value is not None:
                location.append(f"{label}: {float(value):.3f}°")
        elevation = source.get("elevation_m")
        if elevation is not None:
            location.append(f"ELEV: {float(elevation):g} m")
        location.append(f"MILGRAU {escape(__version__)}")
        st.markdown(
            '<div class="mg-site-location">' + " &nbsp; ".join(location) + "</div>",
            unsafe_allow_html=True,
        )


def _session_summary(row: dict[str, Any], total: int) -> None:
    levels = " · ".join(
        label for label, key in LEVEL_TO_PATH.items() if row.get(key)
    )
    st.markdown(
        '<div class="mg-site-session">'
        f'<strong>{escape(row["interval_utc"])}</strong>'
        f'<span>{escape(row["session_id"])}</span>'
        f'<span>{escape(levels)} · {total} sessão(ões) no catálogo</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_rcs(row: dict[str, Any]) -> None:
    available = [
        label for label in ("Level 1", "Level 0")
        if row.get(LEVEL_TO_PATH[label])
    ]
    if not available:
        st.info("Esta sessão não contém produtos L0 ou L1.")
        return
    level = (
        st.selectbox("Fonte do sinal", available, key="mg_signal_level")
        if len(available) > 1 else available[0]
    )
    selected = _product_choice(row, level)
    if level == "Level 1":
        render_level1(selected, mode="Quicklook RCS + perfil")
    else:
        render_level0(selected)


def _render_profiles(row: dict[str, Any]) -> None:
    if not row.get("level1_path"):
        st.info("Os perfis atmosféricos requerem um produto L1 desta sessão.")
        return
    render_atmospheric_profiles(_product_choice(row, "Level 1"))


def _render_qa(row: dict[str, Any]) -> None:
    if row.get("level2_path"):
        render_level2(_product_choice(row, "Level 2"))
        return
    st.info("QA de recuperação L2 indisponível. Mostrando diagnósticos existentes.")
    available = [level for level, path in LEVEL_TO_PATH.items() if row.get(path)]
    if available:
        level = st.selectbox("Nível", available)
        selected = _product_choice(row, level)
        with open_dataset(selected[LEVEL_TO_PATH[level]]) as ds:
            render_diagnostics(ds)


def _render_metadata(row: dict[str, Any], inv: pd.DataFrame) -> None:
    section = st.radio(
        "Informações",
        ["NetCDF", "Figuras prontas", "Sessão e arquivos"],
        horizontal=True, key="mg_metadata_section",
    )
    if section == "Figuras prontas":
        _figure_gallery(row)
        return
    if section == "Sessão e arquivos":
        reference = next(
            (row[key] for key in ("level2_path", "level1_path", "level0_path") if row.get(key)),
            "",
        )
        context = selected_session_context(reference)
        summary = st.columns(4)
        summary[0].metric("Nível máximo", row["highest_available_level"])
        summary[1].metric("Duração", row["duration"])
        summary[2].metric("Regimes solares", context["solar_regimes"])
        summary[3].metric("Figuras", row["figure_count"])
        st.caption(f"Intervalo local: {row['interval_local']}")
        st.caption(f"Segmentos: {context['segments']}")
        records = []
        for level in LEVEL_TO_PATH:
            for file in row.get(f"level{level[-1]}_products", []):
                p = Path(file)
                try:
                    size = round(p.stat().st_size / (1024 * 1024), 1)
                except OSError:
                    size = None
                records.append({
                    "Nível": level, "Nome": p.name, "MiB": size,
                    "Produto": "Canônico" if p.name == (
                        f"{row['session_id']}_L{level[-1]}.nc"
                    ) else "Derivado",
                })
        st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
        with st.expander("Inventário de outras sessões"):
            st.dataframe(
                inv[["human_label", "available_levels", "modified"]].tail(250),
                hide_index=True, use_container_width=True,
            )
        return
    available = [level for level, key in LEVEL_TO_PATH.items() if row.get(key)]
    if not available:
        st.info("Nenhum produto NetCDF nesta sessão.")
        return
    level = st.selectbox("Nível", available, key="mg_metadata_level")
    selected = _product_choice(row, level)
    with open_dataset(selected[LEVEL_TO_PATH[level]]) as ds:
        render_metadata(ds, key_prefix=f"{row['session_id']}_{level}")


def main() -> None:
    st.set_page_config(
        page_title="MILGRAU Explorer",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_theme()
    try:
        source = source_context_from_config("config.yaml")
    except (OSError, ValueError, KeyError) as exc:
        st.error(
            f"Não foi possível carregar config.yaml: {exc}. "
            "Execute o Explorer na pasta do projeto que contém o config.yaml."
        )
        st.stop()
    if st.button("↻ Atualizar catálogo", key="mg_refresh"):
        cached_inventory.clear()
    inv = cached_inventory(str(source["root"]), source["timezone"])
    if inv.empty:
        st.warning(
            "Não foram encontrados NetCDFs na pasta processed_data "
            f"configurada: {source['root']}"
        )
        st.stop()
    dates = available_utc_dates(inv)
    if not dates:
        st.warning("Nenhuma sessão possui identificador temporal UTC válido.")
        st.stop()

    header_left, header_day, header_right = st.columns([2.3, 1.4, 2], gap="medium")
    with header_left:
        st.markdown(
            f'<div class="mg-site-brand">{escape(source["station"])}'
            '<span class="mg-site-divider">|</span>'
            '<span class="mg-site-subbrand">MILGRAU Explorer</span></div>',
            unsafe_allow_html=True,
        )
    with header_day:
        selected_date = st.selectbox(
            "Data UTC", dates,
            format_func=lambda day: day.strftime("%d %b %Y"),
            key="mg_date",
        )
    with header_right:
        coords = []
        if source.get("latitude") is not None:
            coords.append(f"LAT: {float(source['latitude']):.3f}°")
        if source.get("longitude") is not None:
            coords.append(f"LON: {float(source['longitude']):.3f}°")
        if source.get("elevation_m") is not None:
            coords.append(f"ELEV: {float(source['elevation_m']):g} m")
        coords.append(f"v{escape(__version__)}")
        st.markdown(
            '<div class="mg-site-location">' + " &nbsp; ".join(coords) + "</div>",
            unsafe_allow_html=True,
        )
    row = _selected_session(inv, selected_date)
    st.radio(
        "Seções", PAGES, horizontal=True,
        key="mg_page", label_visibility="collapsed",
    )
    _session_summary(row, len(inv))
    page = st.session_state["mg_page"]
    if page == "RCS Maps":
        _render_rcs(row)
    elif page == "Atmospheric Profiles":
        _render_profiles(row)
    elif page == "L2 Products":
        render_l2_products(_product_choice(row, "Level 2"))
    elif page == "QA":
        _render_qa(row)
    else:
        _render_metadata(row, inv)


if __name__ == "__main__":
    main()
