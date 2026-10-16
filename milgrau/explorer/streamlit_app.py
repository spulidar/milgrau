"""MILGRAU Streamlit Explorer: read-only, one active page at a time.

Scientific retrieval remains exclusively in LIBIDS, LIPANCORA and LEBEAR.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from milgrau.explorer.catalog import (
    LEVEL_TO_PATH, discover_products, processed_root_from_config,
    selected_session_context, station_timezone_from_config,
)
from milgrau.explorer.views import render_diagnostics, render_level, render_metadata
from milgrau.explorer.l2_products import render_l2_products


@st.cache_data(ttl=120, show_spinner="Indexando sessões...")
def cached_inventory(processed_root: str, timezone_name: str) -> pd.DataFrame:
    return discover_products(processed_root, timezone_name)


def _choose_session(inv: pd.DataFrame) -> dict:
    st.subheader("Sessões")
    year_options = sorted(int(v) for v in inv["year"].dropna().unique())
    year = st.selectbox("Ano", year_options, index=len(year_options)-1)
    filtered = inv[inv["year"] == year]
    month_options = sorted(int(v) for v in filtered["month"].dropna().unique())
    month = st.selectbox("Mês", month_options, index=len(month_options)-1)
    filtered = filtered[filtered["month"] == month]
    search = st.text_input("Buscar sessão", placeholder="session_id")
    if search:
        filtered = filtered[filtered["session_id"].str.contains(search, case=False, regex=False)]
    if filtered.empty:
        st.warning("Nenhuma sessão encontrada para estes filtros.")
        st.stop()
    choices = filtered.sort_values(["date", "session_id"]).index.to_list()
    index = st.selectbox(
        "Sessão", choices, index=len(choices)-1,
        format_func=lambda i: filtered.loc[i, "human_label"],
    )
    st.caption(f"{len(filtered)} sessões no filtro")
    return filtered.loc[index].to_dict()


def _select_product(row: dict, level: str) -> dict:
    number = level.split()[-1]
    choices = row.get(f"level{number}_products", []) or []
    if len(choices) < 2:
        return row
    updated = dict(row)
    chosen = st.selectbox(
        f"Produto {level}", choices,
        format_func=lambda value: Path(value).name,
        key=f"product_{level}_{row['session_id']}",
    )
    updated[f"level{number}_path"] = chosen
    return updated


def _figure_gallery(row: dict) -> None:
    names = [
        name for name in row.get("figures", [])
        if Path(name).suffix.lower() in (".webp", ".png", ".jpg", ".jpeg")
    ]
    st.subheader("Figuras já produzidas")
    if not names:
        st.info("Nenhuma figura de L1/L2 encontrada nesta pasta.")
        return
    chosen = st.selectbox("Figura", names)
    path = Path(row["product_dir"]) / "figures" / chosen
    if path.is_file():
        st.image(str(path), caption=chosen, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="MILGRAU Explorer", layout="wide")
    st.title("MILGRAU Explorer")
    st.caption("Visualização científica e QA, somente leitura — sessões contínuas em UTC.")
    with st.sidebar:
        st.header("Fonte de dados")
        config_path = st.text_input("config.yaml", value="config.yaml")
        try:
            default_root = processed_root_from_config(config_path)
            timezone_name = station_timezone_from_config(config_path)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Falha ao carregar a configuração: {exc}")
            st.stop()
        processed_root = Path(st.text_input("processed_data", value=str(default_root))).expanduser()
        if st.button("Reindexar sessões"):
            cached_inventory.clear()
        inv = cached_inventory(str(processed_root), timezone_name)
        st.metric("Sessões localizadas", len(inv))
        if inv.empty:
            st.info("Nenhum NetCDF encontrado na estrutura de sessões.")
            st.stop()
        row = _choose_session(inv)
        st.divider()
        page = st.radio(
            "Explorar",
            ["Visão geral", "Level 0 · Sinais", "Level 1 · Sinais",
             "Level 2 · Produtos ópticos", "Level 2 · QA científico",
             "Metadados", "QA · Indicadores", "Galeria de figuras"],
        )
        if page.startswith("Level 0"):
            row = _select_product(row, "Level 0")
        elif page.startswith("Level 1"):
            row = _select_product(row, "Level 1")
        elif page.startswith("Level 2"):
            row = _select_product(row, "Level 2")
        elif page in ("Metadados", "QA · Indicadores"):
            active_levels = [label for label, key in LEVEL_TO_PATH.items() if row.get(key)]
            level = st.selectbox("Nível do produto", active_levels)
            row = _select_product(row, level)

    if page == "Visão geral":
        st.subheader(row["human_label"])
        ctx = selected_session_context(next(
            (row[key] for key in ("level2_path", "level1_path", "level0_path") if row.get(key)),
            "",
        ))
        metrics = st.columns(4)
        metrics[0].metric("Maior nível disponível", row["highest_available_level"])
        metrics[1].metric("Duração", row["duration"])
        metrics[2].metric("Regimes solares", ctx["solar_regimes"])
        metrics[3].metric("Figuras", int(row["figure_count"]))
        st.caption(f"UTC: {row['interval_utc']} · Local: {row['interval_local']}")
        st.caption(f"Segmentos: {ctx['segments']}")
        st.code(row["session_id"], language=None)
        records = []
        for label, key in LEVEL_TO_PATH.items():
            for product in row.get(f"level{label[-1]}_products", []):
                p = Path(product)
                records.append({"Nível": label, "Produto": p.name, "Tamanho (MiB)": round(p.stat().st_size / (1024**2), 1)})
        st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
        if st.checkbox("Mostrar resumo do catálogo completo"):
            st.dataframe(inv[["human_label", "available_levels", "modified"]].tail(250),
                         hide_index=True, use_container_width=True)
    elif page == "Level 0 · Sinais":
        render_level(row, "Level 0")
    elif page == "Level 1 · Sinais":
        render_level(row, "Level 1")
    elif page == "Level 2 · Produtos ópticos":
        render_l2_products(row)
    elif page == "Level 2 · QA científico":
        render_level(row, "Level 2")
    elif page in ("Metadados", "QA · Indicadores"):
        key = LEVEL_TO_PATH[level]
        with __import__("milgrau.explorer.data", fromlist=["open_dataset"]).open_dataset(row[key]) as ds:
            if page == "Metadados":
                render_metadata(ds, key_prefix=f"{row['session_id']}_{level}")
            else:
                render_diagnostics(ds)
    else:
        _figure_gallery(row)


if __name__ == "__main__":
    main()
