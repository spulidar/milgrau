"""MILGRAU Streamlit Explorer: one active scientific view, never a pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from milgrau.explorer.catalog import (
    LEVEL_TO_PATH,
    discover_products,
    processed_root_from_config,
    selected_session_context,
    station_timezone_from_config,
)
from milgrau.explorer.data import open_dataset
from milgrau.explorer.l2_products import render_l2_products
from milgrau.explorer.theme import apply_theme, escape
from milgrau.explorer.views import render_diagnostics, render_level, render_metadata
from milgrau.version import __version__


PAGES = (
    "Visão geral",
    "Sinais L0 / L1",
    "Level 2 · Produtos ópticos",
    "Level 2 · QA científico",
    "Arquivos e QA",
)


@st.cache_data(ttl=120, show_spinner="Indexando sessões...")
def cached_inventory(processed_root: str, timezone_name: str) -> pd.DataFrame:
    return discover_products(processed_root, timezone_name)


def _choose_session(inv: pd.DataFrame) -> dict:
    """Keep the single-session case frictionless; show filters only as needed."""
    st.markdown("##### Sessão científica")
    if len(inv) == 1:
        row = inv.iloc[0]
        st.caption(row["human_label"])
        return row.to_dict()

    with st.expander("Filtrar sessões", expanded=len(inv) > 25):
        search = st.text_input("Buscar session_id", placeholder="Nome ou data UTC")
        filtered = inv
        if search:
            filtered = filtered[
                filtered["session_id"].astype(str).str.contains(
                    search, case=False, regex=False, na=False
                )
            ]
        years = ["Todos"] + sorted(
            (int(v) for v in inv["year"].dropna().unique()), reverse=True
        )
        year = st.selectbox("Ano", years)
        if year != "Todos":
            filtered = filtered[filtered["year"] == year]
        months = ["Todos"] + sorted(
            (int(v) for v in filtered["month"].dropna().unique())
        )
        month = st.selectbox("Mês", months)
        if month != "Todos":
            filtered = filtered[filtered["month"] == month]
        levels = ["Todos", *LEVEL_TO_PATH]
        level = st.selectbox("Nível disponível", levels)
        if level != "Todos":
            filtered = filtered[filtered[LEVEL_TO_PATH[level]].astype(bool)]
    if filtered.empty:
        st.warning("Não há sessões para esses filtros.")
        st.stop()
    options = filtered.sort_values(["date", "session_id"], ascending=False).index.to_list()
    selected = st.selectbox(
        "Abrir sessão",
        options,
        format_func=lambda i: (
            f"{filtered.loc[i, 'human_label']} · "
            f"{filtered.loc[i, 'highest_available_level']}"
        ),
        key="mg_session",
    )
    st.caption(f"{len(filtered)} de {len(inv)} sessões")
    return filtered.loc[selected].to_dict()


def _choose_product(row: dict, level: str) -> dict:
    """Keep canonical and derived products explicit rather than guessing."""
    number = level.split()[-1]
    choices = row.get(f"level{number}_products", []) or []
    if not choices:
        return row
    updated = dict(row)
    if len(choices) > 1:
        path = st.selectbox(
            f"Produto {level} · canônico ou derivado",
            choices,
            format_func=lambda value: Path(value).name,
            key=f"mg_product_{number}_{row['session_id']}",
        )
    else:
        path = choices[0]
        st.caption(f"Produto {level}: {Path(path).name}")
    updated[f"level{number}_path"] = path
    return updated


def _figure_gallery(row: dict) -> None:
    images = [
        filename for filename in row.get("figures", [])
        if Path(filename).suffix.lower() in (".webp", ".png", ".jpg", ".jpeg")
    ]
    if not images:
        st.info("Nenhuma figura gerada pelo LIPANCORA/LEBEAR nesta sessão.")
        return
    st.caption(f"{len(images)} figuras prontas · carregamento sob demanda")
    chosen = st.selectbox("Figura", images, key="mg_gallery")
    path = Path(row["product_dir"]) / "figures" / chosen
    if path.is_file():
        st.image(str(path), caption=chosen, use_container_width=True)


def _go_to(page: str) -> None:
    st.session_state["mg_page"] = page


def _toggle_focus() -> None:
    st.session_state["mg_focus"] = not st.session_state.get("mg_focus", False)


def _header(row: dict) -> None:
    left, right = st.columns([5, 1])
    with left:
        st.markdown('<div class="mg-eyebrow">SPU LIDAR · WORKSPACE CIENTÍFICO</div>', unsafe_allow_html=True)
        st.markdown('<div class="mg-title">MILGRAU Explorer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="mg-subtitle">Observações atmosféricas · L0 LIBIDS / '
            'L1 LIPANCORA / L2 LEBEAR · Somente leitura</div>',
            unsafe_allow_html=True,
        )
    with right:
        focusing = bool(st.session_state.get("mg_focus", False))
        st.button(
            "Sair do foco" if focusing else "⛶ Modo foco",
            on_click=_toggle_focus,
            key="mg_toggle_focus",
            use_container_width=True,
        )
    st.markdown(
        f'<div class="mg-note">Sessão: {escape(row["session_id"])} · '
        f'MILGRAU {escape(__version__)}</div>',
        unsafe_allow_html=True,
    )


def _render_overview(row: dict, inv: pd.DataFrame) -> None:
    context_path = next(
        (row[key] for key in ("level2_path", "level1_path", "level0_path") if row.get(key)),
        "",
    )
    ctx = selected_session_context(context_path)
    pills = []
    for label, key in LEVEL_TO_PATH.items():
        state = "" if row.get(key) else " mg-muted"
        info = "✓ " if row.get(key) else "— "
        pills.append(f'<span class="mg-pill{state}">{info}{escape(label)}</span>')
    st.markdown(
        '<div class="mg-hero">'
        '<div class="mg-eyebrow">SESSÃO SELECIONADA</div>'
        f'<div class="mg-session">{escape(row["human_label"])}</div>'
        f'<div class="mg-hero-meta">UTC: {escape(row["interval_utc"])}</div>'
        '<div class="mg-pills">' + "".join(pills) + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cards = st.columns(4)
    cards[0].metric("Nível máximo", row["highest_available_level"])
    cards[1].metric("Duração", row["duration"])
    cards[2].metric("Regimes solares", ctx["solar_regimes"])
    cards[3].metric("Figuras disponíveis", int(row["figure_count"]))
    st.markdown('<div class="mg-section">Explorar esta sessão</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.button(
        "Produtos ópticos L2 →",
        on_click=_go_to,
        args=("Level 2 · Produtos ópticos",),
        disabled=not bool(row.get("level2_path")),
        use_container_width=True,
    )
    b.button(
        "Inspecionar sinais →",
        on_click=_go_to,
        args=("Sinais L0 / L1",),
        disabled=not bool(row.get("level0_path") or row.get("level1_path")),
        use_container_width=True,
    )
    c.button(
        "Figuras e arquivos →",
        on_click=_go_to,
        args=("Arquivos e QA",),
        use_container_width=True,
    )
    with st.expander("Arquivos da sessão", expanded=False):
        records = []
        for label in LEVEL_TO_PATH:
            for product in row.get(f"level{label[-1]}_products", []):
                file_path = Path(product)
                try:
                    mib = round(file_path.stat().st_size / (1024 ** 2), 1)
                except OSError:
                    mib = None
                records.append({
                    "Nível": label, "Produto": file_path.name, "Tamanho (MiB)": mib,
                    "Tipo": "Canônico" if file_path.name == (
                        f"{row['session_id']}_L{label[-1]}.nc"
                    ) else "Derivado",
                })
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    with st.expander("Identidade científica e segmentos"):
        st.caption(f"Local: {row['interval_local']}")
        st.caption(f"Segmentos: {ctx['segments']}")
        st.code(row["session_id"], language=None)
        st.caption(f"Pasta: {row['product_dir']}")
    with st.expander("Inventário completo", expanded=False):
        st.dataframe(
            inv[["human_label", "available_levels", "modified"]].tail(250),
            hide_index=True, use_container_width=True,
        )


def _render_signals(row: dict) -> None:
    available = [label for label in ("Level 0", "Level 1") if row.get(LEVEL_TO_PATH[label])]
    if not available:
        st.info("Nenhum produto L0 ou L1 disponível para esta sessão.")
        return
    st.markdown('<div class="mg-section">Sinais e perfis verticais</div>', unsafe_allow_html=True)
    level = st.radio("Nível", available, horizontal=True, key="mg_signals_level")
    render_level(_choose_product(row, level), level)


def _render_files(row: dict) -> None:
    st.markdown('<div class="mg-section">Arquivos, metadados e diagnósticos</div>', unsafe_allow_html=True)
    tab = st.radio(
        "Visualizar",
        ["Galeria de figuras", "Metadados NetCDF", "Indicadores QA"],
        horizontal=True,
        key="mg_files_mode",
    )
    if tab == "Galeria de figuras":
        _figure_gallery(row)
        return
    available = [label for label, key in LEVEL_TO_PATH.items() if row.get(key)]
    if not available:
        st.info("Nenhum NetCDF disponível.")
        return
    level = st.selectbox("Nível do produto", available, key="mg_files_level")
    selected = _choose_product(row, level)
    with open_dataset(selected[LEVEL_TO_PATH[level]]) as ds:
        if tab == "Metadados NetCDF":
            render_metadata(ds, key_prefix=f"{row['session_id']}_{level}")
        else:
            render_diagnostics(ds)


def main() -> None:
    st.set_page_config(
        page_title="MILGRAU Explorer",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme(focus=bool(st.session_state.get("mg_focus", False)))
    with st.sidebar:
        st.markdown(
            '<div class="mg-brand"><span class="mg-brand-dot"></span>MILGRAU</div>'
            '<div class="mg-brand-small">SPU LIDAR · EXPLORER</div>',
            unsafe_allow_html=True,
        )
        with st.expander("⚙ Fontes de dados", expanded=False):
            config_path = st.text_input(
                "config.yaml", value="config.yaml", key="mg_config"
            )
            try:
                default_root = processed_root_from_config(config_path)
                timezone_name = station_timezone_from_config(config_path)
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"Não foi possível carregar a configuração: {exc}")
                st.stop()
            processed_root = Path(
                st.text_input(
                    "processed_data", value=str(default_root),
                    key="mg_processed_root",
                )
            ).expanduser()
            if st.button("Reindexar sessões", use_container_width=True):
                cached_inventory.clear()
        inv = cached_inventory(str(processed_root), timezone_name)
        if inv.empty:
            st.info("Nenhum produto NetCDF localizado. Confira Fontes de dados.")
            st.stop()
        row = _choose_session(inv)
        st.divider()
        st.radio("Explorar", PAGES, key="mg_page")
        st.divider()
        st.caption(f"{len(inv)} sessão(ões) no catálogo · UTC como referência")

    _header(row)
    page = st.session_state["mg_page"]
    if page == "Visão geral":
        _render_overview(row, inv)
    elif page == "Sinais L0 / L1":
        _render_signals(row)
    elif page == "Level 2 · Produtos ópticos":
        st.markdown('<div class="mg-section">Produtos ópticos interativos · L2</div>', unsafe_allow_html=True)
        render_l2_products(_choose_product(row, "Level 2"))
    elif page == "Level 2 · QA científico":
        st.markdown('<div class="mg-section">Qualidade da recuperação · L2</div>', unsafe_allow_html=True)
        render_level(_choose_product(row, "Level 2"), "Level 2")
    else:
        _render_files(row)


if __name__ == "__main__":
    main()
