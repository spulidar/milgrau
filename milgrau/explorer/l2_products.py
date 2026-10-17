"""Interactive optical products from existing scientific Level 2 NetCDF.

All views are display-only: never rerun retrieval, and preserve NaNs and
uncertainty provenance. Load only selected wavelengths, times and altitudes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from milgrau.explorer.data import open_dataset
from milgrau.explorer.theme import plot_scientific
from milgrau.explorer.science import (
    OPTICAL_FIELDS, OPTICAL_SCALE, bounded_view, optical_field,
    scenario_options, support_mask, display_block_mask, time_label, wavelength_options,
)


def _matrix(da: xr.DataArray) -> np.ndarray:
    """Read only a bounded time-altitude view."""
    return np.asarray(da.transpose("block_time", "altitude").values, dtype=float)


def _heatmap(
    ds: xr.Dataset, quantity: str, wavelength: int, source: str,
    scenario: float, start: int, stop: int, top_km: float,
    regimes: list[str] | None = None, segments: list[str] | None = None,
) -> go.Figure:
    da = optical_field(ds, quantity, source, wavelength, scenario)
    view = bounded_view(da, start=start, stop=stop, top_km=top_km)
    values = _matrix(view) * OPTICAL_SCALE
    values[~display_block_mask(ds, view["block_time"].values, regimes, segments), :] = np.nan
    if source == "mc":
        valid = support_mask(ds, quantity, wavelength, scenario)
        mask = bounded_view(valid, start=start, stop=stop, top_km=top_km)
        values[~_matrix(mask).astype(bool)] = np.nan
    fig = go.Figure(go.Heatmap(
        x=np.asarray(view["block_time"].values),
        y=np.asarray(view["altitude"].values, dtype=float) / 1000.0,
        z=values.T, colorscale="Viridis",
        colorbar={"title": OPTICAL_FIELDS[quantity]["unit"]},
        hovertemplate="%{x}<br>%{y:.2f} km<br>%{z:.3g}<extra></extra>",
        connectgaps=False,
    ))
    fig.update_layout(
        title=f"{quantity} de aerossóis — {wavelength} nm · {source.upper()}",
        xaxis_title="Bloco UTC", yaxis_title="Altitude acima da estação (km)",
        height=620, margin={"l": 60, "r": 25, "t": 80, "b": 60},
    )
    return fig


def _profile(
    ds: xr.Dataset, quantity: str, wavelength: int, block: int,
    scenario: float, top_km: float,
) -> go.Figure:
    fig = go.Figure()
    specs = OPTICAL_FIELDS[quantity]
    base = bounded_view(
        optical_field(ds, quantity, "nominal", wavelength).isel(
            block_time=slice(block, block+1)
        ),
        top_km=top_km,
    ).isel(block_time=0)
    altitude = np.asarray(base["altitude"].values, dtype=float) / 1000.0
    nominal = np.asarray(base.values, dtype=float) * OPTICAL_SCALE
    fig.add_trace(go.Scatter(
        x=nominal, y=altitude, mode="lines", name="Recuperação nominal",
        line={"color": "#156083", "width": 2.5}, connectgaps=False,
    ))
    if all(name in ds for name in (specs["mc"], specs["lower"], specs["upper"])):
        values = {}
        for field in ("mc", "lower", "upper"):
            da = bounded_view(
                optical_field(ds, quantity, field, wavelength, scenario).isel(
                    block_time=slice(block, block+1)
                ),
                top_km=top_km,
            ).isel(block_time=0)
            values[field] = np.asarray(da.values, dtype=float) * OPTICAL_SCALE
            values[field][~np.isfinite(nominal)] = np.nan
        fig.add_trace(go.Scatter(
            x=values["lower"], y=altitude, mode="lines", name="MC P2.5",
            line={"width": 0}, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=values["upper"], y=altitude, mode="lines", name="MC P97.5",
            fill="tonextx", fillcolor="rgba(30,130,170,0.2)",
            line={"width": 0}, connectgaps=False,
        ))
        fig.add_trace(go.Scatter(
            x=values["mc"], y=altitude, mode="lines", name="Média MC",
            line={"color": "#bc6731", "width": 2}, connectgaps=False,
        ))
    fig.update_layout(
        title=f"{quantity} — {wavelength} nm · {time_label(ds['block_time'].values[block])}",
        xaxis_title=f"{quantity} [{specs['unit']}]",
        yaxis_title="Altitude acima da estação (km)", height=660,
    )
    return fig


def _spectral(
    ds: xr.Dataset, quantity: str, waves: list[int],
    start: int, stop: int, top_km: float,
    regimes: list[str] | None = None, segments: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    for wavelength in waves:
        da = bounded_view(
            optical_field(ds, quantity, "nominal", wavelength),
            start=start, stop=stop, top_km=top_km,
        )
        keep = display_block_mask(ds, da["block_time"].values, regimes, segments)
        da = da.where(xr.DataArray(keep, dims=("block_time",), coords={"block_time": da["block_time"]}))
        profile = da.mean(dim="block_time", skipna=True)
        fig.add_trace(go.Scatter(
            x=np.asarray(profile.values, dtype=float) * OPTICAL_SCALE,
            y=np.asarray(profile["altitude"].values, dtype=float) / 1000.0,
            mode="lines", name=f"{wavelength} nm", connectgaps=False,
        ))
    fig.update_layout(
        title=f"Perfis médios nominais por comprimento de onda — {quantity}",
        xaxis_title=f"{quantity} [{OPTICAL_FIELDS[quantity]['unit']}]",
        yaxis_title="Altitude acima da estação (km)", height=660,
    )
    return fig


def _time_series(
    ds: xr.Dataset, quantity: str, wavelength: int,
    altitudes: list[float], start: int, stop: int,
    regimes: list[str] | None = None, segments: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    for altitude in altitudes:
        da = bounded_view(
            optical_field(ds, quantity, "nominal", wavelength),
            start=start, stop=stop, top_km=max(altitudes) + 1.0,
            max_altitude=2000, max_time=600,
        )
        nearest = da.sel(altitude=altitude * 1000.0, method="nearest")
        actual = float(nearest["altitude"].values) / 1000.0
        vals = np.asarray(nearest.values, dtype=float) * OPTICAL_SCALE
        vals[~display_block_mask(ds, nearest["block_time"].values, regimes, segments)] = np.nan
        fig.add_trace(go.Scatter(
            x=np.asarray(nearest["block_time"].values),
            y=vals,
            mode="lines+markers", name=f"{actual:.2f} km",
            connectgaps=False,
        ))
    fig.update_layout(
        title=f"Evolução temporal — {quantity}, {wavelength} nm",
        xaxis_title="Bloco UTC",
        yaxis_title=f"{quantity} [{OPTICAL_FIELDS[quantity]['unit']}]",
        height=570,
    )
    return fig


def render_l2_products(row: dict) -> None:
    """Explore the saved L2 optical fields without recomputing retrievals."""
    path = row.get("level2_path", "")
    if not path or not Path(path).is_file():
        st.info("Nenhum produto L2 disponível nesta sessão.")
        return
    st.caption(f"Fonte: {Path(path).name} · leitura apenas")
    with open_dataset(path) as ds:
        wavelengths = wavelength_options(ds)
        n = int(ds.sizes.get("block_time", 0))
        if not wavelengths or n == 0:
            st.warning("Este L2 não contém comprimentos de onda e blocos científicos.")
            return
        mode = st.radio(
            "Visualização",
            ["Evolução tempo × altitude", "Perfil de um bloco",
             "Comparação espectral", "Série temporal por altitude"],
            horizontal=True,
            key="mg_l2_view",
        )
        c1, c2 = st.columns([1.5, 1])
        quantity = c1.selectbox(
            "Propriedade óptica", list(OPTICAL_FIELDS), key="l2_science_qty"
        )
        wavelength = c2.selectbox(
            "Comprimento de onda (nm)", wavelengths, key="l2_science_wave"
        )
        # Show the entire scientific altitude range; Plotly handles zoom/pan.
        # Bounded display sampling still limits browser-side matrix size.
        top_km = float(np.nanmax(ds["altitude"].values)) / 1000.0

        # Selectors inside an expander are still available to Streamlit's
        # state engine, but they occupy no space unless the user needs them.
        with st.expander("Período, regime solar e segmentos", expanded=False):
            regime_options = (
                sorted(set(ds["solar_regime"].values.astype(str)))
                if "solar_regime" in ds else []
            )
            segment_options = (
                list(dict.fromkeys(ds["segment_id"].values.astype(str)))
                if "segment_id" in ds else []
            )
            c4, c5 = st.columns(2)
            regimes = c4.multiselect(
                "Regime solar", regime_options, default=regime_options,
                key="mg_l2_regimes",
            )
            segments = c5.multiselect(
                "Segmentos", segment_options, default=segment_options,
                key="mg_l2_segments",
            )
            index_interval = st.slider(
                "Intervalo de blocos UTC",
                min_value=0, max_value=n - 1,
                value=(0, n - 1), key="mg_l2_time",
            ) if n > 1 else (0, 0)
            st.caption(
                f"{time_label(ds['block_time'].values[index_interval[0]])} → "
                f"{time_label(ds['block_time'].values[index_interval[1]])}"
            )
        first, last = index_interval
        eligible = display_block_mask(ds, ds["block_time"].values, regimes, segments)
        indices = [i for i in range(first, last + 1) if bool(eligible[i])]
        if not indices:
            st.info("Nenhum bloco encontrado no intervalo e filtros selecionados.")
            return
        scenarios = scenario_options(ds)
        if mode == "Evolução tempo × altitude":
            source = st.radio(
                "Estimativa", ["nominal", "mc"], horizontal=True,
                format_func=lambda v: "Nominal" if v == "nominal" else "Média Monte Carlo",
                key="mg_l2_estimate",
            )
            scenario = (
                st.selectbox("Fração residual (MC)", scenarios, key="mg_l2_fraction_map")
                if source == "mc" else 0.0
            )
            fig = _heatmap(
                ds, quantity, wavelength, source, scenario,
                first, last + 1, top_km, regimes, segments,
            )
        elif mode == "Perfil de um bloco":
            block = st.selectbox(
                "Bloco UTC", indices,
                format_func=lambda i: time_label(ds["block_time"].values[i]),
                key="mg_l2_block",
            )
            scenario = st.selectbox(
                "Fração residual (MC)", scenarios, key="mg_l2_fraction_profile"
            )
            fig = _profile(ds, quantity, wavelength, block, scenario, top_km)
            st.caption(
                "P2,5–P97,5 descrevem a distribuição Monte Carlo disponível, "
                "não necessariamente a incerteza científica total."
            )
        elif mode == "Comparação espectral":
            waves = st.multiselect(
                "Comprimentos de onda", wavelengths, default=wavelengths,
                key="mg_l2_spectral_waves",
            )
            if not waves:
                st.info("Selecione pelo menos um comprimento de onda.")
                return
            fig = _spectral(
                ds, quantity, waves, first, last + 1, top_km, regimes, segments
            )
            st.caption("Médias para visualização calculadas somente em blocos suportados.")
        else:
            candidates = [
                round(i * 0.5, 1)
                for i in range(1, int(np.floor(top_km / 0.5)) + 1)
            ]
            altitudes = st.multiselect(
                "Altitudes (km)", candidates,
                default=[v for v in (1.0, 3.0, 5.0) if v in candidates],
                key="mg_l2_altitudes",
            )
            if not altitudes:
                st.info("Escolha uma ou mais altitudes.")
                return
            fig = _time_series(
                ds, quantity, wavelength, altitudes, first, last + 1,
                regimes, segments,
            )
        plot_scientific(fig, use_container_width=True)
        st.caption(
            "Regiões sem suporte permanecem vazias (NaN). Os controles nunca "
            "alteram os dados científicos do NetCDF."
        )
