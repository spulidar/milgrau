"""On-demand Streamlit viewers for each scientific level and metadata."""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from milgrau.explorer.catalog import LEVEL_TO_PATH, safe_key
from milgrau.explorer.theme import plot_scientific
from milgrau.explorer.level2 import available_level2_wavelengths, level2_status_summary
from milgrau.explorer.data import (
    open_dataset,
    channel_name,
    coord_values,
    wavelength_values,
    numeric_variables,
    wavelength_label,
    reduce_for_heatmap,
    vertical_values,
    altitude_km,
)
from milgrau.explorer.plots_signals import (
    heatmap_figure,
    profile_figure,
    level1_quicklook_figure,
    global_mean_rcs_figure,
)
from milgrau.explorer.plots_l2qa import (
    qa_gluing_figure,
    qa_molecular_figure,
    qa_scattering_ratio_figure,
    first_available,
    qa_kfs_figure,
)

DIAGNOSTIC_NAMES = [
    "PBL_Height_km",
    "pc_saturation_fraction",
    "deadtime_clipping_fraction",
    "bin_shift_invalid_fraction",
    "channel_correction_success",
    "dark_current_used",
    "Background_Profile_Available",
    "Raw_Data_Range_Resolution",
    "Background_Low",
    "Background_High",
    "gluing_success_flag",
    "gluing_attempted_flag",
    "single_channel_fallback_flag",
    "signal_source_flag",
    "retrieval_input_valid_flag",
    "retrieval_input_invalid_reason",
    "retrieval_input_snr_median",
    "gluing_split_altitude_m",
    "gluing_start_altitude_m",
    "gluing_stop_altitude_m",
    "gluing_slope",
    "gluing_intercept",
    "gluing_correlation",
    "gluing_relative_rmse",
    "gluing_relative_bias",
    "rayleigh_reference_success_flag",
    "rayleigh_reference_altitude_m",
    "rayleigh_reference_valid_fraction",
    "rayleigh_calibration_factor",
    "rayleigh_calibration_intercept",
    "retrieval_success_flag",
    "lidar_ratio_assumed_sr",
    "kfs_branch",
]

def render_level0(row: dict[str, Any]) -> None:
    path = row.get("level0_path", "")
    if not path or not Path(path).exists():
        st.info("Level 0 não disponível para esta medida.")
        return
    st.caption(f"Arquivo: `{path}`")
    with open_dataset(path) as ds:
        if "channel_string" in ds and "channels" in ds.dims:
            ds = ds.assign_coords(channels=ds["channel_string"].values.astype(str))
        variables = [name for name in ("Raw_Lidar_Data", "Background_Profile") if name in ds]
        variables += [name for name in numeric_variables(ds) if name not in variables]
        if not variables:
            st.info("Não encontrei variáveis numéricas plottáveis no Level 0.")
            return
        prefix = safe_key(row.get("session_id"), "level0")
        c1, c2, c3 = st.columns([1.8, 1, 0.7])
        variable = c1.selectbox("Variável", variables, key=f"{prefix}_var")
        channels = coord_values(ds, channel_name(ds))
        channel = c2.selectbox("Canal", channels, key=f"{prefix}_channel") if channels else None
        max_bin = max(1, int(ds.sizes.get("points", ds[variable].shape[-1])) - 1)
        log10 = c3.checkbox("log10", value=False, key=f"{prefix}_log")
        c5, c6, c7 = st.columns([1, 1, 2])
        robust = c5.checkbox("robust color", value=True, key=f"{prefix}_robust")
        percentile_span = c6.slider("percentis", 0.0, 100.0, (2.0, 98.0), step=0.5, key=f"{prefix}_pct")
        color_range = c7.text_input("Range de cor `min,max`", value="", key=f"{prefix}_range")
        view = reduce_for_heatmap(ds[variable], "Level 0", channel, None, float(max_bin))
        title = f"Level 0 · {variable}" + (f" · {channel}" if channel is not None else "")
        plot_scientific(heatmap_figure(view, title, "Level 0", log10, color_range, robust, percentile_span[0], percentile_span[1], ds=ds), use_container_width=True)
        with st.expander("Perfil médio em bins"):
            plot_scientific(profile_figure(view, f"Mean profile · {variable}", "Level 0", smooth_bins=1), use_container_width=True)
        with st.expander("Tempo bruto dos perfis"):
            if "Raw_Data_Start_Time" in ds:
                try:
                    values = ds["Raw_Data_Start_Time"].values
                    if np.asarray(values).ndim == 2:
                        values = np.asarray(values)[:, 0]
                    if np.issubdtype(values.dtype, np.datetime64):
                        times = pd.to_datetime(values)
                    else:
                        raw_date = str(ds.attrs.get("RawData_Start_Date", ""))
                        raw_time = str(ds.attrs.get("RawData_Start_Time_UT", ""))
                        if len(raw_date) == 8 and len(raw_time) == 6:
                            reference = pd.Timestamp(f"{raw_date}{raw_time}", tz="UTC")
                            times = pd.to_datetime(reference + pd.to_timedelta(np.asarray(values, dtype=float), unit="s"))
                        else:
                            times = pd.to_datetime(np.asarray(values, dtype=float), unit="s", utc=True)
                    table = pd.DataFrame({"profile_index": np.arange(len(times)), "Raw_Data_Start_Time": times.astype(str)})
                    st.dataframe(table, use_container_width=True, hide_index=True, height=220)
                except Exception as exc:
                    st.info(f"Não consegui decodificar Raw_Data_Start_Time: {exc}")
            else:
                st.info("Arquivo não tem Raw_Data_Start_Time.")
        with st.expander("Metadados deste nível"):
            render_metadata(ds, key_prefix=f"{prefix}_embedded_metadata")
        with st.expander("Diagnósticos deste nível"):
            render_diagnostics(ds)


def render_level1(row: dict[str, Any], *, mode: str | None = None) -> None:
    path = row.get("level1_path", "")
    if not path or not Path(path).exists():
        st.info("Level 1 não disponível para esta medida.")
        return
    st.caption(f"Arquivo: `{path}`")
    with open_dataset(path) as ds:
        channels = coord_values(ds, channel_name(ds))
        if not channels:
            st.warning("Não encontrei coordenada de canal no Level 1.")
            return
        prefix = safe_key(row.get("session_id"), "level1")
        if mode is None:
            mode = st.radio("Plot", ["Quicklook RCS + perfil", "Global mean RCS", "Genérico"], horizontal=True, key=f"{prefix}_plot_mode")
        altitudes = altitude_km(ds)
        max_alt_km = float(np.nanmax(altitudes)) if altitudes.size else 30.0
        smooth_bins = st.slider("Suavização vertical para perfis (bins)", 1, 80, 20, key=f"{prefix}_smooth")
        if mode == "Quicklook RCS + perfil":
            variables = numeric_variables(ds)
            default_var = "range_corrected_signal" if "range_corrected_signal" in ds else variables[0]
            c1, c2, c3, c4 = st.columns([1.2, 1.1, 0.8, 1.4])
            channel = c1.selectbox("Canal", channels, key=f"{prefix}_channel")
            variable = c2.selectbox("Variável", variables, index=variables.index(default_var), key=f"{prefix}_var")
            log10 = c3.checkbox("log10", value=False, key=f"{prefix}_log")
            color_range = c4.text_input("Range de cor `min,max`", value="", key=f"{prefix}_range")
            c5, c6 = st.columns([1, 2])
            robust = c5.checkbox("robust color", value=True, key=f"{prefix}_robust")
            percentiles = c6.slider("percentis do colormap", 0.0, 100.0, (2.0, 98.0), step=0.5, key=f"{prefix}_pct")
            plot_scientific(level1_quicklook_figure(ds, channel, variable, max_alt_km, log10, smooth_bins, color_range, robust, percentiles[0], percentiles[1]), use_container_width=True)
        elif mode == "Global mean RCS":
            default_channels = channels[: min(6, len(channels))]
            chosen = st.multiselect("Canais", channels, default=default_channels, key=f"{prefix}_global_channels")
            plot_scientific(global_mean_rcs_figure(ds, chosen, max_alt_km, smooth_bins), use_container_width=True)
        else:
            render_generic_level(row, "Level 1", ds=ds)
        with st.expander("Metadados deste nível"):
            render_metadata(ds, key_prefix=f"{prefix}_embedded_metadata")
        with st.expander("Diagnósticos deste nível"):
            render_diagnostics(ds)


def render_level2(row: dict[str, Any]) -> None:
    path = row.get("level2_path", "")
    if not path or not Path(path).exists():
        st.info("Level 2 não disponível para esta medida.")
        return
    st.caption(f"Arquivo: `{path}`")
    with ExitStack() as stack:
        ds_l2 = stack.enter_context(open_dataset(path))
        ds_l1 = (
            stack.enter_context(open_dataset(row["level1_path"]))
            if row.get("level1_path") and Path(row["level1_path"]).exists() else None
        )
        prefix = safe_key(row.get("session_id"), "level2")
        product_summary = level2_status_summary(ds_l2)
        processed_label = ", ".join(
            f"{value} nm" for value in product_summary["processed_wavelengths"]
        ) or "nenhum"
        failed_label = ", ".join(
            f"{value} nm" for value in product_summary["failed_wavelengths"]
        ) or "nenhum"
        status_text = (
            f"Completude: {product_summary['product_completeness']} · "
            f"processados: {processed_label} · falhos: {failed_label}"
        )
        if product_summary["product_completeness"] == "partial":
            st.warning(status_text)
            for failure in product_summary.get("failures", []):
                st.caption(
                    f"{failure['wavelength_nm']} nm: {failure['stage']} / "
                    f"{failure['code']} — {failure['message']}"
                )
        else:
            st.caption(status_text)
        wavelengths = available_level2_wavelengths(ds_l2)
        wavelength = st.selectbox("Comprimento de onda", wavelengths, key=f"{prefix}_wave") if wavelengths else None
        smooth_bins = st.slider("Suavização vertical dos perfis (bins)", 1, 80, 15, key=f"{prefix}_smooth")
        available_modes = ["Genérico"]
        if wavelength is not None:
            if {"glued_range_corrected_signal", "gluing_success_flag", "gluing_split_altitude_m"}.issubset(set(ds_l2.data_vars)):
                available_modes.insert(0, "QA gluing")
            if {"glued_range_corrected_signal_mean", "scaled_molecular_range_corrected_signal", "rayleigh_calibration_factor", "rayleigh_reference_altitude_m"}.issubset(set(ds_l2.data_vars)):
                available_modes.append("QA Rayleigh molecular")
            if "scattering_ratio_mean" in ds_l2:
                available_modes.append("QA scattering ratio")
            if first_available(ds_l2, ["aerosol_backscatter", "aerosol_backscatter_mean"]) and first_available(ds_l2, ["aerosol_extinction", "aerosol_extinction_mean"]):
                available_modes.append("QA KFS retrieval")
        mode = st.radio("Plot", available_modes, horizontal=True, key=f"{prefix}_plot_mode")
        if mode == "QA gluing" and wavelength is not None:
            fig = qa_gluing_figure(ds_l2, ds_l1, wavelength, smooth_bins)
        elif mode == "QA Rayleigh molecular" and wavelength is not None:
            fig = qa_molecular_figure(ds_l2, wavelength, smooth_bins)
        elif mode == "QA scattering ratio" and wavelength is not None:
            fig = qa_scattering_ratio_figure(ds_l2, wavelength, smooth_bins)
        elif mode == "QA KFS retrieval" and wavelength is not None:
            available_altitude = altitude_km(ds_l2)
            max_alt_km = float(np.nanmax(available_altitude)) if available_altitude.size else 30.0
            fig = qa_kfs_figure(ds_l2, wavelength, smooth_bins, max_alt_km)
        else:
            fig = None
            render_generic_level(row, "Level 2", ds=ds_l2, default_wavelength=wavelength)
        if fig is not None:
            plot_scientific(fig, use_container_width=True)
        elif mode != "Genérico":
            st.warning("Esse painel QA não encontrou as variáveis necessárias neste arquivo.")
        with st.expander("Metadados deste nível"):
            render_metadata(ds_l2, key_prefix=f"{prefix}_embedded_metadata")
        with st.expander("Diagnósticos deste nível"):
            render_diagnostics(ds_l2)


def render_generic_level(row: dict[str, Any], level: str, ds: xr.Dataset | None = None, default_wavelength: Any | None = None) -> None:
    if ds is None:
        path = row.get(LEVEL_TO_PATH[level], "")
        if not path or not Path(path).exists():
            st.info(f"{level} não disponível para esta medida.")
            return
        st.caption(f"Arquivo: `{path}`")
        with open_dataset(path) as opened:
            _render_generic_level(row, level, opened, default_wavelength)
        return
    _render_generic_level(row, level, ds, default_wavelength)


def _render_generic_level(
    row: dict[str, Any], level: str, ds: xr.Dataset, default_wavelength: Any | None
) -> None:
    variables = numeric_variables(ds)
    if not variables:
        st.info("Não encontrei variáveis numéricas plottáveis neste nível.")
        return
    prefix = safe_key(row.get("session_id"), level, "generic")
    c1, c2, c3 = st.columns([2, 1, 1])
    variable = c1.selectbox("Variável", variables, key=f"{prefix}_var")
    channels = coord_values(ds, channel_name(ds))
    channel = c2.selectbox("Canal", channels, key=f"{prefix}_channel") if channels else None
    wavelengths = wavelength_values(ds)
    wave_index = wavelengths.index(default_wavelength) if default_wavelength in wavelengths else 0
    wavelength = c3.selectbox("Comprimento de onda", wavelengths, index=wave_index, key=f"{prefix}_wave") if wavelengths else None
    vertical = vertical_values(ds, level)
    max_vertical = float(np.nanmax(vertical)) if vertical.size else (1.0 if level == "Level 0" else 30.0)
    c5, c6, c7, c8 = st.columns([1, 1, 1, 2])
    log10 = c5.checkbox("log10", value="signal" in variable.lower(), key=f"{prefix}_log")
    mode = c6.radio("Modo", ["quicklook", "perfil médio"], horizontal=True, key=f"{prefix}_mode")
    smooth_bins = c7.number_input("smooth bins", min_value=1, value=20, step=1, key=f"{prefix}_smooth")
    color_range = c8.text_input("Range de cor `min,max`", value="", key=f"{prefix}_range")
    c9, c10 = st.columns([1, 2])
    robust = c9.checkbox("robust color", value=True, key=f"{prefix}_robust")
    percentiles = c10.slider("percentis do colormap", 0.0, 100.0, (2.0, 98.0), step=0.5, key=f"{prefix}_pct")
    view = reduce_for_heatmap(ds[variable], level, channel, wavelength, float(max_vertical))
    title = f"{level} · {variable}" + (f" · {channel}" if channel is not None else "") + (f" · {wavelength_label(wavelength)}" if wavelength is not None else "")
    fig = profile_figure(view, title, level, int(smooth_bins)) if mode == "perfil médio" else heatmap_figure(view, title, level, log10, color_range, robust, percentiles[0], percentiles[1], ds=ds)
    plot_scientific(fig, use_container_width=True)

def render_level(row: dict[str, Any], level: str) -> None:
    if level == "Level 0":
        render_level0(row)
        return
    if level == "Level 1":
        render_level1(row)
        return
    if level == "Level 2":
        render_level2(row)
        return
    render_generic_level(row, level)


def render_metadata(ds: xr.Dataset, key_prefix: str) -> None:
    tabs = st.tabs(["attrs globais", "dimensões", "variáveis", "attrs variável", "estatísticas leves"])
    prefix = safe_key(key_prefix)
    with tabs[0]:
        st.dataframe(pd.DataFrame([{"attribute": k, "value": str(v)} for k, v in ds.attrs.items()]), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame([{"dimension": k, "size": int(v)} for k, v in ds.sizes.items()]), use_container_width=True, hide_index=True)
    with tabs[2]:
        rows = []
        for name, da in ds.variables.items():
            rows.append({"name": name, "kind": "coord" if name in ds.coords else "data_var", "dims": " × ".join(da.dims), "shape": " × ".join(str(s) for s in da.shape), "dtype": str(da.dtype), "units": da.attrs.get("units", ""), "long_name": da.attrs.get("long_name", da.attrs.get("standard_name", ""))})
        table = pd.DataFrame(rows)
        query = st.text_input("Filtrar variáveis/attrs", key=f"{prefix}_filter")
        if query:
            table = table[table.apply(lambda col: col.astype(str).str.contains(query, case=False, na=False)).any(axis=1)]
        st.dataframe(table, use_container_width=True, hide_index=True)
    with tabs[3]:
        var = st.selectbox("Variável", list(ds.variables), key=f"{prefix}_attrs_var")
        st.dataframe(pd.DataFrame([{"attribute": k, "value": str(v)} for k, v in ds[var].attrs.items()]), use_container_width=True, hide_index=True)
    with tabs[4]:
        names = numeric_variables(ds)
        chosen = st.multiselect("Variáveis", names, default=names[: min(8, len(names))], key=f"{prefix}_stats_vars")
        rows = []
        for name in chosen:
            da = ds[name]
            total = int(np.prod(da.shape)) if da.shape else 1
            step = max(1, int(np.ceil(total / 300_000)))
            subset = da.isel({dim: slice(None, None, step) for dim in da.dims}) if total > 300_000 else da
            values = np.asarray(subset.values, dtype=float).ravel()
            finite = np.isfinite(values)
            valid = values[finite]
            rows.append({"variable": name, "sampled_values": int(values.size), "finite_fraction": float(finite.mean()) if values.size else np.nan, "min": float(np.nanmin(valid)) if valid.size else np.nan, "median": float(np.nanmedian(valid)) if valid.size else np.nan, "max": float(np.nanmax(valid)) if valid.size else np.nan})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_diagnostics(ds: xr.Dataset) -> None:
    rows = []
    for name in DIAGNOSTIC_NAMES:
        if name in ds.variables and ds[name].size <= 200:
            value = ds[name].values
            value = value.item() if np.asarray(value).size == 1 else np.asarray(value).tolist()
            rows.append({"diagnostic": name, "value": str(value)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum diagnóstico escalar/small-array conhecido encontrado.")


def render_atmospheric_profiles(row: dict[str, Any]) -> None:
    """Inspect saved atmospheric state from L1 without recalculating meteorology."""
    path = row.get("level1_path", "")
    if not path or not Path(path).is_file():
        st.info("Não há arquivo L1 com perfis atmosféricos nesta sessão.")
        return
    st.caption(f"Fonte L1: {Path(path).name}")
    with open_dataset(path) as ds:
        candidates = [
            name for name in (
                "Atmospheric_Temperature_K",
                "Atmospheric_Pressure_hPa",
                "Atmospheric_Relative_Humidity_percent",
            )
            if name in ds and "altitude" in ds[name].dims
        ]
        if not candidates:
            st.info(
                "Este produto L1 não possui perfis atmosféricos em altitude. "
                "Verifique as variáveis em Metadata."
            )
            return
        variable = st.selectbox("Variável atmosférica", candidates, key="mg_atmos_var")
        da = ds[variable]
        # Mean is a display-only reduction; the time-resolved source is unchanged.
        fig = profile_figure(da, variable, "Level 1", smooth_bins=1)
        units = str(da.attrs.get("units", "")).strip()
        fig.update_xaxes(title_text=f"{variable} ({units})" if units else variable)
        plot_scientific(fig, use_container_width=True)
        st.caption(
            "Perfil médio para visualização dos registros disponíveis. "
            "O NetCDF mantém todos os horários originais."
        )
