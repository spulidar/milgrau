"""Runnable Streamlit app for browsing MILGRAU NetCDF products.

The app is read-only: it scans and opens existing files but never writes to the
MILGRAU processed-data directory.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.io.paths import LEVEL1_SUFFIX, LEVEL2_SUFFIX, processed_data_root

LEVEL_TO_PATH = {
    "Level 0": "level0_path",
    "Level 1": "level1_path",
    "Level 2": "level2_path",
}
PREFERRED_VARIABLES = [
    "glued_range_corrected_signal",
    "range_corrected_signal",
    "corrected_signal",
    "Raw_Lidar_Data",
    "glued_corrected_signal",
    "scattering_ratio_mean",
    "aerosol_backscatter_mean",
    "aerosol_extinction_mean",
    "molecular_backscatter",
    "molecular_extinction",
]
DIAGNOSTIC_NAMES = [
    "PBL_Height_km",
    "pc_saturation_fraction",
    "deadtime_clipping_fraction",
    "bin_shift_invalid_fraction",
    "channel_correction_success",
    "dark_current_used",
    "gluing_success_flag",
    "gluing_fallback_flag",
    "gluing_split_altitude_m",
    "gluing_correlation",
    "gluing_relative_rmse",
    "gluing_relative_bias",
    "rayleigh_reference_success_flag",
    "rayleigh_reference_altitude_m",
    "rayleigh_reference_valid_fraction",
    "rayleigh_calibration_factor",
    "valid_retrieval_block_flag",
    "lidar_ratio_assumed_sr",
    "kfs_branch",
]


def processed_root_from_config(config_path: str) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    return processed_data_root(config, root_dir=config_file.parent)


def parse_save_id(save_id: str) -> tuple[date | None, str]:
    try:
        day = datetime.strptime(save_id[:8], "%Y%m%d").date()
    except ValueError:
        day = None
    return day, save_id[-2:] if len(save_id) >= 2 else "--"


def product_paths(product_dir: Path) -> dict[str, str]:
    save_id = product_dir.name
    level0 = product_dir / f"{save_id}.nc"
    if not level0.exists():
        candidates = sorted(
            path
            for path in product_dir.glob("*.nc")
            if not path.name.endswith(LEVEL1_SUFFIX) and not path.name.endswith(LEVEL2_SUFFIX)
        )
        level0 = candidates[0] if candidates else level0
    level1 = product_dir / f"{save_id}{LEVEL1_SUFFIX}"
    level2 = product_dir / f"{save_id}{LEVEL2_SUFFIX}"
    return {
        "level0_path": str(level0) if level0.exists() else "",
        "level1_path": str(level1) if level1.exists() else "",
        "level2_path": str(level2) if level2.exists() else "",
    }


@st.cache_data(show_spinner="Escaneando produtos NetCDF...")
def discover_products(processed_root: str) -> pd.DataFrame:
    root = Path(processed_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for product_dir in sorted(root.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/*")):
        if not product_dir.is_dir():
            continue
        paths = product_paths(product_dir)
        if not any(paths.values()):
            continue
        day, period = parse_save_id(product_dir.name)
        mtimes = [Path(path).stat().st_mtime for path in paths.values() if path]
        rows.append(
            {
                "date": day,
                "save_id": product_dir.name,
                "period": period,
                "product_dir": str(product_dir),
                **paths,
                "available_levels": ", ".join(label for label, key in LEVEL_TO_PATH.items() if paths[key]),
                "modified": datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds") if mtimes else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "save_id"]).reset_index(drop=True) if rows else pd.DataFrame()


@st.cache_resource(show_spinner="Abrindo NetCDF...")
def open_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True)


def coord_name(obj: xr.Dataset | xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in obj.dims or name in getattr(obj, "coords", {}):
            return name
    return None


def altitude_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("altitude", "height", "range", "points", "range_bin"))


def time_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("time", "Time", "Raw_Data_Start_Time", "block_time", "profile"))


def channel_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("channel", "channels"))


def coord_values(ds: xr.Dataset, name: str | None) -> list[Any]:
    if name and name in ds.variables:
        return [value.item() if hasattr(value, "item") else value for value in np.asarray(ds[name].values).ravel()]
    return []


def wavelength_values(ds: xr.Dataset) -> list[Any]:
    for name in ("wavelength", "wavelength_nm"):
        values = coord_values(ds, name)
        if values:
            return values
    return []


def numeric_variables(ds: xr.Dataset) -> list[str]:
    variables = [name for name, da in ds.data_vars.items() if da.ndim <= 5 and np.issubdtype(da.dtype, np.number)]
    return [name for name in PREFERRED_VARIABLES if name in variables] + [name for name in variables if name not in PREFERRED_VARIABLES]


def label(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def select_coord(da: xr.DataArray, dim: str | None, value: Any) -> xr.DataArray:
    if dim is None or value is None or dim not in da.dims:
        return da
    if dim not in da.coords:
        return da.isel({dim: 0})
    labels = [label(value) for value in np.asarray(da[dim].values).ravel()]
    index = labels.index(label(value)) if label(value) in labels else 0
    return da.isel({dim: index})


def reduced_view(da: xr.DataArray, channel: Any | None, wavelength: Any | None, max_alt_km: float) -> xr.DataArray:
    da = select_coord(da, channel_name(da), channel)
    da = select_coord(da, "wavelength" if "wavelength" in da.dims else None, wavelength)
    da = select_coord(da, "wavelength_nm" if "wavelength_nm" in da.dims else None, wavelength)
    alt = altitude_name(da)
    if alt and alt in da.coords:
        try:
            da = da.where(da[alt] <= max_alt_km * 1000.0, drop=True)
        except Exception:
            pass
    tdim = time_name(da)
    keep = {dim for dim in (tdim, alt) if dim and dim in da.dims}
    for dim in list(da.dims):
        if dim not in keep:
            da = da.isel({dim: 0})
    if da.ndim == 1:
        da = da.expand_dims({"profile": [0]})
    while da.ndim > 2:
        da = da.isel({da.dims[-1]: 0})
    slices = {}
    for dim in da.dims:
        limit = 900
        if da.sizes[dim] > limit:
            slices[dim] = slice(None, None, int(np.ceil(da.sizes[dim] / limit)))
    return da.isel(slices) if slices else da


def parse_color_range(text: str) -> tuple[float | None, float | None]:
    if not text.strip():
        return None, None
    try:
        left, right = text.split(",", 1)
        return float(left), float(right)
    except Exception:
        st.warning("Use range de cor no formato `min,max`, ou deixe vazio para automático.")
        return None, None


def heatmap_figure(da: xr.DataArray, title: str, log10: bool, color_range: str) -> go.Figure:
    z = np.asarray(da.values, dtype=float)
    if log10:
        z = np.where(z > 0, np.log10(z), np.nan)
    z = z.T if z.ndim == 2 else np.atleast_2d(z)
    xdim = da.dims[0] if da.dims else "profile"
    ydim = da.dims[-1] if len(da.dims) > 1 else "vertical"
    x = da[xdim].values if xdim in da.coords else np.arange(z.shape[1])
    y = da[ydim].values if ydim in da.coords else np.arange(z.shape[0])
    zmin, zmax = parse_color_range(color_range)
    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, zmin=zmin, zmax=zmax, colorbar={"title": "log10" if log10 else da.name}))
    fig.update_layout(title=title, xaxis_title=xdim, yaxis_title=ydim, margin={"l": 60, "r": 20, "t": 50, "b": 50})
    return fig


def profile_figure(da: xr.DataArray, title: str) -> go.Figure:
    alt = altitude_name(da) or da.dims[-1]
    reduced = da.mean(dim=[dim for dim in da.dims if dim != alt], skipna=True)
    y = reduced[alt].values if alt in reduced.coords else np.arange(reduced.size)
    fig = go.Figure(data=go.Scatter(x=np.asarray(reduced.values, dtype=float), y=y, mode="lines"))
    fig.update_layout(title=title, xaxis_title=reduced.name or "value", yaxis_title=alt, margin={"l": 60, "r": 20, "t": 50, "b": 50})
    return fig


def render_metadata(ds: xr.Dataset) -> None:
    tabs = st.tabs(["attrs globais", "dimensões", "variáveis", "attrs variável", "estatísticas leves"])
    with tabs[0]:
        st.dataframe(pd.DataFrame([{"attribute": k, "value": str(v)} for k, v in ds.attrs.items()]), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame([{"dimension": k, "size": int(v)} for k, v in ds.sizes.items()]), use_container_width=True, hide_index=True)
    with tabs[2]:
        rows = []
        for name, da in ds.variables.items():
            rows.append({"name": name, "kind": "coord" if name in ds.coords else "data_var", "dims": " × ".join(da.dims), "shape": " × ".join(str(s) for s in da.shape), "dtype": str(da.dtype), "units": da.attrs.get("units", ""), "long_name": da.attrs.get("long_name", da.attrs.get("standard_name", ""))})
        table = pd.DataFrame(rows)
        query = st.text_input("Filtrar variáveis/attrs", key=f"filter_{id(ds)}")
        if query:
            table = table[table.apply(lambda col: col.astype(str).str.contains(query, case=False, na=False)).any(axis=1)]
        st.dataframe(table, use_container_width=True, hide_index=True)
    with tabs[3]:
        var = st.selectbox("Variável", list(ds.variables), key=f"attrs_{id(ds)}")
        st.dataframe(pd.DataFrame([{"attribute": k, "value": str(v)} for k, v in ds[var].attrs.items()]), use_container_width=True, hide_index=True)
    with tabs[4]:
        names = numeric_variables(ds)
        chosen = st.multiselect("Variáveis", names, default=names[: min(8, len(names))], key=f"stats_{id(ds)}")
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
        if name in ds.variables and ds[name].size <= 20:
            value = ds[name].values
            value = value.item() if np.asarray(value).size == 1 else np.asarray(value).tolist()
            rows.append({"diagnostic": name, "value": str(value)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum diagnóstico escalar/small-array conhecido encontrado.")


def render_level(row: dict[str, Any], level: str) -> None:
    path = row.get(LEVEL_TO_PATH[level], "")
    if not path or not Path(path).exists():
        st.info(f"{level} não disponível para esta medida.")
        return
    st.caption(f"Arquivo: `{path}`")
    ds = open_dataset(path)
    variables = numeric_variables(ds)
    if variables:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        variable = c1.selectbox("Variável", variables, key=f"var_{level}")
        channels = coord_values(ds, channel_name(ds))
        channel = c2.selectbox("Canal", channels, key=f"channel_{level}") if channels else None
        wavelengths = wavelength_values(ds)
        wavelength = c3.selectbox("Comprimento de onda", wavelengths, key=f"wave_{level}") if wavelengths else None
        max_alt_km = c4.number_input("Altitude máx. (km)", min_value=0.1, value=15.0, step=0.5, key=f"alt_{level}")
        c5, c6, c7 = st.columns([1, 1, 2])
        log10 = c5.checkbox("log10", value="signal" in variable.lower(), key=f"log_{level}")
        mode = c6.radio("Modo", ["quicklook", "perfil médio"], horizontal=True, key=f"mode_{level}")
        color_range = c7.text_input("Range de cor `min,max`", value="", key=f"range_{level}")
        view = reduced_view(ds[variable], channel, wavelength, max_alt_km)
        title = f"{level} · {variable}" + (f" · {channel}" if channel is not None else "") + (f" · {wavelength} nm" if wavelength is not None else "")
        fig = profile_figure(view, title) if mode == "perfil médio" else heatmap_figure(view, title, log10, color_range)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Metadados deste nível"):
        render_metadata(ds)
    with st.expander("Diagnósticos deste nível"):
        render_diagnostics(ds)


def calendar_events(inv: pd.DataFrame) -> list[dict[str, Any]]:
    events = []
    for day, group in inv.dropna(subset=["date"]).groupby("date"):
        periods = ", ".join(sorted(str(period) for period in group["period"].unique()))
        events.append({"title": f"{len(group)} medida(s): {periods}", "start": day.isoformat(), "allDay": True, "extendedProps": {"date": day.isoformat()}})
    return events


def select_day(inv: pd.DataFrame) -> date | None:
    days = sorted(day for day in inv["date"].dropna().unique())
    if not days:
        return None
    try:
        from streamlit_calendar import calendar
    except Exception:
        st.caption("Fallback: instale `streamlit-calendar` para calendário destacado/clicável.")
        return st.selectbox("Dia disponível", days, index=len(days) - 1, format_func=lambda day: day.isoformat())
    response = calendar(
        events=calendar_events(inv),
        options={"initialView": "dayGridMonth", "initialDate": days[-1].isoformat(), "locale": "pt-br", "height": 430, "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth"}},
        custom_css=".fc-event{cursor:pointer;border-radius:999px;padding:2px 6px}.fc-daygrid-day:has(.fc-event){background-color:rgba(76,175,80,.08)}",
        key="milgrau_calendar",
    )
    selected = None
    if isinstance(response, dict):
        if response.get("eventClick"):
            selected = response["eventClick"]["event"].get("extendedProps", {}).get("date")
        elif response.get("dateClick"):
            selected = response["dateClick"].get("date", "")[:10]
    if selected:
        try:
            day = date.fromisoformat(selected)
            if day in days:
                st.session_state["selected_day"] = day
        except ValueError:
            pass
    return st.session_state.get("selected_day", days[-1])


def main() -> None:
    st.set_page_config(page_title="MILGRAU NetCDF Explorer", layout="wide")
    st.title("MILGRAU NetCDF Explorer")
    st.caption("Explorador read-only para Level 0 LIBIDS, Level 1 LIPANCORA e Level 2 LEBEAR.")
    with st.sidebar:
        st.header("Dados")
        config_path = st.text_input("config.yaml", value="config.yaml")
        try:
            default_root = processed_root_from_config(config_path)
        except Exception as exc:
            st.error(f"Não consegui carregar o config: {exc}")
            st.stop()
        processed_root = Path(st.text_input("Pasta processed_data", value=str(default_root))).expanduser()
        if st.button("Atualizar inventário"):
            discover_products.clear()
            open_dataset.clear()
        inv = discover_products(str(processed_root))
        st.metric("medidas encontradas", len(inv))
        if inv.empty:
            st.warning("Nenhum produto .nc encontrado na pasta de dados.")
            st.stop()
        selected_day = select_day(inv)
        if selected_day is None:
            st.stop()
        day_rows = inv[inv["date"] == selected_day].reset_index(drop=True)
        index = st.selectbox("Medida", list(day_rows.index), format_func=lambda i: f"{day_rows.loc[i, 'save_id']} · {day_rows.loc[i, 'available_levels']}")
        row = day_rows.loc[index].to_dict()
    tabs = st.tabs(["Resumo", "Level 0", "Level 1", "Level 2", "Metadados", "QA"])
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("save_id", row["save_id"])
        c2.metric("período", row["period"])
        c3.metric("níveis", row["available_levels"])
        c4.metric("modificado", row["modified"])
        st.caption(f"Pasta de dados: `{processed_root}`")
        st.dataframe(pd.DataFrame([{"level": level, "path": row[key]} for level, key in LEVEL_TO_PATH.items()]), use_container_width=True, hide_index=True)
        with st.expander("Inventário completo"):
            st.dataframe(inv, use_container_width=True, hide_index=True)
    for tab, level in zip(tabs[1:4], ["Level 0", "Level 1", "Level 2"]):
        with tab:
            render_level(row, level)
    with tabs[4]:
        available = [level for level, key in LEVEL_TO_PATH.items() if row.get(key)]
        selected_level = st.selectbox("Nível", available, key="metadata_level")
        render_metadata(open_dataset(row[LEVEL_TO_PATH[selected_level]]))
    with tabs[5]:
        for level, key in LEVEL_TO_PATH.items():
            if row.get(key):
                st.markdown(f"### {level}")
                render_diagnostics(open_dataset(row[key]))


if __name__ == "__main__":
    main()
