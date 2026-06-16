"""Read-only Streamlit explorer for MILGRAU Level 0/1/2 NetCDF files."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.io.paths import LEVEL1_SUFFIX, LEVEL2_SUFFIX, processed_data_root

LEVELS = {
    "Level 0": ".nc",
    "Level 1": LEVEL1_SUFFIX,
    "Level 2": LEVEL2_SUFFIX,
}
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
    "gluing_start_altitude_m",
    "gluing_stop_altitude_m",
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


def _parse_save_id(save_id: str) -> tuple[date | None, str]:
    try:
        day = datetime.strptime(save_id[:8], "%Y%m%d").date()
    except ValueError:
        day = None
    return day, save_id[-2:] if len(save_id) >= 2 else "--"


def _find_level0(product_dir: Path, save_id: str) -> Path | None:
    expected = product_dir / f"{save_id}.nc"
    if expected.exists():
        return expected
    candidates = sorted(
        p for p in product_dir.glob("*.nc")
        if not p.name.endswith(LEVEL1_SUFFIX) and not p.name.endswith(LEVEL2_SUFFIX)
    )
    return candidates[0] if candidates else None


def _product_paths(product_dir: Path) -> dict[str, str]:
    save_id = product_dir.name
    level0 = _find_level0(product_dir, save_id)
    level1 = product_dir / f"{save_id}{LEVEL1_SUFFIX}"
    level2 = product_dir / f"{save_id}{LEVEL2_SUFFIX}"
    return {
        "Level 0": str(level0) if level0 and level0.exists() else "",
        "Level 1": str(level1) if level1.exists() else "",
        "Level 2": str(level2) if level2.exists() else "",
    }


def discover_products(processed_root: Path) -> pd.DataFrame:
    """Return one row per product folder under YYYY/MM/save_id."""

    rows: list[dict[str, Any]] = []
    root = processed_root.expanduser().resolve()
    if not root.exists():
        return pd.DataFrame()
    for product_dir in sorted(root.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/*")):
        if not product_dir.is_dir():
            continue
        save_id = product_dir.name
        paths = _product_paths(product_dir)
        if not any(paths.values()):
            continue
        day, period = _parse_save_id(save_id)
        mtimes = [Path(p).stat().st_mtime for p in paths.values() if p]
        rows.append(
            {
                "date": day,
                "save_id": save_id,
                "period": period,
                "product_dir": str(product_dir),
                "level0_path": paths["Level 0"],
                "level1_path": paths["Level 1"],
                "level2_path": paths["Level 2"],
                "has_level0": bool(paths["Level 0"]),
                "has_level1": bool(paths["Level 1"]),
                "has_level2": bool(paths["Level 2"]),
                "available_levels": ", ".join(k for k, v in paths.items() if v),
                "modified": datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds") if mtimes else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "save_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def processed_root_from_config(config_path: str) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    return processed_data_root(config, root_dir=config_file.parent)


@st.cache_data(show_spinner="Escaneando produtos NetCDF...")
def cached_inventory(processed_root: str) -> pd.DataFrame:
    return discover_products(Path(processed_root))


@st.cache_resource(show_spinner="Abrindo NetCDF...")
def open_nc(path: str) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True)


def coord_name(ds: xr.Dataset | xr.DataArray, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in ds.dims or name in getattr(ds, "coords", {}):
            return name
    return None


def altitude_name(ds: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(ds, ("altitude", "height", "range", "points", "range_bin"))


def time_name(ds: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(ds, ("time", "Time", "Raw_Data_Start_Time", "block_time", "profile"))


def channel_name(ds: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(ds, ("channel", "channels"))


def values_for(ds: xr.Dataset, name: str | None) -> list[Any]:
    if name and name in ds.variables:
        return [v.item() if hasattr(v, "item") else v for v in np.asarray(ds[name].values).ravel()]
    return []


def wavelength_values(ds: xr.Dataset) -> list[Any]:
    for name in ("wavelength", "wavelength_nm"):
        values = values_for(ds, name)
        if values:
            return values
    return []


def numeric_vars(ds: xr.Dataset) -> list[str]:
    preferred = [
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
    vars_ = [n for n, da in ds.data_vars.items() if da.ndim <= 5 and np.issubdtype(da.dtype, np.number)]
    return [n for n in preferred if n in vars_] + [n for n in vars_ if n not in preferred]


def _label(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def select_dim(da: xr.DataArray, dim: str | None, value: Any) -> xr.DataArray:
    if dim is None or dim not in da.dims:
        return da
    if dim not in da.coords:
        return da.isel({dim: 0})
    labels = [_label(v) for v in np.asarray(da[dim].values).ravel()]
    return da.isel({dim: labels.index(_label(value)) if _label(value) in labels else 0})


def quicklook_slice(
    da: xr.DataArray,
    channel: Any | None,
    wavelength: Any | None,
    max_alt_km: float,
    max_profiles: int = 900,
    max_bins: int = 900,
) -> xr.DataArray:
    da = select_dim(da, channel_name(da), channel)
    for wdim in ("wavelength", "wavelength_nm"):
        da = select_dim(da, wdim if wdim in da.dims else None, wavelength)
    alt = altitude_name(da)
    if alt and alt in da.coords:
        try:
            da = da.where(da[alt] <= max_alt_km * 1000.0, drop=True)
        except Exception:
            pass
    tdim = time_name(da)
    keep = {d for d in (tdim, alt) if d and d in da.dims}
    for dim in list(da.dims):
        if dim not in keep:
            da = da.isel({dim: 0})
    if da.ndim == 1:
        da = da.expand_dims({"profile": [0]})
    while da.ndim > 2:
        da = da.isel({da.dims[-1]: 0})
    slices = {}
    for dim in da.dims:
        limit = max_profiles if dim == tdim else max_bins
        if da.sizes[dim] > limit:
            slices[dim] = slice(None, None, int(np.ceil(da.sizes[dim] / limit)))
    return da.isel(slices) if slices else da


def heatmap(da: xr.DataArray, title: str, log10: bool, vmin: float | None, vmax: float | None) -> go.Figure:
    values = np.asarray(da.values, dtype=float)
    if log10:
        values = np.where(values > 0, np.log10(values), np.nan)
    dims = list(da.dims)
    xdim = dims[0] if dims else "profile"
    ydim = dims[-1] if len(dims) > 1 else "vertical"
    z = values.T if values.ndim == 2 else np.atleast_2d(values)
    x = da[xdim].values if xdim in da.coords else np.arange(z.shape[1])
    y = da[ydim].values if ydim in da.coords else np.arange(z.shape[0])
    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, zmin=vmin, zmax=vmax, colorbar={"title": "log10" if log10 else da.name}))
    fig.update_layout(title=title, xaxis_title=xdim, yaxis_title=ydim, margin={"l": 60, "r": 20, "t": 50, "b": 50})
    return fig


def profile(da: xr.DataArray, title: str) -> go.Figure:
    alt = altitude_name(da) or da.dims[-1]
    reduce_dims = [d for d in da.dims if d != alt]
    mean = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
    y = mean[alt].values if alt in mean.coords else np.arange(mean.size)
    fig = go.Figure(data=go.Scatter(x=np.asarray(mean.values, dtype=float), y=y, mode="lines"))
    fig.update_layout(title=title, xaxis_title=mean.name or "value", yaxis_title=alt, margin={"l": 60, "r": 20, "t": 50, "b": 50})
    return fig


def global_attrs(ds: xr.Dataset) -> pd.DataFrame:
    return pd.DataFrame([{"attribute": k, "value": str(v)} for k, v in ds.attrs.items()])


def variables_table(ds: xr.Dataset) -> pd.DataFrame:
    rows = []
    for name, da in ds.variables.items():
        rows.append({
            "name": name,
            "kind": "coord" if name in ds.coords else "data_var",
            "dims": " × ".join(da.dims),
            "shape": " × ".join(str(s) for s in da.shape),
            "dtype": str(da.dtype),
            "units": da.attrs.get("units", ""),
            "long_name": da.attrs.get("long_name", da.attrs.get("standard_name", "")),
        })
    return pd.DataFrame(rows)


def stats_table(ds: xr.Dataset, names: list[str], sample_limit: int = 300_000) -> pd.DataFrame:
    rows = []
    for name in names:
        da = ds[name]
        total = int(np.prod(da.shape)) if da.shape else 1
        step = max(1, int(np.ceil(total / sample_limit)))
        subset = da.isel({dim: slice(None, None, step) for dim in da.dims}) if total > sample_limit else da
        values = np.asarray(subset.values, dtype=float).ravel()
        finite = np.isfinite(values)
        valid = values[finite]
        rows.append({
            "variable": name,
            "sampled_values": int(values.size),
            "finite_fraction": float(finite.mean()) if values.size else np.nan,
            "min": float(np.nanmin(valid)) if valid.size else np.nan,
            "median": float(np.nanmedian(valid)) if valid.size else np.nan,
            "max": float(np.nanmax(valid)) if valid.size else np.nan,
        })
    return pd.DataFrame(rows)


def diagnostics(ds: xr.Dataset) -> dict[str, Any]:
    out = {}
    for name in DIAGNOSTIC_NAMES:
        if name in ds.variables and ds[name].size <= 20:
            value = ds[name].values
            out[name] = value.item() if np.asarray(value).size == 1 else np.asarray(value).tolist()
    return out


def render_metadata(ds: xr.Dataset) -> None:
    tabs = st.tabs(["attrs globais", "dimensões", "variáveis", "attrs variável", "estatísticas leves"])
    with tabs[0]:
        st.dataframe(global_attrs(ds), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame([{"dimension": k, "size": int(v)} for k, v in ds.sizes.items()]), use_container_width=True, hide_index=True)
    with tabs[2]:
        table = variables_table(ds)
        query = st.text_input("Filtrar", key=f"filter_{id(ds)}")
        if query:
            table = table[table.apply(lambda col: col.astype(str).str.contains(query, case=False, na=False)).any(axis=1)]
        st.dataframe(table, use_container_width=True, hide_index=True)
    with tabs[3]:
        var = st.selectbox("Variável", list(ds.variables), key=f"attrs_{id(ds)}")
        st.dataframe(pd.DataFrame([{"attribute": k, "value": str(v)} for k, v in ds[var].attrs.items()]), use_container_width=True, hide_index=True)
    with tabs[4]:
        names = numeric_vars(ds)
        chosen = st.multiselect("Variáveis", names, default=names[: min(8, len(names))], key=f"stats_{id(ds)}")
        if chosen:
            st.dataframe(stats_table(ds, chosen), use_container_width=True, hide_index=True)


def render_level(row: dict[str, Any], label: str) -> None:
    path_key = {"Level 0": "level0_path", "Level 1": "level1_path", "Level 2": "level2_path"}[label]
    path = row.get(path_key, "")
    if not path or not Path(path).exists():
        st.info(f"{label} não disponível para esta medida.")
        return
    st.caption(f"Arquivo: `{path}`")
    ds = open_nc(path)
    names = numeric_vars(ds)
    if names:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        var = c1.selectbox("Variável", names, key=f"var_{label}")
        channels = values_for(ds, channel_name(ds))
        channel = c2.selectbox("Canal", channels, key=f"channel_{label}") if channels else None
        wvals = wavelength_values(ds)
        wavelength = c3.selectbox("Comprimento de onda", wvals, key=f"wavelength_{label}") if wvals else None
        max_alt = c4.number_input("Altitude máx. (km)", min_value=0.1, value=15.0, step=0.5, key=f"alt_{label}")
        c5, c6, c7, c8 = st.columns(4)
        log10 = c5.checkbox("log10", value="signal" in var.lower(), key=f"log_{label}")
        vmin = c6.number_input("mín. cor", value=None, placeholder="auto", key=f"vmin_{label}")
        vmax = c7.number_input("máx. cor", value=None, placeholder="auto", key=f"vmax_{label}")
        mode = c8.radio("Modo", ["quicklook", "perfil médio"], horizontal=True, key=f"mode_{label}")
        view = quicklook_slice(ds[var], channel, wavelength, max_alt)
        title = f"{label} · {var}" + (f" · {channel}" if channel is not None else "") + (f" · {wavelength} nm" if wavelength is not None else "")
        st.plotly_chart(profile(view, title) if mode == "perfil médio" else heatmap(view, title, log10, vmin, vmax), use_container_width=True)
    with st.expander("Metadados deste nível"):
        render_metadata(ds)
    diag = diagnostics(ds)
    with st.expander("Diagnósticos deste nível"):
        if diag:
            cols = st.columns(3)
            for i, (k, v) in enumerate(diag.items()):
                cols[i % 3].metric(k, str(v))
        else:
            st.info("Nenhum diagnóstico escalar/small-array conhecido encontrado.")


def calendar_events(inv: pd.DataFrame) -> list[dict[str, Any]]:
    events = []
    for day, group in inv.dropna(subset=["date"]).groupby("date"):
        periods = ", ".join(sorted(str(p) for p in group["period"].unique()))
        events.append({"title": f"{len(group)} medida(s): {periods}", "start": day.isoformat(), "allDay": True, "extendedProps": {"date": day.isoformat()}})
    return events


def select_day(inv: pd.DataFrame) -> date | None:
    days = sorted(d for d in inv["date"].dropna().unique())
    if not days:
        return None
    try:
        from streamlit_calendar import calendar
    except Exception:
        st.caption("Fallback: instale `streamlit-calendar` para calendário destacado/clicável.")
        return st.selectbox("Dia disponível", days, index=len(days) - 1, format_func=lambda d: d.isoformat())
    resp = calendar(
        events=calendar_events(inv),
        options={"initialView": "dayGridMonth", "initialDate": days[-1].isoformat(), "locale": "pt-br", "height": 430, "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth"}},
        custom_css=".fc-event{cursor:pointer;border-radius:999px;padding:2px 6px}.fc-daygrid-day:has(.fc-event){background-color:rgba(76,175,80,.08)}",
        key="milgrau_calendar",
    )
    selected = None
    if isinstance(resp, dict):
        if resp.get("eventClick"):
            selected = resp["eventClick"]["event"].get("extendedProps", {}).get("date")
        elif resp.get("dateClick"):
            selected = resp["dateClick"].get("date", "")[:10]
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
            cached_inventory.clear()
            open_nc.clear()
        inv = cached_inventory(str(processed_root))
        st.metric("medidas encontradas", len(inv))
        if inv.empty:
            st.warning("Nenhum produto .nc encontrado na pasta de dados.")
            st.stop()
        day = select_day(inv)
        day_rows = inv[inv["date"] == day].reset_index(drop=True)
        idx = st.selectbox("Medida", list(day_rows.index), format_func=lambda i: f"{day_rows.loc[i, 'save_id']} · {day_rows.loc[i, 'available_levels']}")
        row = day_rows.loc[idx].to_dict()
    tabs = st.tabs(["Resumo", "Level 0", "Level 1", "Level 2", "Metadados", "QA"])
    with tabs[0]:
        cols = st.columns(4)
        cols[0].metric("save_id", row["save_id"])
        cols[1].metric("período", row["period"])
        cols[2].metric("níveis", row["available_levels"])
        cols[3].metric("modificado", row["modified"])
        st.caption(f"Pasta de dados: `{processed_root}`")
        st.dataframe(pd.DataFrame([{"level": k, "path": row[v]} for k, v in {"Level 0": "level0_path", "Level 1": "level1_path", "Level 2": "level2_path"}.items()]), use_container_width=True, hide_index=True)
        with st.expander("Inventário completo"):
            st.dataframe(inv, use_container_width=True, hide_index=True)
    for tab, label in zip(tabs[1:4], ["Level 0", "Level 1", "Level 2"]):
        with tab:
            render_level(row, label)
    with tabs[4]:
        available = [label for label, key in {"Level 0": "level0_path", "Level 1": "level1_path", "Level 2": "level2_path"}.items() if row.get(key)]
        label = st.selectbox("Nível", available, key="metadata_level")
        path = row[{"Level 0": "level0_path", "Level 1": "level1_path", "Level 2": "level2_path"}[label]]
        render_metadata(open_nc(path))
    with tabs[5]:
        for label, key in {"Level 0": "level0_path", "Level 1": "level1_path", "Level 2": "level2_path"}.items():
            if row.get(key):
                st.markdown(f"### {label}")
                diag = diagnostics(open_nc(row[key]))
                if diag:
                    st.dataframe(pd.DataFrame([{"diagnostic": k, "value": str(v)} for k, v in diag.items()]), use_container_width=True, hide_index=True)
                else:
                    st.info("Nenhum diagnóstico escalar/small-array conhecido encontrado.")


if __name__ == "__main__":
    import streamlit as st

    main()
