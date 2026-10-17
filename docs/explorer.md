# MILGRAU Web Explorer — local, read-only

The old Streamlit explorer has been removed. The new application starts from
the MILGRAU processing product contracts and reuses only core configuration
and path functions: `milgrau.config.loader` and `milgrau.io.paths`.

## Installation

FastAPI, Uvicorn and Plotly are **core dependencies**. No optional explorer
extra is needed. From the project directory:

```bash
python -m pip install -e .
milgrau-explorer
```

Open http://127.0.0.1:8765 in a browser. The application binds to loopback
by default, loads `config.yaml` from the working directory and resolves
`directories.processed_data` and the referenced `station.yaml`.
For a different configuration: `milgrau-explorer --config /path/to/config.yaml`.
Port is configurable with `--port`. No Node, CDN, cloud service or frontend
build is required: Plotly.js is served from the installed Python package.

## Phase 1: RCS Maps

- One compact session selector in the header, listing all scientifically continuous
  acquisitions in newest-first UTC order (no separate date selector or duplicated
  session header), followed by the available L1 product and channel selectors.
  Canonical and SCC-derived variants remain separate saved products.
- All channels in the selected L1 product, not a fixed hard-coded channel list.
- `range_corrected_signal`, `corrected_signal` and
  `signal_pre_background` when present as saved `(time, channel, altitude)`
  variables.
- Time–altitude heatmap and accompanying time-mean vertical profile.
- Hover inspection, interactive Plotly zoom, pan and PNG export.
- Excel-like **in-plot editor**: click the heatmap colorbar (or the small
  “Edit plot” button) to change robust percentiles (default P2–P98), auto/full
  color limits, manual min/max (scientific notation), palette and log10. Click
  a time/altitude axis label or the “Axes” tab to edit view ranges and reset
  zoom. Plotly also supports native on-plot editable text and interactive
  zoom/pan. Appearance updates occur in the browser without reopening NetCDF,
  and preserve the current Plotly zoom.
  Robust percentiles describe finite cells in the *display-reduced* matrix,
  not quantiles of the entire scientific NetCDF. Values above/below the color
  limits are saturated in the palette, never removed from the dataset.
  Log10 excludes zero/negative cells from the heatmap (without altering data)
  and manual limits in that mode are log10 units; the time-mean profile remains
  in the original linear signal units.
- Explicit visual aggregation (`mean` or `maximum`) to limit browser
  resolution. These displayed aggregate values are **not** original native
  RCS samples or new scientific retrievals.
- Large time gaps are represented by missing heatmap columns; NaNs are kept
  as missing values. The original file is never modified.
- The interface says “Explorer” rather than “Scientific Explorer”, omits
  redundant session explanations, and shows existing configured figure logos
  at the bottom when the referenced image files exist. Logos are resolved
  once from `visualization.logos` and served through an allowlisted route;
  the browser cannot request arbitrary local file paths.
- Other tabs are intentionally placeholders until later phases.

## Internal responsibilities

- `milgrau/web/catalog.py`: filename-only session discovery, UTC sorting and product IDs.
- `milgrau/web/rcs.py`: L1-only channel selection and bounded display reads.
- `milgrau/web/app.py`: FastAPI endpoints and local bundled frontend.
- `milgrau/web/static/`: independent HTML/CSS/JavaScript visual interface.
- `milgrau/cli/explorer.py`: main CLI entrypoint.

The backend accepts registered session/product identifiers, never arbitrary
paths from the browser. It serves data read-only and does not call LIBIDS,
LIPANCORA or LEBEAR processing. No six-hour publication windows are used:
scientific sessions remain continuous.

## Scope and validation

Phase 1 only supports saved L1 signals and does not perform retrieval,
calibration, atmospheric inference or L2 QA. Test with `pytest -q
tests/test_web_explorer.py` and inspect manually with a real L1 NetCDF,
particularly gaps, plots and memory use.
