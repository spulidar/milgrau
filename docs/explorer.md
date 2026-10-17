# MILGRAU Explorer — local, read-only

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

## Interactive scientific views

- One compact session selector in the black header, listing continuous
  scientific acquisitions newest-first in UTC. A white navigation toolbar
  displays the view tabs, wavelength/channel menu and the signal selector
  only when more than one signal is available. There is no product dropdown.
  The RCS view opens the canonical `SESSION_L1.nc` automatically, and shows a
  clear missing-L1 message rather than silently picking a derivative if no
  canonical L1 is available.
- All channels in the selected L1 product, not a fixed hard-coded channel list.
- `range_corrected_signal`, `corrected_signal` and
  `signal_pre_background` when present as saved `(time, channel, altitude)`
  variables.
- Time–altitude heatmap and accompanying time-mean vertical profile. Plot title,
  panel heading and mean-profile x-axis update with the selected saved L1
  variable: range_corrected_signal, corrected_signal or signal_pre_background.
  An uncorrected/pre-background signal is never labeled as RCS.
- Hover inspection, interactive Plotly zoom, pan and PNG export.
- Excel-like **in-plot editor**: click the heatmap colorbar (or the compact
  “Plot options” button in the figure header) to change robust percentiles (default P2–P98), auto/full
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
- Display cells use fixed mean aggregation to bound browser memory. There is
  no user-facing “Display reduction” selection, because it is an implementation
  detail and not a scientific processing algorithm. Displayed aggregated cells
  are **not** original native RCS samples or new scientific retrievals.
- Large time gaps are represented by missing heatmap columns; NaNs are kept
  as missing values. The original file is never modified.
- The header and white toolbar are visually modeled on the Measurements
  index page, without copying the public website's publication-window
  semantics or static-image rendering. The interface says “Explorer”, removes
  the redundant session banner and does not show footer logos.
- Figure titles and explicit axes (UTC, altitude, RCS, mean profile) are
  visible and editable; plot controls no longer overlay the heatmap.
- **Overview** mirrors the summary in `milgrau-inspect`: session start and end,
  local and UTC times, duration, recorded solar segment sequence, channels,
  figures and availability of canonical L0/L1/L2 files.
- **Atmospheric Profiles** plots saved L1
  `Atmospheric_Temperature_K`/`Atmospheric_Pressure_hPa`, selected by
  `atmosphere_time`, with recorded source when available.
- **L2 Products** plots saved `aerosol_backscatter_mean`,
  `aerosol_extinction_mean` or `temporal_support_fraction` by wavelength.
  No retrieval is performed; the view explicitly reports missing L2.
- **Scientific QA** visualizes stored `background_estimate`,
  `background_outlier_fraction`, saturation/dead-time and a few saved
  L2 retrieval/gluing flags as time series. These are recorded diagnostic
  values, not new quality judgments.
- **Metadata** exposes dimensions, coordinate bounds, names/shapes/units
  of variables, and global attributes for canonical L0/L1/L2. Large
  signal arrays are not loaded to build this inventory.
- Views load only on demand and use bounded profiles/QA; data endpoints
  retain canonical product-path checks and never accept filesystem paths.
  Missing inputs produce explicit unavailable states instead of fake plots.

## Internal responsibilities

- `milgrau/explorer/catalog.py`: filename-only session discovery, UTC sorting and product IDs.
- `milgrau/explorer/rcs.py`: L1-only channel selection and bounded display reads.
- `milgrau/explorer/views.py`: inspect-compatible session summary and bounded
  metadata, atmospheric, optical and QA read-only endpoints.
- `milgrau/explorer/static/js/views.js`: tab-specific UI and interactive Plotly profiles.
- `milgrau/explorer/app.py`: FastAPI endpoints and local bundled frontend.
- `milgrau/explorer/static/`: independent HTML/CSS/JavaScript visual interface.
- `milgrau/cli/explorer.py`: main CLI entrypoint.

The backend accepts registered session/product identifiers, never arbitrary
paths from the browser. It serves data read-only and does not call LIBIDS,
LIPANCORA or LEBEAR processing. No six-hour publication windows are used:
scientific sessions remain continuous.

## Scope and validation

All tabs display existing saved values and do not perform retrieval,
calibration, atmospheric inference or new QA decisions. Test with
`pytest -q tests/test_web_explorer.py tests/test_web_views.py`,
and inspect manually with real L1/L2 NetCDF products, particularly
axis labels, scientific units, gaps, zoom and memory use.
