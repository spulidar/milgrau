# MILGRAU Explorer — read-only scientific inspection

The Explorer never executes LIBIDS, LIPANCORA or LEBEAR, never rewrites a
NetCDF, and does not create new scientific session identities. It is an
optional Streamlit application using Plotly for interactive views.

## Install and start

```bash
python -m pip install -e ".[explorer]"
milgrau-explorer
```

Supply the absolute or project-relative `config.yaml` and the processed-data
directory in the sidebar. Sessions are identified from
`processed_data/station/YYYY/MM/session_id/`. Variant products
(`day/night`, SCC, derived segment/UTC selections) are listed under the same
continuous source session and can be selected independently.

## Pages and performance

- **Visão geral**: fast filename-only session inventory, products and figures.
  Segment table metadata is read only for the selected session.
- **Level 0 / Level 1**: sampled time-height heatmaps and chunked mean profiles.
- **Level 2 Produtos ópticos**: interactive nominal/MC aerosol backscatter and
  extinction maps, individual block profiles with MC quantiles, wavelength
  comparison, and fixed-height time series.
- **Level 2 QA científico**: existing gluing, Rayleigh, scattering-ratio and
  KFS diagnostics, shown only when explicitly selected.
- **Metadados / QA Indicadores**: on-demand inspection of one file.
- **Galeria de figuras**: reuse existing L1/L2 image files without loading NetCDF.

Only the active page opens its dataset; file handles are closed upon rendering.
Display reductions select wavelength, regime, segment, and altitude before
sampling values. Plot dimensions are capped, but the source scientific arrays
are not decimated or changed. Cache lifetime for the file inventory is bounded.

## Scientific limits and terminology

All times are displayed in UTC unless a *local* session label is explicit.
Filtering solar regimes or segments does not merge sessions; excluded blocks
are represented as missing values at their real times, not interpolated.
The nominal optical fields follow the saved supported KFS retrieval.
MC averages and the 2.5–97.5% bands are **Monte Carlo distribution summaries**,
not an assertion that total uncertainty is exhausted by those intervals.
A finite scattering-ratio diagnostic alone is not evidence of a valid
aerosol inversion. Missing and unsupported cells remain NaN.

The Explorer does **not** recompute retrievals, calibrate detector saturation,
estimate aerosol lidar ratios, certify atmospheric quality, or publish data.

## Software architecture

- `catalog.py`: filename-only session and variant discovery, selected metadata
- `data.py`: lazy file access and bounded profile/heatmap helpers
- `plots_signals.py`: reusable L0/L1 visual presentations
- `plots_l2qa.py`: existing QA figure types
- `science.py`: lightweight scientific L2 array selectors and support masks
- `l2_products.py`: four L2 optical interaction modes
- `views.py`: on-demand per-level panels, metadata and diagnostics
- `streamlit_app.py`: sidebar, selected session and active-page orchestration

New scientific formulas and product contracts belong to their respective
processing modules, **not** the Explorer. 
