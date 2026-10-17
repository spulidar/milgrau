# MILGRAU Explorer — read-only scientific inspection

The Explorer never executes LIBIDS, LIPANCORA or LEBEAR, never rewrites a
NetCDF, and does not create new scientific session identities. It is an
optional Streamlit application using Plotly for interactive views.

## Install and start

```bash
python -m pip install -e ".[explorer]"
milgrau-explorer
```

Run \`milgrau-explorer\` from the project directory containing \`config.yaml\`.
The Explorer **automatically resolves** its read-only source from
\`directories.processed_data\` in that configuration and the referenced
station catalog (\`station.yaml\`). There is no sidebar or folder picker.

## Interface — measurements-inspired layout

The Explorer has a compact station header with a **UTC date selector**,
a horizontal navigation bar (\`RCS Maps\`, \`Atmospheric Profiles\`,
\`L2 Products\`, \`QA\`, \`Metadata\`), and a full-width interactive plotting area.
Only one view is rendered per request. The selected UTC date lists every
continuous scientific session that intersects that calendar day: a session
crossing midnight appears on both days **without** being split or merged.
Where several sessions intersect, an explicit session selector is shown.

The RCS channel selectbox includes every channel present in the selected
NetCDF, including analog/photon-counting and Raman channels when present.
The \`Atmospheric Profiles\` tab reads the saved L1 pressure and temperature
fields. No altitude-range widgets are shown; Plotly zoom/pan is used for
exploration of the full saved vertical span, with plot-only decimation for
large arrays. Product variants and Monte Carlo/quality selectors remain
specific to their respective views.

The fixed theme is implemented within \`theme.py\`; no
\`.streamlit/config.toml\` is needed. The application remains read-only.
The refresh button simply invalidates the short-lived file inventory cache.

## Pages and performance

- **RCS Maps**: sampled L0 or L1 interactive time-height heatmaps; the saved L1 figure includes its mean profile.
- **Atmospheric Profiles**: saved time-resolved L1 thermodynamics, reduced only for visual inspection.
- **L2 Products**: interactive nominal/MC aerosol backscatter and
  extinction maps, individual block profiles with MC quantiles, wavelength
  comparison, and fixed-height time series.
- **QA**: existing gluing, Rayleigh, scattering-ratio and
  KFS diagnostics, shown only when explicitly selected.
- **Metadata**: on-demand single-file NetCDF metadata, session inventory, and a gallery of already rendered L1/L2 figures.

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
- `navigation.py`: strict UTC calendar selection with continuous-session overlap
- `streamlit_app.py`: top header, horizontal navigation and active-page orchestration

New scientific formulas and product contracts belong to their respective
processing modules, **not** the Explorer. 
