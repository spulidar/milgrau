# MILGRAU scientific products and measurements publication

MILGRAU is the sole source of truth for instrument/station metadata, scientific
configuration, the continuous session identity, L0/L1/L2, quality control,
generic Matplotlib renderers, and project path utilities. A scientific session
uses `station_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ` and **never** splits on an
HTML/site publication boundary. Exact profile timestamps remain in L1 NetCDF.

The separate `spulidar/measurements` repository owns only station-local
public views (00–06, 06–12, 12–18, 18–24), website-only WEBPs,
sidecar provenance, publication state, public HTML and eventual cloud upload.
It must consume `milgrau.config.loader.load_config`,
`milgrau.io.paths`, and `milgrau.viz`; do not duplicate configuration,
station definitions or science in this repository.

A single scientific session can appear in multiple
`YYYYMMDD_station_HH` publication windows; multiple sessions can contribute
to the same window. Such data are concatenated **in memory for visualization**
only. Gaps remain gaps; overlapping timestamps fail; scientific NetCDFs
are not split or combined.

## One Python environment

In the shared workspace, with Python 3.12+:

```bash
cd milgrau
pip install -e ".[web]"
# Or for full development:
pip install -e ".[dev,explorer,web]"
cd ../measurements
python update-site.py --stage-only --publish-date 20250510
```

The `explorer` extra is intentionally empty: FastAPI, Plotly and Uvicorn
already belong to core. `web` adds `boto3`; the publisher stays in the
`measurements` checkout. No second environment or measurements-specific
requirements/YAML is needed in publication v2.

Workspace root should contain `config.yaml`, `station.yaml`,
`01-data/`, `02-processed_data/`, `03-site-products/`, plus sibling
`milgrau/` and `measurements/` checkouts. The MILGRAU loader accepts an
explicit config path or `MILGRAU_CONFIG`. Paths are resolved relative to
the loaded MILGRAU config file's directory, not a hardcoded repository root.

Selected visualization logos must be resolvable relative to the MILGRAU
visualization root. Their bytes and relevant renderer source code contribute
to publication-v2 staging fingerprints.

## Deployment isolation

As long as `measurements/publication-v2` is under development:
`measurements/main` is the frozen live site and current R2 objects must
remain unchanged. Generate new publication staging under
`03-site-products/station/YYYY/MM/YYYYMMDD` only. Do not upload,
delete, bootstrap production state, push generated HTML, or reprocess
`01-data`. A later, separately approved cutover will rebuild and
validate history, publish to separate staging, switch production and only
after validation retire legacy storage.
