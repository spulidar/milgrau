# Canonical code ownership

This inventory lists the sole current owner of each production responsibility.
Superseded implementations are not compatibility paths and remain available
only through Git history.

## Command-line pipelines

| Command | CLI owner | Processing owner |
| --- | --- | --- |
| `milgrau-libids` | `milgrau.cli.libids` | `milgrau.level0.processing` |
| `milgrau-lipancora` | `milgrau.cli.lipancora` | `milgrau.level1.lipancora` |
| `milgrau-liracos` | `milgrau.cli.liracos` | `milgrau.viz.liracos` |
| `milgrau-lebear` | `milgrau.cli.lebear` | `milgrau.level2.lebear` |
| `milgrau-explorer` | `milgrau.cli.explorer` | `milgrau.explorer.streamlit_app` |

## Shared infrastructure

| Responsibility | Canonical owner |
| --- | --- |
| configuration loading/resolution | `milgrau.config` |
| dated station/instrument resolution | `milgrau.config.station` |
| paths and measurement IDs | `milgrau.io.paths` |
| Level 0/1/current Level 2 contract entrypoints | `milgrau.io.contracts` |
| structured execution results | `milgrau.operations` |
| incremental currentness | `milgrau.incremental` |
| provenance and source-content identity | `milgrau.provenance` |

## Level 0

| Responsibility | Canonical owner |
| --- | --- |
| Licel parsing | `milgrau.io.licel` |
| measurement grouping | `milgrau.level0.inventory` |
| acquisition filtering | `milgrau.level0.filtering` |
| SCC-compatible NetCDF construction | `milgrau.level0.netcdf` |
| end-to-end orchestration | `milgrau.level0.processing` |

## Level 1

| Responsibility | Canonical owner |
| --- | --- |
| Level 0 ingestion/canonicalization | `milgrau.level1.ingestion` |
| instrumental corrections | `milgrau.level1.corrections` |
| background diagnostics and uncertainty | `milgrau.level1.background` |
| radiosonde/ERA5/USSA76 atmosphere | `milgrau.level1.thermodynamics` |
| PBL/tropopause diagnostics | `milgrau.level1.pbl`, `milgrau.level1.diagnostics` |
| NetCDF assembly/orchestration | `milgrau.level1.lipancora` |

## Level 2

| Responsibility | Canonical owner |
| --- | --- |
| strict recipe parsing | `milgrau.level2.config` |
| wavelength/channel discovery | `milgrau.level2.discovery` |
| block construction and source selection | `milgrau.level2.signal_selection` |
| analog/photon-counting gluing | `milgrau.level2.gluing` |
| signal/gluing/molecular preparation | `milgrau.level2.retrieval` |
| molecular atmosphere and signal | `milgrau.level2.molecular` |
| robust background and Rayleigh candidates | `milgrau.level2.rayleigh_candidates` |
| progressive vertical grid | `milgrau.level2.adaptive_grid` |
| high-column candidate catalogue | `milgrau.level2.high_column` |
| prioritized reference selection | `milgrau.level2.high_column_selector` |
| Klett–Fernald–Sasano kernel | `milgrau.level2.kfs` |
| two-sided single-profile retrieval | `milgrau.level2.two_sided_retrieval` |
| selection-aware Monte Carlo | `milgrau.level2.uncertainty_mc` |
| multispectral Level 2 dataset assembly | `milgrau.level2.level2_dataset` |
| current NetCDF contract | `milgrau.level2.level2_schema` |
| file/batch orchestration | `milgrau.level2.lebear` |
| QA orchestration | `milgrau.level2.qa` |
| scientific QA figures | `milgrau.viz.level2_qa` |
| small QA display helpers | `milgrau.viz.profile_helpers` |

## Scientific validation

Analytical and synthetic validation belongs in `tests/`. A numerical helper may
remain in `milgrau.level2` only when it is shared by the productive retrieval.
One-off campaign comparison CLIs and alternate retrieval assemblies are not
installed as parallel products.

## Public compatibility rule

- A productive scientific meaning has one owner.
- Removed product generations are not imported through aliases or wrappers.
- NetCDF provenance identifies software version and exact source content.
- Visualization consumes product variables but does not own scientific
  thresholds, masking or retrieval decisions.
