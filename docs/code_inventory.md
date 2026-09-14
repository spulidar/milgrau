# MILGRAU code ownership inventory

This document is the P2 code-use baseline. Its purpose is not to freeze the current layout; it records why code exists so unused compatibility and accidental public API can be removed deliberately.

## Classification

Every retained module/symbol should fit one of these roles:

- **productive public API** — supported entry point or reusable contract intentionally exposed to users;
- **productive internal** — required by the current L0/L1/L2 processing path but not intended as a package-level API;
- **research / diagnostic API** — numerically or scientifically useful outside the productive policy, retained intentionally for validation/research;
- **optional UI / visualization** — presentation or interactive tooling that must not decide retrieval science;
- **compatibility** — temporary path with a named consumer and removal criterion;
- **unused** — no current consumer or justified role; remove it.

Compatibility with no named consumer is not a valid category.

## Package-level ownership

| Package / module group | Current role | Public surface / ownership decision |
| --- | --- | --- |
| `milgrau` | productive public API | Root package intentionally exposes only `__version__`. Subpackages are imported explicitly so `import milgrau` stays lightweight and has no plotting/scientific side effects. |
| `milgrau.cli` | productive public entry points | Console scripts in `pyproject.toml` are the supported CLI boundary. CLI modules translate arguments/results; they do not own science. |
| `milgrau.config` | productive public API | `load_config` is the root configuration loader. Stage-specific scientific validation belongs to `level0/config.py`, `level1/config.py`, `level2/config.py`, or visualization config. |
| `milgrau.io` | shared productive API | File/path/logging/input helpers. Existing package re-exports are retained for now; individual consumer audit remains part of P2. |
| `milgrau.level0` | productive processing API | `process_level_0` plus selected acquisition/netCDF/quality helpers. Detailed re-export necessity remains to be checked against external/docs consumers. |
| `milgrau.level1` | productive processing API | `process_level_1` and selected correction/PBL/tropopause helpers. Lower-level package re-exports are retained pending consumer audit. |
| `milgrau.level2` | productive API + explicit research API | Productive orchestration is backward KFS. Multi-mode KFS/gluing kernels remain available for validation/research but do not define productive policy. |
| `milgrau.physics` | shared scientific kernel API | Pure atmosphere physics shared across processing levels. No filesystem/config policy belongs here. |
| `milgrau.viz` | optional visualization API | Plotting only; must not feed retrieval decisions back into L0/L1/L2. |
| `milgrau.explorer` | optional interactive UI | Streamlit application and explorer helpers. Large mixed UI module is a P2 cohesion-review candidate, not a scientific owner. |
| `milgrau.incremental` | productive internal/shared | Currentness/reuse mechanics. Scientific identity checks are supplied by the relevant processing level. |
| `milgrau.operations` | productive internal/shared | Execution result/status/exit semantics shared by CLIs and stages. |
| `milgrau.provenance` | productive internal/shared | Reproducibility/provenance serialization and comparison. |
| `milgrau.scientific` | productive metadata | Versioned scientific identity; records method identity rather than implementing retrieval equations. |

## Level 0 modules

| Module | Role |
| --- | --- |
| `level0.common` | internal shared L0 helpers |
| `level0.config` | strict productive L0 configuration |
| `level0.inventory` | productive measurement inventory |
| `level0.libids` | productive L0 orchestration |
| `level0.netcdf` | productive L0 schema/materialization; large-file cohesion review candidate |
| `level0.processing` | productive signal/acquisition processing |
| `level0.quality` | productive acquisition QA |
| `level0.time` | small deterministic time classification helpers |

## Level 1 modules

| Module | Role |
| --- | --- |
| `level1.common` | internal shared L1 helpers |
| `level1.config` | strict productive L1 configuration; large-file cohesion review candidate |
| `level1.corrections` | productive instrumental corrections |
| `level1.diagnostics` | productive/diagnostic correction metadata |
| `level1.ingestion` | productive L0→L1 input preparation |
| `level1.lipancora` | productive L1 orchestration; large-file cohesion review candidate |
| `level1.pbl` | productive PBL algorithm |
| `level1.thermodynamics` | productive atmosphere-source selection/materialization |
| `level1.tropopause` | productive thermal tropopause calculation |

## Level 2 modules

The P1 cleanup removed `_retrieval_impl.py` and `level2.atmosphere`; no compatibility wrapper replaces either one.

| Module | Role |
| --- | --- |
| `level2.block_average` | productive internal generic block aggregation |
| `level2.cloud_screening` | research/diagnostic capability; not currently an enabled productive Rayleigh gate |
| `level2.completeness` | productive product-completeness/failure contract |
| `level2.config` | strict productive L2 configuration; only productive source of KFS/Rayleigh/gluing semantic settings |
| `level2.constants` | scientific/internal constants |
| `level2.contracts` | typed productive data/result contracts |
| `level2.dataset` | productive NetCDF/schema assembly; must describe science but not choose it; large-file review candidate |
| `level2.discovery` | productive Level 1 input discovery |
| `level2.gluing` | numerical gluing kernel; used productively through `signal_selection`, also retained as a lower-level research API |
| `level2.kfs` | numerical multi-mode Fernald/KFS kernel; backward/forward/two-sided research capability; productive caller passes explicit `backward` |
| `level2.lebear` | productive Level 2 file/orchestration boundary |
| `level2.molecular` | molecular/Rayleigh numerical physics and reference-search utilities |
| `level2.optical_retrieval` | productive Rayleigh calibration/QA + backward KFS block retrieval/aggregation |
| `level2.qa` | product QA/report statistics; no feedback into retrieval science |
| `level2.rayleigh_window` | productive physical-width→grid conversion |
| `level2.result_assembly` | productive block→typed-result assembly only |
| `level2.retrieval` | productive one-wavelength orchestration and canonical L1-atmosphere/PC-guard boundary |
| `level2.retrieval_input_qa` | productive supported-domain pre-QA |
| `level2.signal_selection` | productive blocking, AN/PC selection, gluing and single-channel fallback; large-file review candidate |
| `level2.time_window` | productive explicit time-window filtering |

### Level 2 package exports

The following distinction is intentional:

- **productive/public processing:** `process_level_2`, `process_single_level1_file`, completeness/result contracts, discovery and current cloud-screening API;
- **shared scientific/public helpers:** `calculate_molecular_profile`, `find_optimal_reference_altitude`;
- **research/numerical public API retained intentionally:** `fernald_inversion`, `kfs_inversion_monte_carlo`, `slide_glue_signals`, `propagate_glued_error`.

The research kernels are not evidence that productive L2 is two-sided. Productive direction is controlled by strict L2 configuration and the productive optical wrapper passes backward mode explicitly. P2 must decide whether these low-level symbols remain package-level exports long term or move to documented module-qualified research APIs; no removal should occur without an explicit API decision.

## IO / configuration / visualization modules

`milgrau.io` currently contains contracts, ERA5, filesystem, Licel, logging, paths, radiosonde and surface-weather modules. Their responsibilities are distinct; package-level re-export necessity remains to be checked symbol by symbol.

`milgrau.config` contains the root loader and station catalog resolver. Station/instrument truth belongs to the station catalog, not stage-local scientific defaults.

`milgrau.viz` contains visualization config, Level 2 QA plots, LIRACOS plots, quicklooks and shared style. `viz/level2_qa.py` is a large-file cohesion-review candidate, but size alone is not a reason to split it.

`milgrau.explorer/streamlit_app.py` is currently the largest Python UI module and is a P2 cohesion-review candidate. Optional UI imports must remain outside the scientific processing core.

## Compatibility inventory

Known Level 2 compatibility paths with no real consumer were removed in P1:

- `_retrieval_impl.py` — removed;
- `level2.atmosphere` alias — removed;
- `backward_retrieval.py` — removed;
- `scientific_policy.py` — removed.

No new Level 2 compatibility wrapper was introduced in their place. Compatibility outside Level 2 still requires explicit consumer/removal-criterion review before P2 can claim repository-wide completion.

## Automated guardrails introduced with this inventory

- package `__all__` entries are regression-tested to resolve to real bound symbols;
- wildcard imports under `milgrau/` are rejected by an AST regression;
- productive Level 2 scientific mappings cannot use local `.get(..., fallback)` semantics in the canonical retrieval/selection/QA path;
- Ruff is a development dependency with a deliberately neutral first rule set (`E4`, `E7`, `E9`, `F`) for import/name/dead-code-class findings.

Ruff is not yet a CI gate and a repository-wide clean Ruff/full-pytest run is **not** claimed at this stage.

## Next audit passes

1. Enumerate actual in-repository consumers of every package re-export and mark external/documented APIs explicitly.
2. Run Ruff and classify findings: fix true dead/unused code; suppress only deliberate API/re-export patterns with narrow rationale.
3. Search compatibility/deprecation names outside Level 2 and require a named consumer/removal criterion.
4. Review broad `except Exception` boundaries and keep only intentional orchestration/optional-diagnostic containment.
5. Audit large modules by responsibility, not line count: `explorer/streamlit_app.py`, `viz/level2_qa.py`, `level2/signal_selection.py`, `level2/dataset.py`, `level0/netcdf.py`, `level1/config.py`, `level2/kfs.py`, `level2/contracts.py`, `level1/lipancora.py`, and `config/station.py` are the current first-pass candidates.
6. Add CI only after the static/full-test baseline is known and reproducible.
