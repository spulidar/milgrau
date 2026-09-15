# MILGRAU code ownership inventory

This document is the P2 code-use baseline. Its purpose is to record why code exists so accidental API, compatibility residue, duplicate ownership and dead helpers can be removed deliberately.

## Classification

Every retained module/symbol must fit one current role:

- **productive public API** — supported processing entry point or reusable contract intentionally exposed to users;
- **productive internal** — required by the current L0/L1/L2 path but not intended as package-level API;
- **research / diagnostic API** — intentionally retained numerical/scientific capability used for validation or research;
- **optional UI / visualization** — presentation code that cannot decide retrieval science;
- **compatibility** — temporary path with a named consumer and a removal criterion;
- **unused** — no current consumer or justified role; remove it.

Compatibility without a named consumer is not a valid category.

## Package-level public surface

The root package intentionally exposes only `__version__`; subpackages are explicit imports so `import milgrau` stays light. The supported `__all__` surface of `milgrau.io`, `level0`, `level1`, `level2`, `physics`, and `viz` is now regression-pinned exactly in `tests/test_imports.py`. New or removed package-level exports therefore require a deliberate API review instead of appearing by accident.

Current decisions:

| Package | Role and API decision |
| --- | --- |
| `milgrau.cli` | Console-entry boundary from `pyproject.toml`; translates CLI arguments/results and owns no science. |
| `milgrau.config` | Public `load_config`; stage policy belongs in strict stage config modules. |
| `milgrau.io` | Shared reusable IO/contract/path/weather/radiosonde/Licel API; exact convenience exports are intentionally retained and pinned. |
| `milgrau.level0` | Productive L0 API centered on `process_level_0` plus documented reusable helpers. |
| `milgrau.level1` | Productive L1 API; exports point directly to their canonical owner modules. |
| `milgrau.level2` | Productive API plus explicit research/numerical kernels. Productive policy remains backward KFS. |
| `milgrau.physics` | Shared pure scientific kernels; no filesystem/config-discovery policy. |
| `milgrau.viz` | Optional presentation API; never feeds decisions back into retrieval. |
| `milgrau.explorer` | Optional Streamlit UI. |
| `milgrau.incremental` | Shared currentness/reuse mechanics. |
| `milgrau.operations` | Shared execution result/status semantics. |
| `milgrau.provenance` | Shared reproducibility/provenance serialization/comparison. |
| `milgrau.scientific` | Versioned method identity/metadata, not retrieval implementation. |

## Canonical scientific ownership

One productive scientific behavior should have one owner. Current canonical ownership is:

| Behavior | Canonical owner |
| --- | --- |
| standard atmosphere / geopotential conversion | `milgrau.physics.atmosphere` |
| Level 1 instrumental corrections | `milgrau.level1.corrections` |
| PBL retrieval | `milgrau.level1.pbl` |
| atmosphere source selection/materialization | `milgrau.level1.thermodynamics` |
| tropopause diagnostics | `milgrau.level1.tropopause` |
| strict L2 recipe | `milgrau.level2.config` |
| AN/PC gluing numerical kernel | `milgrau.level2.gluing` |
| productive AN/PC selection/gluing/fallback | `milgrau.level2.signal_selection` |
| molecular/Rayleigh physics and reference search | `milgrau.level2.molecular` |
| Fernald/KFS numerical kernel | `milgrau.level2.kfs` |
| productive Rayleigh QA + backward KFS orchestration | `milgrau.level2.optical_retrieval` |
| wavelength orchestration / L1 atmosphere boundary | `milgrau.level2.retrieval` |
| block→typed-result assembly | `milgrau.level2.result_assembly` |
| NetCDF/schema materialization | `milgrau.level2.dataset` |

P1 removed the duplicate `_retrieval_impl.py`, `level2.atmosphere`, `backward_retrieval.py`, and `scientific_policy.py`; no compatibility wrappers replaced them. The package-level `fernald_inversion`, `kfs_inversion_monte_carlo`, `slide_glue_signals`, and `propagate_glued_error` exports are retained intentionally as numerical/research APIs and do not define productive policy.

## Level 0 ownership

| Module | Role |
| --- | --- |
| `level0.common` | internal deterministic helpers |
| `level0.config` | strict productive L0 configuration |
| `level0.inventory` | productive measurement inventory |
| `level0.libids` | productive L0 orchestration |
| `level0.netcdf` | productive L0 schema/materialization; cohesion review candidate |
| `level0.processing` | productive acquisition/signal processing |
| `level0.quality` | productive acquisition QA |
| `level0.time` | deterministic time classification |

`level0.quality` contains expected malformed-data failures per acquisition group, but unexpected runtime/programming failures propagate.

## Level 1 ownership

| Module | Role |
| --- | --- |
| `level1.common` | small internal helpers only |
| `level1.config` | strict productive recipe + station calibration resolution |
| `level1.corrections` | productive instrumental correction kernels |
| `level1.diagnostics` | productive correction metadata/diagnostics |
| `level1.ingestion` | productive L0→L1 boundary |
| `level1.lipancora` | productive L1 orchestration |
| `level1.pbl` | productive PBL algorithm |
| `level1.thermodynamics` | productive atmosphere-source selection/materialization |
| `level1.tropopause` | productive thermal tropopause calculation |

Removed compatibility residue:

- `level1.common.level1_output_path()` — orchestration uses `milgrau.io.paths.level1_output_path` directly.
- `level1.common.get_channel_constant(..., logger)` — obsolete pre-resolver calibration helper; `resolve_channel_calibration()` owns current calibration resolution.

The broad catch around each channel in `apply_all_physical_corrections()` is intentional: one failed channel is represented explicitly while valid channels may still form a Level 1 product. Outer file processing is an execution boundary.

## Level 2 ownership

| Module | Role |
| --- | --- |
| `level2.block_average` | productive accepted-block aggregation |
| `level2.cloud_screening` | research/diagnostic capability; not enabled as productive Rayleigh gate |
| `level2.completeness` | productive multispectral completeness/failure contract |
| `level2.config` | strict productive L2 settings |
| `level2.constants` | scientific/internal constants |
| `level2.contracts` | typed productive data/result contracts |
| `level2.dataset` | productive schema assembly; describes but does not choose science |
| `level2.discovery` | productive Level 1 discovery |
| `level2.gluing` | numerical gluing kernel; productive use is through `signal_selection` |
| `level2.kfs` | multi-mode research/numerical kernel; productive caller explicitly passes `backward` |
| `level2.lebear` | productive file/orchestration boundary |
| `level2.molecular` | molecular/Rayleigh numerical physics/reference search |
| `level2.optical_retrieval` | productive Rayleigh calibration + backward block retrieval/aggregation |
| `level2.qa` | optional QA orchestration/currentness; no feedback into retrieval |
| `level2.rayleigh_window` | physical-window→grid conversion |
| `level2.result_assembly` | productive result construction |
| `level2.retrieval` | productive one-wavelength orchestration |
| `level2.retrieval_input_qa` | productive supported-domain pre-QA |
| `level2.signal_selection` | productive blocking/source selection/gluing/fallback |
| `level2.time_window` | productive time-window filtering |

`level2.lebear.attempt_wavelength()` intentionally contains failures per wavelength to support explicit partial-product diagnostics. Outer Level 2 containment is an execution boundary, not a silent scientific fallback.

## Compatibility decisions

- Configuration loader no longer creates old `site`, `hardware`, `physics.channels`, or similar compatibility views. Station-owned lidar-ratio climatology is intentionally materialized into the strict L2 recipe because the resolver consumes that numerical view; provenance remains station-owned.
- Historical/custom Licel files may lack valid channel `NShots` but carry the older file-level shot value. `_resolved_channel_shots()` gives channel `NShots` priority and uses the global value only as a named file-format compatibility fallback. Remove it only when those historical/custom files are explicitly unsupported or migrated.
- Removed Level 1/2 compatibility paths are guarded against reintroduction in both package and tests.

## Configuration/default ownership

Strict recipe resolvers are `level0/config.py`, `level1/config.py`, and `level2/config.py`. They may use structural absence sentinels during validation but may not invent scientific recipe values through local `.get(..., literal_default)`. AST regressions enforce this.

The explicit Level 1 `neutral_with_warning` historical-calibration policy is not a hidden default: it must be configured, correction values are forced to exactly zero, and use is persisted/warned.

## Exception-boundary policy

Broad exceptions are audited by role rather than mechanically banned.

- **Retain** at outer CLI/file orchestration boundaries where failures become explicit execution results.
- **Retain** per-wavelength L2 containment because partial multispectral products have an explicit failure contract.
- **Retain** per-channel L1 containment where failed-channel state is persisted rather than silently accepted.
- **Retain** filesystem action containment where the API returns explicit `ExecutionResult` failures.
- **Retain** optional QA/UI containment when presentation failure cannot alter the saved scientific product.
- **Narrow/remove** inside scientific/config/data parsers when broad catches can hide implementation defects.

Completed narrowing includes Level 0 acquisition QA, station-config loading, Level 1 helpers, and Licel parsing. Licel inventory/group parsing now catches expected `OSError`/`ValueError` but explicitly allows unexpected `RuntimeError` to propagate; tests pin that boundary.

## Static/test/CI guardrails

Committed guardrails now include:

- exact package `__all__` surfaces for `io`, `level0`, `level1`, `level2`, `physics`, `viz`;
- root API restricted to `__version__`;
- Level 1 exports pinned to canonical owners;
- removed compatibility helpers/paths regression-pinned absent;
- no wildcard imports under `milgrau/`;
- no local semantic-default fallbacks in strict stage resolvers/productive L2 mappings;
- Ruff correctness/dead-code gate `E4,E7,E9,F` (`E731` excluded as style-only);
- full pytest baseline;
- GitHub Actions on push to `new-architecture` and pull requests.

Local baseline before the latest two Licel exception tests: Ruff clean, **347 passed / 0 failed** on Windows/Python 3.14. GitHub Actions run `34916492050` on `e0bed288e1d4829f0ffaa61647e1257b9b0ca5ab` passed Ruff and all pytest jobs on Ubuntu/Windows × Python 3.12/3.14.

NumPy 2.5.3 + published netCDF4 1.7.4 remains a supported environment. The visible `ndarray.shape` deprecation is an upstream netCDF4 write-path warning; it is tracked rather than hidden or worked around by pinning NumPy backwards.

## Remaining P2 audit targets

- Remove confirmed definition-only QA helpers `viz/level2_qa.py::_legacy_ylim` and `_visual_scale_to_reference`; `_legacy_scale_factor` is active and stays.
- Finish a repository-wide duplicate-science review and record any newly found conflict; current productive owner table must stay one-owner-per-behavior.
- Complete the final broad-exception classification; narrow only helper/parser catches that can hide defects.
- Review large files by cohesion rather than line count: `explorer/streamlit_app.py`, `viz/level2_qa.py`, `level2/signal_selection.py`, `level2/dataset.py`, `level0/netcdf.py`, `level1/config.py`, `level2/kfs.py`, `level2/contracts.py`, `level1/lipancora.py`, `config/station.py`.
- Rerun the same CI workflow after final P2 cleanup; mark P2 complete only on a green final HEAD.
