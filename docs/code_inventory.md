# MILGRAU code ownership inventory

This document is the P2 code-use baseline. It records why code exists so accidental API, compatibility residue, duplicate ownership, hidden scientific defaults and dead helpers can be removed deliberately.

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

The root package intentionally exposes only `__version__`; subpackages are explicit imports so `import milgrau` stays light. The supported `__all__` surface of `milgrau.io`, `level0`, `level1`, `level2`, `physics`, and `viz` is regression-pinned exactly in `tests/test_imports.py`. New or removed package-level exports therefore require deliberate API review.

| Package | Role and API decision |
| --- | --- |
| `milgrau.cli` | Console-entry boundary from `pyproject.toml`; translates CLI arguments/results and owns no science. |
| `milgrau.config` | Public `load_config`; stage policy belongs in strict stage config modules. |
| `milgrau.io` | Shared reusable IO/contract/path/weather/radiosonde/Licel API; exact convenience exports are intentional. |
| `milgrau.level0` | Productive L0 API centered on `process_level_0` plus documented reusable helpers. |
| `milgrau.level1` | Productive L1 API; exports point directly to canonical owner modules. |
| `milgrau.level2` | Productive API plus explicit research/numerical kernels; productive policy remains backward KFS. |
| `milgrau.physics` | Shared pure scientific kernels; no filesystem/config-discovery policy. |
| `milgrau.viz` | Optional presentation API; never feeds decisions back into retrieval. |
| `milgrau.explorer` | Optional Streamlit UI. |
| `milgrau.incremental` | Shared currentness/reuse mechanics. |
| `milgrau.operations` | Shared execution result/status semantics. |
| `milgrau.provenance` | Shared reproducibility/provenance serialization/comparison. |
| `milgrau.scientific` | Versioned method identity/metadata, not retrieval implementation. |

## Canonical scientific ownership

Repository-wide P2 review found no second productive implementation for the behaviors below. Research kernels may expose extra modes, but productive policy has one owner.

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

P1 removed `_retrieval_impl.py`, `level2.atmosphere`, `backward_retrieval.py`, and `scientific_policy.py`; no compatibility wrappers replaced them. Package-level `fernald_inversion`, `kfs_inversion_monte_carlo`, `slide_glue_signals`, and `propagate_glued_error` remain intentionally as numerical/research APIs and do not define productive policy.

The current gluing selector is residual-based. P2 removed `inversion.gluing.gaussian_threshold` because it was an inert historical knob: it reached diagnostics but did not participate in acceptance/ranking. Old configs containing it now fail explicitly instead of implying that the setting affects science. Active acceptance/ranking inputs remain explicit in `level2.gluing`; the score constants themselves are a P3 documentation/versioning task.

## Level 0 ownership

| Module | Role |
| --- | --- |
| `level0.common` | internal deterministic helpers |
| `level0.config` | strict productive L0 configuration |
| `level0.inventory` | productive measurement inventory |
| `level0.libids` | productive L0 orchestration |
| `level0.netcdf` | productive L0 schema/materialization |
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

The broad catch around each channel in the correction orchestrator is intentional: one failed channel is represented explicitly while valid channels may still form a Level 1 product. In contrast, diagnostic reduction helpers inside `level1.corrections` now only contain expected conversion failures (`TypeError`, `ValueError`, `OverflowError`); unexpected runtime defects propagate and are regression-tested.

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

`level2.lebear.attempt_wavelength()` intentionally contains failures per wavelength to support explicit partial-product diagnostics. The outer file boundary converts failures into `ExecutionResult`; neither catch supplies alternate retrieval science.

## Compatibility decisions

- Configuration loader no longer creates old `site`, `hardware`, `physics.channels`, or similar compatibility views. Station-owned lidar-ratio climatology is intentionally materialized into the strict L2 recipe because the resolver consumes that numerical view; provenance remains station-owned.
- Historical/custom Licel files may lack valid channel `NShots` but carry the older file-level shot value. `_resolved_channel_shots()` gives channel `NShots` priority and uses the global value only as a named file-format compatibility fallback. Remove it only when those historical/custom files are explicitly unsupported or migrated.
- Removed Level 1/2 compatibility paths are guarded against reintroduction in both package and tests.

No other unnamed compatibility path was found in the final P2 audit.

## Configuration/default ownership

Strict recipe resolvers are `level0/config.py`, `level1/config.py`, and `level2/config.py`. They may use structural absence sentinels during validation but may not invent scientific recipe values through local `.get(..., literal_default)`. AST regressions enforce this.

The explicit Level 1 `neutral_with_warning` historical-calibration policy is not a hidden default: it must be configured, correction values are forced to exactly zero, and use is persisted/warned.

## Exception-boundary policy — final P2 classification

Broad exceptions are retained only where containment is itself part of the API contract:

- outer CLI/file orchestration, where failures become explicit execution results;
- per-wavelength L2 handling, where partial multispectral products have a failure contract;
- per-channel L1 orchestration, where failed-channel state is persisted;
- filesystem mutation boundaries (`quarantine_file`, `delete_file`), which return explicit failed `ExecutionResult` objects;
- optional QA/UI presentation, whose failure cannot alter the already-saved scientific product;
- product-currentness/integrity checks, where unreadable/invalid output is conservatively classified as not current.

Broad catches were removed/narrowed where they could hide implementation defects:

- Level 0 acquisition QA and station config;
- Licel parsing (`ValueError`/`OSError` only for malformed/IO input; unexpected `RuntimeError` propagates);
- Open-Meteo cache/retry handling: retries only expected IO/payload failures, while unexpected runtime failures propagate;
- Level 1 diagnostic min/max reduction helpers;
- raw-tree path resolution: only expected `OSError`/`RuntimeError` resolution failures are contained.

Optional plotting helpers such as `level2_qa` deliberately remain presentation-tolerant. Their broad catches are not used to select, modify, fill or validate scientific retrieval values.

## Dead-code result

P2 removed confirmed dead/obsolete code rather than preserving it for compatibility:

- Level 2 monolith/compatibility modules listed above;
- obsolete Level 1 wrappers;
- Ruff-discovered unused imports;
- `viz.level2_qa._legacy_ylim` and `_visual_scale_to_reference`, both definition-only;
- inert `gaussian_threshold` configuration/API plumbing.

`viz.level2_qa._legacy_scale_factor` remains because the gluing QA plot actively uses it as a display-only fallback when operational coefficients are unavailable.

## Large-file cohesion review

P2 reviewed the large-file list by responsibility rather than line count. No split is required merely to reduce LOC:

- `explorer/streamlit_app.py` remains one optional UI application boundary;
- `viz/level2_qa.py` remains one presentation domain and no longer contains the confirmed dead helpers;
- `level2/signal_selection.py` remains cohesive around block preparation/source selection/gluing/fallback;
- `level2/dataset.py` and `level0/netcdf.py` remain schema/materialization boundaries;
- `level1/config.py` remains the strict L1 recipe/calibration resolver;
- `level2/kfs.py` remains the numerical KFS research kernel;
- `level2/contracts.py` remains typed Level 2 contracts;
- `level1/lipancora.py` remains L1 file orchestration;
- `config/station.py` remains station-catalog resolution/history ownership.

Future splits should be triggered by a real second responsibility, not a line-count threshold.

## Static/test/CI guardrails

Committed guardrails include:

- exact package `__all__` surfaces for `io`, `level0`, `level1`, `level2`, `physics`, `viz`;
- root API restricted to `__version__`;
- Level 1 exports pinned to canonical owners;
- removed compatibility helpers/paths regression-pinned absent;
- no wildcard imports under `milgrau/`;
- no local semantic-default fallbacks in strict stage resolvers/productive L2 mappings;
- Ruff correctness/dead-code gate `E4,E7,E9,F` (`E731` excluded as style-only);
- full pytest baseline;
- GitHub Actions on pushes to `new-architecture` and pull requests, with Ruff plus pytest on Ubuntu/Windows × Python 3.12/3.14.

The local baseline reached **347 passed / 0 failed** before the final boundary-regression tests were added. CI repeatedly remained green through the P2 cleanup sequence. NumPy 2.5.3 + published netCDF4 1.7.4 is a supported environment; the visible `ndarray.shape` deprecation is a known upstream netCDF4 write-path warning and is not hidden by pinning NumPy backwards or filtering warnings.

## P2 closure status

The code-use, ownership, duplicate-science, dead-helper, semantic-default and exception-boundary audits are complete. The only closure condition is a green CI run on the final tracker/documentation HEAD. Branch protection/required checks remains an optional repository-policy follow-up rather than a scientific/code-quality blocker.

The next engineering/scientific phase is P3: make the Level 2 product schema and FAIR metadata self-describing without changing the already validated backward-KFS baseline as collateral work.
