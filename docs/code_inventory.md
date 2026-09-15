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

## Package-level ownership

| Package | Role and current API decision |
| --- | --- |
| `milgrau` | Productive public root. Intentionally exposes only `__version__`; subpackages are explicit imports so root import stays light. |
| `milgrau.cli` | Public console-entry boundary defined by `pyproject.toml`; CLI code translates arguments/results and owns no science. |
| `milgrau.config` | Public configuration loader (`load_config`); stage-specific policy belongs to stage config modules. |
| `milgrau.io` | Shared IO API. Current direct re-exports are retained during P2 until the external/public surface is deliberately versioned; internals should import canonical owner modules directly. |
| `milgrau.level0` | Productive processing API centered on `process_level_0`, with direct helper re-exports. |
| `milgrau.level1` | Productive processing API. Re-exports now point directly to canonical owners rather than passing through `lipancora`. |
| `milgrau.level2` | Productive API plus explicit research kernels. Productive policy is backward KFS; multi-mode low-level KFS/gluing functions remain research/numerical API. |
| `milgrau.physics` | Shared pure scientific kernels; no filesystem/config-discovery policy. |
| `milgrau.viz` | Optional visualization API; never feeds decisions back into processing. |
| `milgrau.explorer` | Optional Streamlit UI. |
| `milgrau.incremental` | Shared productive currentness/reuse mechanics. |
| `milgrau.operations` | Shared execution result/status semantics. |
| `milgrau.provenance` | Shared reproducibility/provenance serialization and comparison. |
| `milgrau.scientific` | Versioned method identity/metadata, not a retrieval implementation. |

Package exports are tested to resolve to real bound objects. Wildcard imports under `milgrau/` are prohibited.

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

`level0.quality` intentionally catches malformed-data exceptions (`KeyError`, `TypeError`, `ValueError`) per acquisition group, but unexpected runtime/programming failures now propagate. A regression pins that behavior.

## Level 1 ownership

| Module | Role |
| --- | --- |
| `level1.common` | small internal shared helpers only; obsolete calibration/output-path compatibility helpers removed |
| `level1.config` | strict productive L1 recipe + station calibration resolution |
| `level1.corrections` | productive instrumental correction kernels |
| `level1.diagnostics` | productive correction metadata/diagnostics |
| `level1.ingestion` | productive L0→L1 input preparation |
| `level1.lipancora` | productive L1 orchestration; imports canonical owners directly |
| `level1.pbl` | productive PBL algorithm |
| `level1.thermodynamics` | productive atmosphere-source selection/materialization |
| `level1.tropopause` | productive thermal tropopause calculation |

Removed P2 compatibility residue:

- `level1.common.level1_output_path()` — one-line alias removed; orchestration now imports `milgrau.io.paths.level1_output_path` directly.
- `level1.common.get_channel_constant(..., logger)` — obsolete pre-resolver calibration helper removed; the unused `logger` existed only for call-site compatibility. Current calibration ownership is `resolve_channel_calibration()`.

The broad catch around each channel in `apply_all_physical_corrections()` is retained intentionally: one bad channel is recorded as a failed channel while valid channels may still form a Level 1 product. The outer `process_single_file()` catch is an execution/orchestration boundary.

## Level 2 ownership

P1 removed `_retrieval_impl.py`, `level2.atmosphere`, `backward_retrieval.py` and `scientific_policy.py`; no compatibility wrappers replaced them.

| Module | Role |
| --- | --- |
| `level2.block_average` | productive internal accepted-block aggregation |
| `level2.cloud_screening` | research/diagnostic capability; not an enabled productive Rayleigh gate |
| `level2.completeness` | productive multispectral completeness/failure contract |
| `level2.config` | strict productive L2 configuration; authoritative KFS/Rayleigh/gluing settings |
| `level2.constants` | scientific/internal constants |
| `level2.contracts` | typed productive data/result contracts |
| `level2.dataset` | productive NetCDF/schema assembly; describes but does not choose science |
| `level2.discovery` | productive Level 1 discovery |
| `level2.gluing` | numerical gluing kernel, used productively through `signal_selection`; also research API |
| `level2.kfs` | multi-mode Fernald/KFS numerical kernel; productive caller passes explicit `backward` |
| `level2.lebear` | productive file/orchestration boundary |
| `level2.molecular` | molecular/Rayleigh numerical physics and reference search |
| `level2.optical_retrieval` | productive Rayleigh QA/calibration + backward block retrieval/aggregation |
| `level2.qa` | QA orchestration/currentness; no feedback into retrieval |
| `level2.rayleigh_window` | productive physical-window→grid conversion |
| `level2.result_assembly` | productive block→typed-result assembly |
| `level2.retrieval` | productive one-wavelength orchestration and L1-atmosphere/PC-guard boundary |
| `level2.retrieval_input_qa` | productive supported-domain pre-QA |
| `level2.signal_selection` | productive blocking, AN/PC selection, gluing and fallback |
| `level2.time_window` | productive explicit time-window filtering |

The package-level research exports `fernald_inversion`, `kfs_inversion_monte_carlo`, `slide_glue_signals` and `propagate_glued_error` are retained intentionally for validation/research. Their availability does not change the productive backward-only contract.

`level2.lebear.attempt_wavelength()` intentionally contains failures per wavelength to support explicit partial-product diagnostics. The outer Level 2 catch is an orchestration boundary. These are not silent scientific fallbacks.

## Configuration/default ownership

Strict recipe resolvers are:

- `level0/config.py`
- `level1/config.py`
- `level2/config.py`

They may use structural absence sentinels while validating mappings, but may not invent recipe values through local `.get(..., scientific_literal_default)`. AST regressions now guard both the canonical productive L2 path and strict L0/L1/L2 resolver mappings.

The explicit Level 1 `neutral_with_warning` historical-calibration policy is not a hidden default: it must be configured, its correction values are forced to exactly zero, and its use is persisted/warned.

## Exception-boundary policy

Broad exceptions are audited by role rather than banned mechanically.

- **Retain** broad containment at outer CLI/file orchestration boundaries where failures are converted to explicit `ExecutionResult` records.
- **Retain** per-wavelength Level 2 containment because partial multispectral products have an explicit failure contract.
- **Retain** optional QA/UI containment where plotting failure cannot alter the saved scientific product.
- **Narrow/remove** broad catches inside scientific/config/data-quality helpers when they can hide programming/runtime defects.

P2 already narrowed Level 0 acquisition QA and station-config loading, with regressions proving unexpected `RuntimeError` propagates.

## Static and test guardrails

Current committed guardrails:

- every package `__all__` entry resolves to a bound object;
- Level 1 public helpers are pinned to canonical owner modules;
- removed Level 1 compatibility helpers are regression-pinned absent;
- wildcard imports under `milgrau/` are rejected;
- productive L2 scientific mappings may not use local semantic defaults;
- strict L0/L1/L2 config resolvers may not use semantic literal `.get` fallbacks;
- Ruff development gate uses `E4`, `E7`, `E9`, `F`, with only `E731` excluded because lambda-vs-def is stylistic rather than a correctness/dead-code rule.

The user-confirmed local baseline after the first Ruff cleanup is clean. Full repository pytest/CI is still a separate gate and must not be inferred from focused tests.

## Remaining P2 audit targets

- Finish deliberate public-surface decisions for `io`, `level0`, `level1`, `level2`, `physics`, `viz`; do not remove externally plausible convenience/research APIs accidentally.
- Audit compatibility/deprecation paths outside already-cleaned L1/L2 and require a named consumer/removal criterion.
- Search repository-wide for duplicate scientific equations/selection rules, not merely similar utility code.
- Finish broad-exception review in processing helpers; preserve intentional orchestration/optional-UI boundaries.
- Remove confirmed dead QA/display helpers. Current candidates in `viz/level2_qa.py` include `_legacy_ylim` and `_visual_scale_to_reference`; they require a dedicated safe edit because the plotting module is large.
- Review large files by cohesion, not line count: `explorer/streamlit_app.py`, `viz/level2_qa.py`, `level2/signal_selection.py`, `level2/dataset.py`, `level0/netcdf.py`, `level1/config.py`, `level2/kfs.py`, `level2/contracts.py`, `level1/lipancora.py`, `config/station.py`.
- Run full pytest in the reproducible dev environment, then add CI for Ruff + architecture guards + full tests.
