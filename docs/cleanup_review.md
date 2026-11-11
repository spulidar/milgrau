# MILGRAU cleanup review — evidence-based, non-destructive

This register is an **audit of possible obsolescence and unused code** encountered while documenting the user-facing scientific workflow. It is **not approval to delete code, records or science**. Preserve external API compatibility, historical optical results and instrument source provenance. A file not called by the current CLI may still be an intentionally supported scientific API.

## Findings from the Level 0 / operation review (11 November 2026)

| Candidate | Evidence from current `main` | Assessment | Suggested next action |
| --- | --- | --- | --- |
| `milgrau/level0/solar.py` | Defines `annotate_solar_context`, `solar_segment_table` and `solar_metadata`. Repository search found **no imports/calls outside their own definitions**. The production `process_session_group` calls `milgrau.level0.processing._annotate_solar_context`, which uses acquisition **midpoint** times; the alternate helper uses **start** times. | **Strong candidate: duplicate/unreferenced local helper.** Not equivalent near a day/night threshold. | Confirm downstream/external imports and test equivalence/differences; propose removing the unused module in a separately approved cleanup change, **without changing** production solar calculations. |
| `.github/workflows/ci.yml` branch list | Push trigger still lists `milgraumaxxing` and `integration/scc-atmosphere-20261111`, although GitHub branch listing currently contains **only `main`**. | **Confirmed stale branch trigger configuration.** | Remove two obsolete trigger names in a separate low-risk workflow cleanup, retain `main` and `pull_request` triggers; validate Actions still runs. |
| Canonical L0 `Processing_level` metadata | `build_level0_global_attributes` chooses a text label based on whether SCC mappings are available: `Level 0: Raw Licel to SCC-compatible NetCDF` versus `Level 0: Raw Licel NetCDF (SCC mapping unavailable)`. | **Wording concern, not dead code.** The canonical scientific L0 is SCC-independent, but its description changes with SCC mapping availability. | Propose stable, neutral canonical wording and separate interoperability metadata, check existing NetCDF consumers and scientific provenance tests. Metadata text changes should not be confused with numerical changes. |
| `milgrau/level2/cloud_screening.py` + disabled recipe | The distributed `inversion.cloud_screening.enabled: false` leaves preliminary cloud screening off. The functions `detect_anomalous_layer_mask` and `detect_reference_contamination` **are exported in `milgrau.level2.__init__` and exercised by several tests**. | **Intentionally inactive/public research functionality**, not demonstrated obsolete. | Keep for now; later decide explicitly whether to validate a scientifically defensible production classifier, retain an experimental API, or retire with a compatibility plan. Do not label it active cloud classification. |
| `milgrau/io/filesystem.py` manual quarantine | Raw discovery only classifies/filters; `quarantine_file`/`quarantine_files` are separately exported library functions with tests, and preserve a hash/JSON sidecar. | **Deliberate audit/recovery capability**, not automatic cleanup. | Retain; document separately from acquisition QA; never auto-delete rejected original Licel files. |
| Raw `dark` path classification | `classify_raw_file` currently classifies any candidate whose **full path contains `dark`**, ignoring case, as a dark acquisition. | **Potential scientific selection ambiguity**, not obsolete code. | Validate actual raw-tree naming with representative historical source paths before tightening this heuristic; add regression tests for false positives before changing behavior. |
| `milgrau/level0/header_cache.py` | Header parser cache includes a `file_type` argument that is explicitly discarded because classification is recalculated during filesystem discovery. Its signature is used by current callers. | **Possible small API cleanup**, not a scientific bug. | Only simplify if callers/compatibility permits; keep cache data and source-file identity intact. |
| `docs/scc_phase5_validation.md`, `docs/history/scc_integration_development_tracker.md` | Record superseded experiments, earlier branch plan and remote SCC provenance. User operation is now documented in `docs/scc_batch_operator_guide.md`. | **Historical evidence that should remain accessible.** | Preserve and clearly mark historical; revise misleading navigation, but do not erase evidence of previous experiments. |

## Evidence and constraints

- Existing searches use the current GitHub repository and code review. Search indexing and third-party imports cannot prove the absence of all external users.
- A path only appearing in a test or public `__init__` can still be an intentionally supported interface.
- Removing any data, journal, cache/provenance record, calibration or source file is **out of scope** of documentation cleanup.
- Never modify the productive solar-midpoint/segment semantics based only on unused helper discovery.
- Prior Level 0/1/2 hashes and SCC remote state must not be silently changed as part of documentation work.

## Cleanup decision queue

1. **High-confidence, low-risk:** remove obsolete non-existing CI branch trigger names — propose as standalone change after confirmation.
2. **High-confidence, with compatibility consideration:** retire the unused alternate `level0/solar.py` helper; preserve authoritative `physics/solar.py` and `level0/processing.py`.
3. **Wording/reference repair:** revise canonical L0 `Processing_level` to an SCC-neutral description while preserving a separate SCC availability indicator.
4. **Requires observational evidence:** classify dark acquisition by a safer filename/path policy, only after inspecting actual historical trees.
5. **Keep for now:** preliminary cloud screening, explicit quarantine helpers, header cache, and archived SCC development records.

## Safe process for a future cleanup batch

For each candidate: record all internal and public consumers; state whether it changes measured values, source selection or NetCDF metadata; run synthetic and relevant real-data regressions; obtain explicit approval for removal; and make a small reversible commit.

**Status: audit completed for these candidates; no code, instrument data, SCC remote record or scientific output deleted.**
