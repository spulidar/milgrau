# MILGRAU cleanup review — evidence-based, non-destructive

This register is an **audit of possible obsolescence and unused code** encountered while documenting the user-facing scientific workflow. It is **not approval to delete code, records or science**. Preserve external API compatibility, historical optical results and instrument source provenance. A file not called by the current CLI may still be an intentionally supported scientific API.

## Findings from the Level 0 / operation review (11 November 2026)

| Candidate | Evidence from current `main` | Assessment | Suggested next action |
| --- | --- | --- | --- |
| `milgrau/level0/solar.py` | No repository imports/callers or public `level0.__init__` export found. The alternate uses acquisition **start** while the productive `processing._annotate_solar_context` uses **midpoint** and `physics.solar`. | **Retired (user-approved)**. This unused alternate was removed; canonical solar calculations remain unchanged. | Retain historical code in Git; monitor any third-party imports that were not visible to repository search. |
| `.github/workflows/ci.yml` branch list | Push trigger named two retired branches. Repository branch listing contains **only `main`**. | **Cleaned (user-approved)**: pushes target `main`; pull-request checks remain. | Confirm GitHub Actions passes on the cleanup commit. |
| Canonical L0 `Processing_level` metadata | `build_level0_global_attributes` chooses a text label based on whether SCC mappings are available: `Level 0: Raw Licel to SCC-compatible NetCDF` versus `Level 0: Raw Licel NetCDF (SCC mapping unavailable)`. | **Wording concern, not dead code.** The canonical scientific L0 is SCC-independent, but its description changes with SCC mapping availability. | Propose stable, neutral canonical wording and separate interoperability metadata, check existing NetCDF consumers and scientific provenance tests. Metadata text changes should not be confused with numerical changes. |
| `milgrau/level2/cloud_screening.py` + disabled recipe | The distributed `inversion.cloud_screening.enabled: false` leaves preliminary cloud screening off. The functions `detect_anomalous_layer_mask` and `detect_reference_contamination` **are exported in `milgrau.level2.__init__` and exercised by several tests**. | **Intentionally inactive/public research functionality**, not demonstrated obsolete. | Keep for now; later decide explicitly whether to validate a scientifically defensible production classifier, retain an experimental API, or retire with a compatibility plan. Do not label it active cloud classification. |
| `milgrau/io/filesystem.py` manual quarantine | Raw discovery only classifies/filters; `quarantine_file`/`quarantine_files` are separately exported library functions with tests, and preserve a hash/JSON sidecar. | **Deliberate audit/recovery capability**, not automatic cleanup. | Retain; document separately from acquisition QA; never auto-delete rejected original Licel files. |
| Raw `dark` path classification | `classify_raw_file` currently classifies any candidate whose **full path contains `dark`**, ignoring case, as a dark acquisition. | **Potential scientific selection ambiguity**, not obsolete code. | Validate actual raw-tree naming with representative historical source paths before tightening this heuristic; add regression tests for false positives before changing behavior. |
| `milgrau/level0/header_cache.py` | Header parser cache includes a `file_type` argument that is explicitly discarded because classification is recalculated during filesystem discovery. Its signature is used by current callers. | **Possible small API cleanup**, not a scientific bug. | Only simplify if callers/compatibility permits; keep cache data and source-file identity intact. |
| `docs/history/scc_phase5_validation.md`, `docs/history/scc_integration_merge_plan.md`, and `docs/history/scc_integration_development_tracker.md` | Superseded experiments and integration plans moved out of the current operator documentation. | **Archived (user-approved)**, preserving original review evidence and source history. | Use current `docs/scc_batch_operator_guide.md` and scientific tracker for operation and outstanding validation; do not treat archived plan as live guidance. |

## Evidence and constraints

- Existing searches use the current GitHub repository and code review. Search indexing and third-party imports cannot prove the absence of all external users.
- A path only appearing in a test or public `__init__` can still be an intentionally supported interface.
- Removing any data, journal, cache/provenance record, calibration or source file is **out of scope** of documentation cleanup.
- Never modify the productive solar-midpoint/segment semantics based only on unused helper discovery.
- Prior Level 0/1/2 hashes and SCC remote state must not be silently changed as part of documentation work.

## Cleanup decision queue

1. **Completed:** simplify CI push branches to `main` while retaining pull-request checks.
2. **Completed:** retire unused `level0/solar.py` without modifying `physics/solar.py` or `level0/processing.py`; third-party compatibility cannot be exhaustively proven.
3. **Completed:** archive obsolete SCC phase/merge plans and consolidate duplicated configuration guidance into `configuration_reference.md`.
4. **Wording/reference repair:** revise canonical L0 `Processing_level` to an SCC-neutral description while preserving a separate SCC availability indicator.
5. **Requires observational evidence:** classify dark acquisition by a safer filename/path policy, only after inspecting actual historical trees.
6. **Keep for now:** preliminary cloud screening, explicit quarantine helpers, header cache, and archived SCC development records.

## Safe process for a future cleanup batch

For each candidate: record all internal and public consumers; state whether it changes measured values, source selection or NetCDF metadata; run synthetic and relevant real-data regressions; obtain explicit approval for removal; and make a small reversible commit.

**Status: approved cleanup removed one unused solar helper, two stale CI push branch names and a duplicate configuration document; two obsolete SCC planning documents were moved into `docs/history/`. No productive scientific calculation, original observation, SCC remote record or NetCDF output was deleted.**
