# Candidate cleanup review — MILGRAU

This is a **non-destructive audit register**, not authorization to remove code or data. It records possible duplicate, inactive or historical material encountered during scientific documentation. Each candidate must be checked against all internal uses, tests, published APIs, historical data compatibility and external workflows before removal.

## Classification

- **Suspected duplicate / unused:** investigation required; do not delete based on text search.
- **Disabled or experimental:** retained behavior may be intentionally off; not automatically obsolete.
- **Historical evidence:** keep accessible even if not part of today's operational user guide.
- **Confirmed removable:** only after call-site, compatibility and regression review and explicit approval.

## Candidates identified while documenting A and B

| Candidate | Current evidence | Why inspect | Required review before action |
| --- | --- | --- | --- |
| `milgrau/level0/solar.py` versus `milgrau/level0/processing.py::_annotate_solar_context` | Both define solar-context/segment preparation; productive `process_session_group` calls the processing helper. The separate solar module has `annotate_solar_context` and `solar_segment_table` utilities. | Possible duplicated logic with different definitions of timing/segment boundaries. | Search all imports and third-party users; compare output for identical sessions; retain/reconcile the correct midpoint-based solar computation and tests. |
| `milgrau/level2/cloud_screening.py` and `inversion.cloud_screening.enabled: false` | Preliminary cloud-screening utilities are exported publicly and tested; distributed configuration disables this optional behavior. | Clarify whether this is an intentionally retained research API, future feature or unmaintained experimental path. | Trace productive call sites, tests, public import compatibility and scientific effect; do **not** describe it as a production cloud classifier or remove simply because disabled. |
| `docs/scc_phase5_validation.md` and `docs/history/scc_integration_development_tracker.md` | Documents preserve earlier SCC and experimental ERA5/GRIB work but are no longer the current operator guide. | Reduce user confusion from obsolete instructions while keeping traceability. | Keep archive/history; move links or add historical headers if needed, but do not delete scientific/evolution evidence. |
| `processing.quarantine_dir` and explicit filesystem quarantine | Raw discovery skips rejected/spurious candidates without moving them; a separate manual quarantine API retains audit sidecars. | Clarify whether a routine operator workflow should expose quarantine at all. | Verify all callers and intended retention rules; never delete or move historical source evidence as cleanup. |
| `Processing_level` text in Level 0 metadata | Some wording refers to SCC compatibility or SCC mapping availability. | Verify whether normal canonical L0 language implies SCC is part of the scientific product. | Check canonical versus SCC variants, metadata tests and downstream consumers before changing the field. |

## Suggested next audit workflow

1. Record candidate, call sites, actual present usage and scientific/provenance implications.
2. Categorize it as used, intentionally retained, inactive, redundant or genuinely unreachable.
3. Prepare a minimal cleanup diff on a separate change with tests and explicit acceptance.
4. Never remove raw observations, archive inputs, cache provenance, remote SCC journals or scientific historical records merely to reduce file count.
5. Revise this register and the tracker only after completing that evidence.

**Status:** exploratory candidates only; no code, API, raw data, SCC state or archive was deleted as part of documentation work.
