# Phase 5 — SCC regression and validation evidence

The SCC integration is still experimental on `scc-integration-refactor`.
This document separates evidence obtainable in GitHub Actions from
validation that requires real scientific input files or explicit permission
to operate the remote SCC service.

## 1. Automated regression matrix

All tests in this section use local temporary files and simulated/mock
SCC clients; they do **not** authorize remote POST or DELETE.

| Requirement | Automated evidence |
|---|---|
| Continuous scientific session, midnight/month boundaries | `tests/test_inventory.py`, `tests/test_scc_staging.py` |
| Session-level QA and dark-current association | `tests/test_level0_quality.py`, `tests/test_inventory.py` (optimized vs exhaustive assignment) |
| Scientific incremental processing independent of SCC | `tests/test_incremental.py`, `tests/test_scc_staging.py` |
| Optional SCC error cannot invalidate canonical L0 | `tests/test_level0_processing_results.py`, `tests/test_scc_staging.py` |
| SCC-only re-export cannot overwrite canonical L0 or accepted session inputs | `tests/test_scc_staging.py` |
| LIPANCORA observed-sounding attachment and fail/skip policies | `tests/test_scc_sounding_finalize.py` |
| Strict input filename, ID, scientific provenance and checksums | `tests/test_scc_client.py`, `tests/test_scc_staging.py`, `tests/test_scc_backup.py` |
| New upload and replacement with explicit waiver | `tests/test_scc_workflow.py`, `tests/test_scc_batch.py` |
| Existing SCC ID without replacement authorization | `tests/test_scc_client.py`, `tests/test_scc_workflow.py` |
| Uncertain DELETE: recover once with current-revision waiver or verified backup | `tests/test_scc_workflow.py` |
| Unknown/incorrect upload confirmation remains uncertain, no retry | `tests/test_scc_workflow.py`, `tests/test_scc_client.py` |
| Mixed valid/invalid batch and fail-stop on operational errors | `tests/test_scc_batch.py` |
| Remote status independent of original staged inputs | `tests/test_scc_client.py`, `tests/test_scc_bulk_read.py` |
| External completed results without the original journal | `tests/test_scc_results.py`, `tests/test_scc_bulk_read.py` |
| Immutable result generations, checksum validation, no input cleanup | `tests/test_scc_results.py` |
| Old SQLite journal entries readable without granting new upload | `tests/test_scc_state.py`, `tests/test_scc_results.py` |
| Safe-GET retry configuration; no POST replay | `tests/test_scc_client.py`, `tests/test_scc_configuration_split.py` |

**Important:** these tests verify known invariants and simulated scenarios,
not bitwise scientific equivalence on the user's real archive. In particular,
passing CI alone is not evidence that the real SCC portal accepts a given
NetCDF/LR/sounding combination.

## 2. Non-destructive operator verification on real data

Before changing remote data, explicitly choose one scientific session and
Measurement_ID, and preserve the canonical scientific files (L0/L1/L2),
its associated original Licel measurements and existing journal.

1. Record SHA-256 values for the canonical scientific L0 and any relevant
   L1/L2 outputs, plus current SCC input manifest and existing remote status.
2. Run `milgrau-scc upload --dry-run` (or `--id MEASUREMENT_ID --dry-run`)
   locally; it does not connect to SCC. Record all validation failures.
3. Run `milgrau-scc status --id MEASUREMENT_ID --remote`, with normal
   authenticated SCC access. This issues remote reads, and can update
   **local journal classification** for observed states; it does not send
   an upload or delete POST.
4. For an already existing record, optionally request a verified read-only
   remote archive using `milgrau-scc status --id MEASUREMENT_ID --remote
   --backup`. Ensure the archive and hashes are preserved.
5. Check L0/L1/L2 SHA-256 values against the original science; staging and
   status must not modify them.

If a case is not clean, do not remove local inputs or reset the journal to
make a remote upload appear eligible.

## 3. Live remote SCC gates — not executed by GitHub CI

Only after a separate, explicit selection of a *safe, authorized test ID*:

- Confirm a single `upload --id MEASUREMENT_ID --confirm` using
  known-good current input files, followed by an independent API/portal
  confirmation of the SCC configuration, generation and processing state.
- If an observed sounding is attached, verify the actual Quick Upload form
  accepts its native filename and observed metadata; never substitute an
  unobserved modelled profile.
- When ELPP and ELDA reach their completed status, use
  `download --id MEASUREMENT_ID` and validate the saved manifest/archive
  hashes and generation identity.
- Test destructive replacement only under **separate explicit permission**
  for a disposable/test record and with remote preflight. Back up remotely
  first if the original must be retained. An uncertain POST/DELETE must
  be investigated before any subsequent operation.

Never test this by deleting a valuable production measurement. Real SCC
availability, credentials, and the original local scientific archive are
outside GitHub Actions.

## 4. Remaining scientific equivalence gate

A representative before/after comparison requires the same real
scientific inputs and configuration, including session selection, calibration,
dark-current links, temporal QA, meteorological inputs and instrumental
settings. Record the session IDs and hashes for L0/L1/L2 under both revisions,
then compare relevant science variables with their defined tolerances and
provenance. Differences must be explained; do not infer equivalence from an
unchanged set of unit tests. No such real-data comparison is claimed here.

## 5. CI provenance

Use the Actions run linked from `.github/tracker.md` for the exact code
commit, Python environments and regression count. Documentation-only
follow-up commits may create separate Actions runs; the evidence applies
to the code SHA tested by that run.
