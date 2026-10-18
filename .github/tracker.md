# Phase 12 — Optional SCC integration and publication workflow

**Branch:** `scc-integration-refactor` (base: `milgraumaxxing`)  
**Status:** Phases 12.1–12.6 implemented and CI validated; Phase 12.7 sequential batch-upload code/mock CI 6/6 passed; live single-ID upload/replacement and full batch pending operator validation  
**Scope:** SCC configuration, optional export, staging, native upload/status/download/cleanup client.
This branch's tracker intentionally contains **only Phase 12**. The earlier scientific tracker remains on the base branch.

## Accepted invariants

1. `station.yaml` owns observational station data, instrument hardware, calibration and climatology.
2. `config.yaml` owns scientific processing controls, optional SCC activation and operational policy.
3. `scc.yaml` owns SCC-specific configuration IDs, channel IDs, product IDs and external-service policy.
4. SCC integration is disabled by default. Disabled scientific processing must not load, validate or require `scc.yaml`.
5. SCIENTIFIC `Session_ID` and continuous canonical L0/L1/L2 products are independent of SCC `Measurement_ID` and publication boundaries.
6. Staged SCC NetCDFs are disposable. The durable submission history and downloaded outputs are not.
7. Remote upload is never implicit without explicit operator configuration; no silent delete or replacement of SCC measurements.
8. Unknown SCC IDs are errors, never inferred from other identifier types.
9. Raman retrieval products are not replaced with elastic climatological assumptions.
10. Retain lossless station profile history through configuration migration; avoid unrequested compatibility layers.

## 12.1 — Configuration ownership and migration

- [x] Create the dedicated refactor branch and Phase-12-only tracker.
- [x] Define/validate the strict `config.yaml` / `station.yaml` / `scc.yaml` ownership contract.
- [x] Retain physical instrument metadata, calibration, observational climatology, overlap and scientific Raman characterization in `station.yaml`.
- [x] Migrate SCC policy, configuration/channel mappings and SCC-only Raman mapping metadata into `scc.yaml`.
- [x] Move processing and feature-switch policies to `config.yaml`, without copying observational values.
- [x] Require `scc.yaml` only when `scc.enabled: true`; disabled mode must neither load nor validate SCC content.
- [x] Preserve all historical station profile/date and solar day/night mappings.
- [x] Provide strict station/SCC consistency checks, with no fabricated identifiers or silent aliases.
- [x] Remove SCC export from the disabled LIBIDS path and decouple disabled scientific incremental validation from SCC derivatives.
- [x] Update unit/contract tests and confirm no SCC extras are required for scientific processing.
- [x] Document the new configuration ownership and migration.
- [x] Run all applicable CI tests and review failures before marking 12.1 complete (all six checks green in run 39046944304).

**Phase 12.1 implementation evidence:** `milgrau/config/scc.py`, `milgrau/config/loader.py`, `scc.yaml`, `milgrau/config/station.py`, `milgrau/level0/libids.py`, `milgrau/level0/processing.py`, `milgrau/level1/ingestion.py`, dedicated `tests/test_scc_configuration_split.py`, and updated SCC/Raman regression tests. Branch CI includes `scc-integration-refactor`.

**Gate 12.1:** Fresh install without SCC extras can process L0/L1/L2 with `scc.enabled: false` and without `scc.yaml`; enabled mode preserves historical mapping.

## 12.2 — Product registry and climatological LR

- [x] Migrate all SCC configuration IDs 248, 294, 484, 565, 1046, 1047 and their channel IDs.
- [x] Enter verified SCC product IDs and physical product types/use cases/wavelengths from screenshots.
- [x] Support legitimate reuse of product IDs by different systems.
- [x] Resolve multiple 1064 nm product variants where their use case matters.
- [x] Read monthly station climatology and stated spread without copying values into SCC configuration.
- [x] Generate compliant LR ancillary NetCDF for eligible elastic products only.
- [x] Preserve Raman retrieval independence and ensure ancillary ID/schema/altitude contracts.
- [x] Test representative day/night and historical configurations.

**Implementation evidence:** `scc.yaml` now records all 30 distinct optical
products and the six source-system memberships from user-provided SCC screenshots,
including product type, wavelength, usecase, channel count and parent relations.
`milgrau/config/scc.py` resolves typed elastic targets without duplicated ID
lists, allows cross-system reuse and preserves both 746 and 3683 where present.
`milgrau/level0/scc_lidar_ratio.py` produces station-month LR ancillary profiles
with validated coverage of all detector ranges; only eligible elastic products
receive LR_Input=0. New `tests/test_scc_product_registry.py` and extended
`tests/test_scc_lidar_ratio_ancillary.py` cover current and historical modes,
shared IDs, bad IDs, Raman guards, monthly values and NetCDF schema.

**Validation:** All six checks passed in GitHub Actions run
[39047834262](https://github.com/spulidar/milgrau/actions/runs/39047834262)
on implementation commit `6e21d7cad504f5d0ec267dd582b982a13e3d6670`
(Ubuntu + Windows, Python 3.12/3.14, Ruff and Explorer). Subsequent edits
are documentation/tracker-only. This phase does **not** claim a successful
live SCC upload; that is a separate gate in Phase 12.7.

**Gate 12.2:** Every SCC product and LR ancillary is traceable to real SCC product IDs.

## 12.3 — LIBIDS staging separation

- [x] Separate canonical L0 incrementality from SCC export validity.
- [x] Stage SCC-only L0, optional LR and manifest under `.cache/scc/`.
- [x] Preserve source session and profile/segment metadata.
- [x] Handle month boundaries, regime splits, multiple segments and ID collisions without splitting scientific sessions.
- [x] Use atomic writes and SCC-only reexport for historical L0.
- [x] Define safe treatment of old `*_L0_scc.nc` under scientific folders.

**Implementation evidence:** `milgrau/level0/scc_staging.py` implements
`.cache/scc/pending/<Measurement_ID>/`, atomic per-package staging,
SHA256 manifest verification, explicit identity-collision failures and
local file source/profile lineage. `milgrau/level0/processing.py` partitions
SCC-only packets by day/night and UTC month while leaving continuous
scientific L0 unchanged; individual measurements crossing a month boundary
are explicitly refused (not silently assigned a climatology).
`milgrau/level0/libids.py` verifies scientific incrementality independently
of SCC cache, and `milgrau-libids --scc-reexport -i SESSION_ID` regenerates
upload packages from archived raw data only for an existing valid canonical L0.
Legacy SCC files in processed folders are retained, never migrated/deleted.
Dedicated `tests/test_scc_staging.py` guards atomicity, tampering, collisions,
regime/month partitioning, source profiles and scientific independence.
Phase 12.3 does not submit or delete remote measurements: those capabilities
remain scheduled for Phases 12.4–12.6.

**Validation:** All six checks passed in GitHub Actions run
[39049077255](https://github.com/spulidar/milgrau/actions/runs/39049077255),
at implementation commit `130f5a3c1e9b5c35b1b195f772fb876145d78269`
(Ubuntu and Windows, Python 3.12/3.14, Ruff and Explorer). Subsequent
changes are documentation/tracker-only. Real SCC submission is not yet tested.

**Real-data local validation (2026-10-17):** `milgrau-libids --scc-reexport`
finished with **19 processed / 0 skipped / 0 errors** on archived acquisitions
(9 day using SCC configuration 1047, 10 night using configuration 1046).
All 19 `.cache/scc/pending/<Measurement_ID>/` manifests passed local
checksum/size validation and each included raw NetCDF plus LR auxiliary.
Two unrelated dark-current Licel headers were skipped for invalid `NShots`;
some profile rows also failed acquisition QA as logged. The run validates
local staging, **not** SCC remote acceptance or all scientific NetCDF content;
validate dark-current availability for affected sessions before live upload.

**Gate 12.3:** Removing staging cache never triggers canonical L0 reprocessing.

## 12.4 — Optional native SCC client

- [x] Add `.[scc]` extra and `milgrau-scc` CLI with only `upload`, `status`, `download`, `cleanup`.
- [x] Implement protocol based on SCC's public documented authenticated Quick upload form, optical ancillary field and measurement API, including HTTPS/TLS, CSRF protection, timeouts, safe bounded GET retries, duplicate rejection and no automatic POST retries.
- [x] **Live read-only SCC authentication + API validation:** operator successfully authenticated and queried an existing SCC measurement (20240913spu1549, configuration 1047).
- [ ] **Live multipart upload validation:** not performed; do not attempt the above Measurement_ID because it already exists on SCC.
- [x] Keep credentials out of repository and configuration files (website and optional HTTP Basic credentials only from environment).
- [x] Validate pending submission and support `upload --dry-run` entirely offline, without credentials.
- [x] Submit raw and optional climatological LR together; verify remote Measurement_ID before reporting success; test all network interactions with mocks.
- [x] Require an explicit `--id ... --confirm` to submit only ONE remote measurement. Forbid delete, replacement, batch submission and unsafe redirects.
- [x] Leave `download` and `cleanup` as explicit non-mutating gated commands until durable lifecycle exists in Phases 12.5–12.6.
- [x] Document operator installation, environment-only credentials, CLI behavior and incident recovery in `docs/scc_native_client.md`.

**Implementation evidence:** `milgrau/scc/packages.py`, `milgrau/scc/client.py`,
`milgrau/cli/scc.py`, `tests/test_scc_client.py`, `pyproject.toml`,
`config.yaml`, `scc.yaml`, `docs/scc_native_client.md`. Upload uses
`data` plus `lidar_ratio_file` under SCC's HTTPS quick-upload form and
checks `api/v1/measurements/<Measurement_ID>/` to reject existing remote IDs
and confirm response. Read-only `status --remote --id` is available; remote
durable state/download/cleanup are **intentionally pending**.

**Validation:** mocked HTTP protocol and local packet validators tested on CI;
a real SPU SCC login/upload is still required before declaring live compatibility.
**No real remote upload was performed by this development session.**

**Real local CLI validation (2026-10-17):** Operator installed `.[scc]` after updating the branch and ran `milgrau-scc upload --dry-run` and `milgrau-scc status` successfully on **19/19** pending uploads (**9** SCC day configuration 1047, **10** SCC night configuration 1046); each package includes `raw.nc` plus a climatological `lr_<Measurement_ID>.nc`. No network request was made or required. **Next:** read-only SCC login/measurement-status check with environment-only credentials; actual remote Quick upload is still unverified.

**Live read-only authentication test (2026-10-17):** An initial
`status --id 20240913spu1549 --remote` received HTTP 401 at the
pre-website login layer. After configuring access, the same read-only command
**succeeded** and returned the full SCC JSON for existing measurement
`20240913spu1549` (SPU, configuration 1047, start
2024-09-13T15:49:32, stop 2024-09-13T17:50:42). The record reports
`is_being_processed: false`, `is_queued: false` and successful
`exit_code: 0` for ELPP, ELDA, Cloudmask, ELIC, ELQUICK and HiRELPP;
ELDEC has no results. The remote measurement record was created
2026-10-17T14:29:53. The remote system's `products` also includes
product 3687 of type 'High Resolution pre-processed data', which is
**not** an additional elastic optical backscatter product and therefore
is outside the 30-product optical registry as currently scoped.

**Evidence limits:** successful read-only authentication and remote processing
of an **existing** measurement do not establish that the MILGRAU native
multipart uploader was used or that its LR ancillary was accepted; do not
re-upload this ID. The next live-test candidate must first be verified absent
on the remote SCC, with operator approval. The 19 locally staged files remain
untouched. No remote upload was performed by this development session.

**Gate 12.4:** Offline client and authentication mocks pass; real upload remains
explicitly opt-in. Live integration validation is tracked separately in 12.7.

## 12.5 — Durable submission state

**Approved identity and replacement policy:** SCC Measurement_ID is stable.
Never fabricate a new measurement identifier merely to accommodate a different
period, set of source profiles, or climatological LR. Follow the established
SCC force-upload semantics (delete the existing remote record, then upload
the new raw + ancillary under the **same ID**) rather than introducing
version-suffix IDs. No remote deletion is permitted before implementing
safeguards and explicit operator confirmation.

- [x] Track original session, SCC Measurement ID, configuration/products, source periods, raw/LR checksums, timestamps, and remote IDs outside disposable cache.
- [x] Compare existing SCC metadata and, when necessary, downloaded remote raw NetCDF/LR references; do not infer LR equivalence solely from API status or the remote timestamp.
- [x] Preserve stable Measurement_ID when local period/profiles/LR differ from the remote record.
- [x] Before replacement, validate new raw and LR package completely, archive original remote raw/configuration and downloadable products, and verify the archive locally.
- [x] Implement exact-ID opt-in `upload --id ID --replace-existing --confirm`, keeping destructive replacement disabled by default.
- [x] Use SCC's supported authenticated web deletion flow (not an invented overwrite API), verify disappearance, then submit original ID with new raw+LR.
- [x] Protect against intervening remote changes and concurrent clients; record all transitions in durable state outside cache, with process locking.
- [x] Record replacement as non-atomic: if deletion succeeds but POST or confirmation fails, keep explicit recoverable/uncertain state and do not falsely report success or automatically loop/overwrite.
- [x] Represent pending/uploading/accepted/processing/completed/failed/uncertain and replacement-specific states explicitly.
- [x] Guarantee retry reconciliation, prevent accidental duplicates, and make `status` useful offline and online.
- [ ] Test deletion/backup failure, incomplete archive, same-data no-op, LR-only change, period-only change, stale-remote conflict, ambiguous POST, rollback best effort and absence of silent data loss with mocked server.

**Implementation evidence:** `milgrau/scc/state.py` provides the
durable SQLite ledger, append-only event journal and exclusive per-ID lock
in `.state/scc` (ignored by Git; independent of `.cache/scc`).
`milgrau/scc/backup.py` archives original remote raw NetCDF/ZIP and all
advertised completed results with streaming limits, container verification
and SHA-256 checks. If an existing external LR cannot also be retrieved,
replacement is blocked before the remote delete.
`milgrau/scc/workflow.py` integrates full local preflight, status/fingerprint
checks, immutable archive verification, proven-identical no-op, and
delete-then-upload under the **same** Measurement_ID. The native
`milgrau/scc/client.py` implements the SCC administrative CSRF deletion form
with `not_delete_related`, followed by authoritative API recheck. Uncertain
POST outcomes always block further automatic attempts. `milgrau-scc status`
now displays durable state even after deleting the disposable cache.
New `tests/test_scc_state.py`, `tests/test_scc_backup.py`,
`tests/test_scc_workflow.py` and expanded `tests/test_scc_client.py`
exercise the non-destructive and simulated replacement cases. A separate
read-only `milgrau-scc status --id ID --remote --backup` command can verify
actual downloadable SCC archives **without deleting or uploading any
measurement**, to expose remote format/ancillary limitations before a first
live replacement.

**Operator live read-only backup validation (2026-10-17):** The operator ran
`milgrau-scc status --id 20240913spu1549 --remote --backup` successfully.
Local pending integrity passed; the client downloaded and verified a remote
archive under `.state/scc/archives/20240913spu1549/20261017T204440914284Z`.
The verified archive-manifest SHA-256 was
`5faba8977f1c3687fea369616cbd6c939d73d9ff4b7347d9f842a2b851c40c52`.
SCC confirmed system 1047, start `2024-09-13T15:49:32`, stop
`2024-09-13T17:50:42`, `is_queued=false`,
`is_being_processed=false`, ELPP=127 and ELDA=127. This establishes
**real-server read-only archival compatibility for this measurement**, not
safe live delete/reupload or broad archival compatibility for all systems.
No remote POST, replacement, cleanup or deletion occurred. Before any live
replacement, inspect the retained archive and compare archived raw/LR
with staged inputs, ensure a recoverable restoration path, and explicitly
authorize the risk of losing the previously processed SCC results.

**Remaining gates:** Simulated rollback/recovery after every destructive failure,
especially when the SCC's original LR or raw archive schema differs, requires
additional integration evidence. The SCC has no atomic compare-and-swap
on deletion: independent portal edits between the last GET and deletion
cannot be eliminated client-side; the operator must coordinate a maintenance
window. **No live remote DELETE or replacement POST occurred.** Phase 12.6
is still responsible for final download and cleanup.

**Validation:** All six checks passed in GitHub Actions run
[39057561308](https://github.com/spulidar/milgrau/actions/runs/39057561308)
on implementation commit `78b388b180b44150e031ba86bcbb1b5e6d0fdd90`
(Ubuntu and Windows, Python 3.12/3.14, Ruff, Explorer). This is **offline
and mocked-server** validation only. A read-only `status --remote --backup`
against the live SCC is the next prerequisite before considering any real
delete-and-reupload. Automatic rollback after unknown destructive outcomes
is intentionally unavailable; the verified archive, durable journal and
explicit manual reconciliation are the safe recovery foundation.

**Gate 12.5:** No prior SCC measurement can be deleted without a locally verified archive of available remote inputs/results, explicit exact-ID operator authorization, and durable recoverability tracking. Interrupted replacement must never be reported as successful without API confirmation.

## 12.6 — Download and cleanup

- [x] Download/verify SCC ELPP/ELDA and any other completed downloadable SCC products to separate persistent `scc-results/` snapshots.
- [x] Persist remote provenance, processing configuration, source Session_ID, result links, timestamps, immutable generation identity and SHA-256 manifest.
- [x] Clean only safe staged files after confirmed MILGRAU upload **and** verified completed remote processing **and** download, retaining failed/uncertain and external portal submissions.
- [x] Never erase raw Licel, scientific L0/L1/L2, old SCC replacement archives or durable SQLite submission history.
- [x] Provide `milgrau-scc download --id ID [--session SESSION_ID] [--keep-cache]` and `cleanup --id ID --dry-run|--confirm` under the existing four-subcommand CLI.
- [x] Migrate old SQLite 12.5 journals non-destructively to record results path/checksum and downloaded/cleaned states.
- [x] Add regression tests for interrupted/partial downloads, duplicate/stale remote generations, other people's existing SCC uploads, failed/incomplete ELPP/ELDA, changed checksums and cache idempotence.
- [x] **Live read-only SCC processed-result download** validated with SPU operator for one existing measurement; no live SCC uploads, replacements or deletions during development.

**Implementation evidence:** `milgrau/scc/results.py` atomically stages
verified SCC archives into
`<scientific-session>/scc-results/<Measurement_ID>/generations/remote-<hash>/`.
Each local generation is keyed to remote creation date without changing the
original SCC Measurement_ID; older results remain preserved after reprocessing.
Remote URLs, creation timestamp, processor statuses, SCC configuration and
hashes are retained in `results.json`. `milgrau/scc/state.py` persists
downloaded results checksums and retains earlier archive/event history.
`milgrau/scc/cleanup.py` independently checks local and remote provenance,
staged raw/LR hashes, the journal's accepted upload generation and verified
ELPP/ELDA results before removing **only**
`.cache/scc/pending/<Measurement_ID>/`. The configured `after_download`
policy automatically invokes this safe cleanup for journal-confirmed uploads;
`--keep-cache` or `manual` preserve the cache. A historical SCC record
downloaded with `--session` but not uploaded by MILGRAU is never
cleanup-eligible. Interrupted and uncertain submissions are excluded.

**Validation:** All six jobs passed in
[GitHub Actions run 39069033200](https://github.com/spulidar/milgrau/actions/runs/39069033200)
on commit `447e9fb7fe4b01e34fd8099f5df2041534dc2925`
(Ubuntu/Windows; Python 3.12/3.14; Ruff; Explorer).
The subsequent commits changed only README, SCC documentation and tracker.
The SCC processed-results workflow is covered by mocked downloads,
corruption tests, interrupted transfers, journal migration and safe cleanup;
**live remote SCC result download has subsequently been validated for one pre-existing measurement**.
No real SCC upload, deletion or replacement POST occurred.

**Real SCC download validation (operator, 2026-10-17):** The operator
ran the Phase 12.6 `milgrau-scc download` workflow against existing remote
Measurement_ID `20240913spu1549`, linked to scientific Session_ID
`spu_20240913-1549Z_20240913-1750Z`. The CLI confirmed:
`SCC downloads VERIFIED` with archived remote product groups
`cloudmask, elda, elic, elpp, elquick, hirelpp` (six groups).
Immutable remote generation
`scc-results/20240913spu1549/generations/remote-a5beca97ee50b9df37ff/`
was saved under the canonical scientific session. Crucially,
`upload_owned=False`: this historic SCC record was not submitted by
the MILGRAU native client and is **not eligible for cache cleanup**;
staged scientific raw/LR inputs must remain untouched. This validates
read-only live downloading and local verification for this specific
measurement, **not** a new live upload, same-ID replacement,
automatic cleanup after a verified MILGRAU-owned upload, or numerical
cross-comparison of SCC outputs. These remain Phase 12.7 gates.

**Gate 12.6:** SCC submissions and external scientific results remain
traceable after cache cleanup; uncertain states and original scientific
products are never deleted by SCC cleanup.

### Phase 12.7 milestone — sequential batch SCC upload (operator requested)

- [x] Enable `milgrau-scc upload --confirm` without `--id` to process every locally pending Measurement_ID in deterministic order, one SCC API session at a time.
- [x] Default to **skip remote-existing** IDs without remote mutation or inventing MILGRAU upload ownership.
- [x] Provide the intentionally destructive opt-in `milgrau-scc upload --confirm --replace-existing` for remote-existing IDs, applying the same per-ID validated archive → compare → delete → reupload under the stable Measurement_ID, while uploading missing IDs normally.
- [x] Prevalidate **all** staged packages offline before any network request or POST. Print per-ID progress plus summary counts submitted/replaced/identical/skipped_existing/failed/unprocessed.
- [x] Stop on first failure, stale/uncertain journal, mismatched remote state or ambiguous HTTP operation; do not blindly continue or repeat POSTs/DELETEs. Retain cache and earlier journal/archives.
- [x] Preserve `upload --id ID --confirm` and `--id ID --replace-existing --confirm` for a single-ID staged live trial.
- [x] Add simulated batch tests including an inventory of 19 staged ID names, default skip, explicit replacement, network interruption and no-partial-send preflight rejection.
- [x] Document the single-ID backup/destructive trial and safe launch sequence in `docs/scc_batch_operator_guide.md`.
- [ ] **Live controlled single-ID upload and replacement** with explicit operator authorization, including proof that climatological LR was received and ELPP/ELDA completed. This is NOT implied by earlier live read-only SCC downloads.
- [ ] **Live full 19-packet batch** after controlled upload/replacement validation; retain cache with `cleanup.policy: manual` and `download --keep-cache`.

**CI evidence:** All six jobs in
[GitHub Actions run 39070844616](https://github.com/spulidar/milgrau/actions/runs/39070844616)
passed on batch test commit `7e71b239539bf1d531d837b68620c8e41bdecf06`
(Ruff, Explorer, Ubuntu/Windows and Python 3.12/3.14).
Later commits updated only operator documentation, README and tracker.
Never mark the full SCC integration released merely because mocked
integration tests pass. Batch replacement deletes real SCC records and has
no server-side atomic rollback. Do not execute it until one approved
single-ID replacement is independently validated, and archives have
been retained.

## 12.7 — Validation and release

- [ ] CI across supported Python versions and operating systems, including installation without optional SCC extra.
- [ ] Test disabled integration and enabled mode without credentials.
- [ ] Live upload test: system 1047 day and system 1046 night; verify LR ancillary acceptance.
- [ ] Test real historical configurations, crossing midnight/month and repeated day/night segments.
- [ ] Compare retrieved SCC products under matched inputs with MILGRAU processing.
- [ ] Update README/configuration reference/CLI docs and retire old SCC producer paths.
- [ ] Require real upload/download evidence before declaring release ready.

**Live same-ID delete incident (operator, 2026-10-17):** On
`milgrau-scc upload --id 20240913spu1549 --replace-existing --confirm`,
the SCC admin deletion POST raised an unknown transport outcome. Follow-up
authenticated API GET returned **NOT FOUND**, and the original workflow
stopped **before any replacement-upload POST**, with durable state
`uncertain`. The locally verified original measurement archive remains
at `.state/scc/archives/20240913spu1549/20261017T212943822169Z`.
The `--dry-run` still validates all **19/19** staged packages
(9 day system 1047 and 10 night system 1046). Read-only
`status --remote --backup` correctly cannot back up an absent remote
record; it does **not** indicate the earlier local archive was lost.

**Recovery implementation:** `milgrau-scc upload --id
20240913spu1549 --resume-after-delete --confirm` is an explicitly
authorized **single-ID-only** recovery operation, not a batch or retry
of deletion. `workflow.resume_after_uncertain_delete` verifies the
journal's most recent `delete_result_uncertain` event, prior
`backup_verified` and `delete_requested` audit events, retained
original SCC raw/results archive checksums, exact staged source
hashes/period/configuration and read-only SCC API absence. It then
attempts exactly one *upload-only* POST under the same ID, storing
`accepted` if and only if the post-upload API confirms the new remote
record; any uncertain recovery is blocked again. Never erase or
manually reset the SQLite journal. The admin-deletion transport
exception class is now exposed for **future** failures; the original
cause remains unknown from the recorded message.
Mocked tests cover successful recovery, absent/present SCC record,
tampered/missing backup, wrong uncertainty event and uncertain
recovery-upload failure.

**Recovery regression validation:** All six jobs passed in
[GitHub Actions run 39071990618](https://github.com/spulidar/milgrau/actions/runs/39071990618)
for recovery code commit `d43d2da858a95fa7caacbf5f7a1df5b490b6723f`
(Ubuntu/Windows Python 3.12/3.14, Ruff, Explorer).
This includes simulated success and failure of the explicit
`--resume-after-delete` command. The real SCC recovery upload
is **NOT** validated by this CI run; it remains an operator-controlled
one-time live test, not a batch action.

**Open live gate:** Operator must verify retained local backup independently,
confirm SCC ID remains absent, decide whether to restore original or
finish replacement, and only then authorize the single-ID
`--resume-after-delete` command. Neither full batch mode may be used
before the resulting new SCC upload and LR acceptance are validated.
An SCC administrator should inspect server access/error logs for the
unexpected POST transport failure. The new recovery path has no
confirmed live POST yet.

**Second live upload recovery incident (operator, 2026-10-17):** An explicitly
authorized `milgrau-scc upload --id 20240913spu1549
--resume-after-delete --confirm` reached the upload stage but raised
`Upload response ... was not confirmed by the SCC API; remote state is
uncertain. Do not repeat the POST.` This does **not** establish whether
the server accepted, delayed indexing, or rejected the upload. The
durable journal should now have `state=uncertain` with last action
`recovery_upload_uncertain`; no batch or repeat upload is authorized.
The operator has been asked for a **new read-only** `status --remote`
and confirmation in the SCC portal.

The official `scc_access` implementation recognizes that API visibility
may be delayed after POST (multiple safe GETs) and that HTTP 200 ending
on the Quick upload form can signal rejected form validation. The
MILGRAU client now tries **three read-only GET confirmations** without
a second POST and gives distinct diagnostics for Quick-form rejection
versus unconfirmed delayed API visibility. New unit tests simulate
both scenarios. Another potential explanation is an already registered
auxiliary LR file; the official `scc_access` checks LR existence before
multipart attachment, whereas MILGRAU currently attaches the staged LR.
Do not silently reuse an unverified remote LR: it must match the intended
scientific auxiliary by content, not just filename. The previous HTTP
response body was not recorded, so its form-validation reason cannot be
retroactively determined. A new live POST is **not** warranted without
remote diagnosis and explicit manual reconciliation.

**Diagnostic CI:** All six jobs passed in
[GitHub Actions 39072914060](https://github.com/spulidar/milgrau/actions/runs/39072914060)
on `1e2d8c674fc833e607737f0c81eb41639a9c561e`, covering safe
GET-only confirmation after single multipart POST and detection of
possible Quick upload form rejection; subsequent commits updated
documentation and tracker only. This does **not** establish that the
real SCC upload was accepted. The operator should provide only
read-only API status, journal events and ancillary LR registry
information before a further change is attempted.

**Likely SCC upload validation incompatibility (2026-10-17):**
The authoritative [SCC 5.2 raw NetCDF input specification](https://docs.scc.imaa.cnr.it/en/latest/file_formats/netcdf_file.html)
requires the uploaded raw file name to be `<Measurement_ID>.nc`;
the original native MILGRAU multipart used `raw.nc`, the filename of
the *disposable local staging file*. SCC returned HTTP 200 without an
authoritative API measurement after the previously attempted POST.
A Quick upload form validation rejection is a plausible explanation,
but its original response HTML was not captured, so causation is **not
proven**. The multipart filename was corrected to
`<Measurement_ID>.nc` while the local cache and checksum manifest
continue to use `raw.nc`; the regression test asserts the exact
transmitted filename. The official SCC specification also requires
`LR_Input=0` when using an external LR; the existing
`scc_policy.lr_input.source=climatology_ancillary` generator resolves
these external-profile flags to zero for eligible elastic channels.
The operator's read-only LR endpoint returned HTTP 200 with no objects
printed for the requested `lr_20240913spu1549.nc`; this suggests no
file was found by that lookup, *not* LR upload success.
**Do not re-upload or re-run the batch merely because this client-side
filename defect was fixed.** The current persistent journal remains
`recovery_upload_uncertain`. First reconcile the SCC portal/API and
determine whether the failed upload was rejected; only then design a
single exact-ID, explicit reconciliation/retry procedure with no remote
deletion or automatic batch resumption.

**Filename compatibility fix validation:** All six cross-platform jobs
passed in [GitHub Actions run 39073865678](https://github.com/spulidar/milgrau/actions/runs/39073865678)
on commit `0401f560b7d9f70cbb3ce7df4f2f7e005f427581`.
The native client now transmits the staged raw bytes as
`<Measurement_ID>.nc`, conforming to the SCC input-file filename
specification, while retaining `raw.nc` and all checksum/provenance
data unchanged locally. The previous real upload is still unresolved,
the journal remains `recovery_upload_uncertain`, and no further
POST has been made. Never interpret passing CI as live acceptance
or automatic permission to resend.

**SCC upload identity-name migration and second recovery attempt (2026-10-17):**
The operator reconfirmed the remote `20240913spu1549` is absent
in **both** authenticated API and the SCC website. The full local
19-packet inventory and scientific journal remain preserved; this
conversation has not issued any live SCC API POST.

The on-disk stage format now uses `<Measurement_ID>.nc` rather
than `raw.nc` for new SCC raw exports, matching the required
multipart upload filename. The companion
`lr_<Measurement_ID>.nc` and manifest are unchanged except that
`manifest.files` names the ID-based raw file. An offline
`milgrau-scc upload --migrate-cache` command migrates **existing**
pending packages without regenerating scientific or NetCDF payloads:
it checks SHA-256 hashes, copies the old file to the new name,
atomically publishes the revised manifest, verifies the new package,
then removes only the now-redundant `raw.nc`. It preserves the
existing journal, uncertainty events, original SCC backups, LR
and results, and can resume safely if interrupted.

A separate `milgrau-scc upload --id 20240913spu1549
--retry-after-upload --confirm` is now available **only** for the
exact prior `recovery_upload_uncertain` last journal event. It
requires unchanged local raw/LR hashes, matching session/period,
product IDs, segments, SCC catalog hash, a verified original SCC
backup and a fresh 404 before attempting **one upload-only** POST
of `20240913spu1549.nc`. It never deletes the remote record or
resumes the 19-ID batch. A subsequent ambiguous result records
`manual_corrected_upload_uncertain`, which blocks further
automatic attempts.

**Real SCC acceptance remains unverified.** The final cross-platform
CI must pass and the operator must run the offline migration/19-of-19
preflight before choosing to authorize the single-ID corrected
upload on their local machine. Explicit cache retention:
`scc.cleanup.policy: manual` and `download --keep-cache`.

**Phase 12.7 local-ID stage migration and guarded retry — CI green:** All six
jobs in [GitHub Actions run 39075402576](https://github.com/spulidar/milgrau/actions/runs/39075402576)
passed on `8710ac28797b4a523b575c429223f235947d0b6d`
(Ubuntu/Windows Python 3.12/3.14, Ruff, Explorer). The
subsequent commits updated only documentation and tracker.
Tests cover offline migration of **19** legacy packages, byte-for-byte
raw/LR preservation, journal/event unchanged, collision/tampering/crash
recovery, current NetCDF file naming, upload-only recovery success,
missing/archive/remote-present refusal, and no second POST after
subsequent uncertainty. This is code/fixture validation **only**;
the operator has not yet run local cache migration or the newly
corrected live SCC upload on their Arch Linux machine.

**SCC deployment LR Quick Upload form rejection (2026-10-17):**
The operator manually selected system 1047, uploaded
`20240913spu1549.nc` and `lr_20240913spu1549.nc`, and the
**actual SCC form rendered a red validation error requiring**
`lr_YYYYMMDDcccNN.nc` (exactly **two** digits in NN). This
explains the form-level rejection observed in the previous
`manual_corrected_upload_uncertain` attempt, independently
of raw NetCDF filename. The 20240913 raw file's NetCDF
`LR_File_Name` was also confirmed to reference the incompatible
long auxiliary. NetCDF4 and LR NETCDF4_CLASSIC files, their sizes
and variables were inspected locally; no full successful SCC raw/LR
validation has occurred yet. Generic SCC documentation's
`lr_<Measurement_ID>.nc` examples differ from the enforced
two-digit suffix in this deployment, so the **live form wins**.

**Offline correction implemented:** New raw exports reference
`lr_YYYYMMDDcccHH.nc` where HH is the first measurement's
UTC starting hour, a stable two-digit NN. For this case it is
`lr_20240913spu15.nc`. Same-day/same-hour local collisions
are blocked rather than silently reusing LR.
`milgrau-scc upload --repair-lr-cache` copies each original
pending packet into retained `.cache/scc/legacy-lr/<ID>/`,
updates the original NetCDF's `LR_File_Name` in a separate
copied raw NetCDF, renames only the LR file without modifying
its contents, regenerates SHA-256 manifest metadata and verifies
the corrected package before publishing. It is offline; the
canonical science NetCDF, original remote SCC archive,
old results generation, and local journal are preserved.
For the existing `uncertain` journal, an auditable
`local_lr_filename_repair` event stores old/new raw hashes,
unchanged LR hash and preceding failed action without
resetting the unknown remote upload status.

A guarded exact-ID `--retry-after-lr-fix --confirm` operation
accepts only this specific journal repair event, original
backup, unchanged scientific identity/configuration, newly
corrected checksums, and remote SCC 404. No remote deletion,
no automatic retry, no batch; failure moves journal to
`manual_lr_fixed_upload_uncertain` which blocks further POSTs.
Before a live attempt, verify the corrected ancillary name is
not already registered with status OK in the SCC portal.
**Actual upload after repair remains untested**; do not run
the full 19-packet batch until a single-ID corrected upload
and processed LR use have been demonstrated.

**Local LR fix regression CI:** All six jobs passed on code/test
commit `e78d0956f99b94157f947feb54a6831d40725c6e`:
[GitHub Actions 39077750685](https://github.com/spulidar/milgrau/actions/runs/39077750685).
Ubuntu/Windows 3.12/3.14, Ruff and Explorer all green. This
validates mock/offline cache repair, digest/journal safeguards and
single-ID upload-only gating, **not real SCC acceptance**. The
operator must first run the offline `--repair-lr-cache`, check
19/19 `--dry-run`, confirm SCC ID and corrected ancillary name
are absent, and perform an expressly authorized one-ID trial
before proceeding to a bulk upload.

**Optional SCC radiosounding support (2026-10-17):**
The SCC input specification v3.6, table 2 permits an observed
`rs_YYYYMMDDcccNN.nc` sounding with altitude above the sounding
station, temperature (°C), pressure (hPa), latitude/longitude/altitude,
and synoptic launch date/time. When supplied, the SCC raw file must set
`Sounding_File_Name` and `Molecular_Calc=1`. The official
`scc_access` Quick Upload field is `sounding_file`.

MILGRAU now includes `milgrau/level0/scc_sounding.py` for optional,
**policy-configured (now enabled in the SCC example)**, observed Wyoming cache-only sounding staging in
**new** SCC Level 0 exports. It uses the pre-existing scientific
radiosonde nearest-synoptic policy and its historical cached
CSV/JSON metadata; it does not issue weather-provider requests,
substitute ERA5 or USSA76 values or change Level 1's productive
atmosphere recipe. The observed SBMT launch-site metadata are stored
separately from the SPU lidar station coordinates.
When an eligible sounding is present, the SCC-only raw export gains
`Molecular_Calc=1` and the matching `Sounding_File_Name`, the
manifest records the sounding SHA-256, package validation verifies
the sounding NetCDF physical/schema fields, and the client attaches
the verified file under `sounding_file`. The persistent submission
journal records a separate nullable `sounding_sha256`, including
migration of previous SQLite journals, and guarded cleanup checks it.
If disabled or optional data absent, the legacy `Molecular_Calc=0`
and no sounding file remain unchanged. With `missing_policy=fail`,
unavailable observational data stop the SCC export.

**Incident isolation:** Do not re-export or attach sounding to
`20240913spu1549` while its previous attempted uploads remain
`uncertain` and historical raw/LR SHA-256 must be reconciled.
First run the *offline* `upload --repair-lr-cache` and dry-run for all
19 stages, then verify the upload form accepts the LR of the single
authorized incident ID. SCC sounding support is not a reason
to repeat or add another live upload to the unresolved incident.

**Validation:** New regression tests cover exact sounding NetCDF
types/units/launch time, source cache provenance, refusal to synthesize
missing data, unphysical profiles, multipart field routing and prior
journal migration. All **6/6** cross-platform checks passed in
[GitHub Actions run 39079349685](https://github.com/spulidar/milgrau/actions/runs/39079349685)
(Ubuntu/Windows Python 3.12/3.14, Ruff and Explorer); later commits changed
documentation/tracker only. **No live SCC sounding upload has been tested**.
The original production incident has now been accepted with LR and downloaded; do not retroactively rewrite it with sounding.

**New 12.7.1 — LIBIDS → LIPANCORA SCC radiosounding handoff (2026-10-17):**
Confirmed the user's LR-corrected `20240913spu1549` upload was
accepted by the remote SCC API (`configuration=1047`) and its
products were downloaded/verified. The local
`after_download` cleanup automatically removed that pending packet;
persistent downloaded products and journal remain. No live
re-submission or remote deletion was performed by the changes below.

- [x] LIBIDS stages raw SCC `<ID>.nc` and valid two-digit LR filename,
  with `sounding_stage=awaiting_lipancora` **only when**
  `scc_policy.sounding.enabled=true`; default example now enabled
  for new SCC-enabled packets.
- [x] SCC `upload --dry-run` refuses `awaiting_lipancora` stages
  before an L1 run. Unchanged SCC-disabled flows and previous pending
  stages retain their existing behavior.
- [x] LIPANCORA completes the pending SCC sounding step after successfully
  writing a new scientific L1 **or verifying an incremental L1**,
  reusing only the same selected, provenance-traceable observed Wyoming
  sounding from cache. An absent or mismatched sounding is handled
  according to explicit `skip`/`fail` policy; no ERA5/USSA76
  data are labelled as radiosonde observations.
- [x] Atomic staged directory revision: preserve original raw/LR
  provenance, attach validated `rs_YYYYMMDDcccNN.nc`, update raw
  `Molecular_Calc` and `Sounding_File_Name`, recompute manifest
  SHA-256. Refuse changes whenever an SCC submission journal row
  already exists, or if `scc.yaml` changed after LIBIDS.
- [x] Guarded validation, durable optional sounding hash, and
  cache-cleanup protection from previous steps remain unchanged.
- [x] New tests exercise awaiting state, valid observed sounding,
  no-sounding permissive/strict policy, idempotence and journal refusal.
- [x] Example `config.yaml scc.cleanup.policy` changed to `manual`
  to retain pending inputs after a successful download, matching
  the operator's requested cache retention.
- [ ] Validate one **new** SCC sounding upload against the actual
  server. Keep the already accepted `20240913spu1549` untouched.
- [ ] Cross-platform CI on final documentation/implementation commit.

**Historical SCC sounding backfill (operator request, 2026-10-17):**
The previous LIPANCORA run did not add `rs_...` to **18/18**
previously staged historical packets because their manifests were
created before the `awaiting_lipancora` handoff. The operator
explicitly requested a safe retrospective conversion from existing
L1 and observed radiosonde caches, with no reprocessing and no SCC
API writes.

- [x] `milgrau-scc sounding --dry-run [--id ID]` performs read-only
  per-ID inventory of old `sounding_stage=not_requested` packets,
  checks the canonical L0/L1 relationship, L1 Session_ID and
  radiosonde observation flag/target, station code and the actual
  cached Wyoming sounding CSV/JSON with no remote calls.
- [x] `milgrau-scc sounding --confirm [--id ID]` enriches only
  eligible, *unsubmitted* historical packets. All existing
  `SubmissionStore` rows block file changes. Older scientific
  L0/L1, raw source files and LR hashes remain unchanged; SCC-specific
  raw + manifest update atomically with original catalog provenance
  retained and newly used sounding policy catalog hash recorded
  separately.
- [x] Missing observed soundings leave original historical packets
  unchanged (SKIP), and invalid packages stop modification (ERROR).
  A confirmed backfill produces `sounding_stage=ready`,
  `rs_YYYYMMDDcccNN.nc`, `Sounding_File_Name`, `Molecular_Calc=1` and
  checked SHA-256 values.
- [x] Tests added for preview no mutation, successful one-ID
  conversion, missing observation no-op, journal safeguard,
  CLI dry-run/confirm and repeated invocation.
- [x] LIPANCORA session-midpoint radiosonde selection validated
  (rather than comparing the sounding with the first laser profile).
- [x] All six jobs passed in
  [GitHub Actions run 39085228536](https://github.com/spulidar/milgrau/actions/runs/39085228536):
  Ruff, Explorer, Ubuntu/Windows Python 3.12 and 3.14.
- [ ] Operator runs the real 18-packet offline preview, applies the
  eligible subset and validates **one** new remote SCC sounding
  upload before beginning any bulk replacement.

This historical backfill is **not automatic on LIPANCORA**: the
explicit `sounding --confirm` action is required to change pre-existing
SCC packets. Newly created `awaiting_lipancora` packets still follow
the normal two-stage automatic finalization. No SCC POST/DELETE was
executed in development.

**SCC sounding filename fix (2026-10-18):**
The SCC Quick Upload HTML form explicitly rejects `rs_<Measurement_ID>.nc`
and requires `rs_YYYYMMDDcccNN.nc`, with a **two-digit** suffix
(e.g., `rs_20241008spu16.nc`). Generation, manifest name validation,
`Sounding_File_Name`, and outgoing `sounding_file` multipart now
use the same canonical `scc_sounding_name()` function.
New tests assert the exact expected name and validate the end-to-end
multipart upload filename. This change does not retroactively alter
operator-side cache or remotely submit files. Manual rename of an
already-created packet must also update raw references and file hashes.

**SCC batch incident + offline sounding filename repair (2026-10-18):**
The operator ran `milgrau-scc upload --confirm` over 18 pending
packets. IDs `20241008spu1650` and `20241127spu1600` were read-only
skipped as already existing. `20241220spu0414` then reached a real
multipart Quick Upload POST and the server returned to the form,
leaving its journal `upload_result_uncertain`; the batch stopped
without submitting the 15 later IDs. A subsequent
`upload --replace-existing --confirm` made **no remote requests**:
the preflight failed on `20241008spu1650` because the old
`rs_20241008spu1650.nc` manifest entry's recorded checksum did not
match a file after manual renaming.

- [x] Introduce `milgrau-scc sounding-repair --id ID
  --dry-run|--confirm` to reconcile *only* legacy full-ID sounding
  names for **unjournalled** packets. It verifies original recorded
  checksums, files and NetCDF identity, stages a correct
  `rs_YYYYMMDDcccNN.nc`, revises the raw
  `Sounding_File_Name` and digest, validates locally, saves the old
  cache in `.cache/scc/legacy-sounding/ID` and atomically publishes.
  No SCC GET/POST/DELETE or scientific L0/L1 modifications.
- [x] Block any repairs with **any** SQLite submission journal row,
  including uncertain/accepted. Refuse other manual content edits
  not matched by original SHA-256 rather than rewriting hashes.
- [x] Regression tests cover preview non-mutation, half-completed
  physical file rename, original old physical file, journal block
  and tamper refusal.
- [ ] Operator: `milgrau-scc status --id 20241220spu0414 --remote`,
  verify portal status, then reconcile the uncertain *remote*
  upload before a second upload or any batch replacement.
- [ ] Validate repaired `20241008spu1650` cache and full 18-packet
  `upload --dry-run` before any authorized single-ID reattempt.

**SCC sounding-repair bulk usability (operator request, 2026-10-18):**
The original `milgrau-scc sounding-repair` required `--id` and
a protected/corrupt ID interrupted repairs across the other stages.
It now accepts an **optional** `--id`. Without it,
`sounding-repair --dry-run` and `--confirm` traverse all complete
pending stage directories in deterministic order, report each
`CURRENT/REPAIRABLE/REPAIRED/BLOCKED/ERROR`, and print totals.
Any unverified checksum or journal-protected ID is isolated to
that ID rather than aborting work on unrelated IDs; the summary
exits nonzero if actual `ERROR` items occurred.
The same per-ID atomic backup and verification remains in place.
No code path requests SCC remote upload, DELETE, or clears durable
`uncertain` journal state. Tests cover four mixed statuses and
continuation, old existing repaired packets and single-ID and
bulk no-network paths.

**Gate 12.7:** End-to-end SCC interoperability confirmed with real responses.
