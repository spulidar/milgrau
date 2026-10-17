# Phase 12 — Optional SCC integration and publication workflow

**Branch:** `scc-integration-refactor` (base: `milgraumaxxing`)  
**Status:** Phases 12.1–12.6 implemented and cross-platform CI validated; live SCC processed-results download and replacement still pending; Phase 12.7 not started  
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
- [ ] **Live read-only SCC processed-result download** with the SPU operator; no live SCC uploads, replacements or deletions during development.

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
**live remote SCC result download is still an explicit open gate**.
No real SCC upload, deletion or replacement POST occurred.

**Gate 12.6:** SCC submissions and external scientific results remain
traceable after cache cleanup; uncertain states and original scientific
products are never deleted by SCC cleanup.

## 12.7 — Validation and release

- [ ] CI across supported Python versions and operating systems, including installation without optional SCC extra.
- [ ] Test disabled integration and enabled mode without credentials.
- [ ] Live upload test: system 1047 day and system 1046 night; verify LR ancillary acceptance.
- [ ] Test real historical configurations, crossing midnight/month and repeated day/night segments.
- [ ] Compare retrieved SCC products under matched inputs with MILGRAU processing.
- [ ] Update README/configuration reference/CLI docs and retire old SCC producer paths.
- [ ] Require real upload/download evidence before declaring release ready.

**Gate 12.7:** End-to-end SCC interoperability confirmed with real responses.
