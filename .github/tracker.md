# Phase 12 — Optional SCC integration and publication workflow

**Branch:** `scc-integration-refactor` (base: `milgraumaxxing`)  
**Status:** Phases 12.1–12.3 implemented and cross-platform validated; Phase 12.4 pending  
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

- [ ] Add `.[scc]` extra and `milgrau-scc` CLI with only `upload`, `status`, `download`, `cleanup`.
- [ ] Validate real SCC protocol/authentication and implement safe HTTP/TLS, timeouts and bounded retry.
- [ ] Keep credentials out of repository and configuration files.
- [ ] Validate pending submission and support `upload --dry-run`.
- [ ] Include raw and ancillary uploads; mock all network tests.

**Gate 12.4:** Client passes offline integration tests; real upload remains opt-in.

## 12.5 — Durable submission state

- [ ] Track original session, SCC Measurement ID, configuration/products, hashes, timestamps and remote IDs outside cache.
- [ ] Represent pending/uploading/accepted/processing/completed/failed states explicitly.
- [ ] Guarantee retry reconciliation, duplicate protection and process locking.
- [ ] Make `status` useful offline and online.

**Gate 12.5:** Interrupted upload cannot silently cause duplicate remote submission.

## 12.6 — Download and cleanup

- [ ] Download/verify SCC ELPP/ELDA products to distinct persistent locations.
- [ ] Persist remote provenance, processing configuration and result links.
- [ ] Clean only safe staged files after confirmation and download, retaining failures/uncertain uploads.
- [ ] Never erase raw Licel, scientific NetCDF or durable submission history.

**Gate 12.6:** Submission and external scientific results remain traceable after cleanup.

## 12.7 — Validation and release

- [ ] CI across supported Python versions and operating systems, including installation without optional SCC extra.
- [ ] Test disabled integration and enabled mode without credentials.
- [ ] Live upload test: system 1047 day and system 1046 night; verify LR ancillary acceptance.
- [ ] Test real historical configurations, crossing midnight/month and repeated day/night segments.
- [ ] Compare retrieved SCC products under matched inputs with MILGRAU processing.
- [ ] Update README/configuration reference/CLI docs and retire old SCC producer paths.
- [ ] Require real upload/download evidence before declaring release ready.

**Gate 12.7:** End-to-end SCC interoperability confirmed with real responses.
