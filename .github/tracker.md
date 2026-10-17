# Phase 12 — Optional SCC integration and publication workflow

**Branch:** `scc-integration-refactor` (base: `milgraumaxxing`)  
**Status:** In progress — phase 12.1  
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
- [ ] Define/validate the strict `config.yaml` / `station.yaml` / `scc.yaml` ownership contract.
- [ ] Retain physical instrument metadata, calibration, observational climatology, overlap and scientific Raman characterization in `station.yaml`.
- [ ] Migrate SCC policy, configuration/channel mappings and SCC-only Raman mapping metadata into `scc.yaml`.
- [ ] Move processing and feature-switch policies to `config.yaml`, without copying observational values.
- [ ] Require `scc.yaml` only when `scc.enabled: true`; disabled mode must neither load nor validate SCC content.
- [ ] Preserve all historical station profile/date and solar day/night mappings.
- [ ] Provide strict station/SCC consistency checks, with no fabricated identifiers or silent aliases.
- [ ] Remove SCC export from the disabled LIBIDS path and decouple disabled scientific incremental validation from SCC derivatives.
- [ ] Update unit/contract tests and confirm no SCC extras are required for scientific processing.
- [ ] Document the new configuration ownership and migration.
- [ ] Run all applicable CI tests and review failures before marking 12.1 complete.

**Gate 12.1:** Fresh install without SCC extras can process L0/L1/L2 with `scc.enabled: false` and without `scc.yaml`; enabled mode preserves historical mapping.

## 12.2 — Product registry and climatological LR

- [ ] Migrate all SCC configuration IDs 248, 294, 484, 565, 1046, 1047 and their channel IDs.
- [ ] Enter verified SCC product IDs and physical product types/use cases/wavelengths from screenshots.
- [ ] Support legitimate reuse of product IDs by different systems.
- [ ] Resolve multiple 1064 nm product variants where their use case matters.
- [ ] Read monthly station climatology and stated spread without copying values into SCC configuration.
- [ ] Generate compliant LR ancillary NetCDF for eligible elastic products only.
- [ ] Preserve Raman retrieval independence and ensure ancillary ID/schema/altitude contracts.
- [ ] Test representative day/night and historical configurations.

**Gate 12.2:** Every SCC product and LR ancillary is traceable to real SCC product IDs.

## 12.3 — LIBIDS staging separation

- [ ] Separate canonical L0 incrementality from SCC export validity.
- [ ] Stage SCC-only L0, optional LR and manifest under `.cache/scc/`.
- [ ] Preserve source session and profile/segment metadata.
- [ ] Handle month boundaries, regime splits, multiple segments and ID collisions without splitting scientific sessions.
- [ ] Use atomic writes and SCC-only reexport for historical L0.
- [ ] Define safe treatment of old `*_L0_scc.nc` under scientific folders.

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
