# MILGRAU configuration and scientific ownership

MILGRAU separates **processing choices**, **observational reality**, and **implemented physics** so provenance remains readable and scientific behavior does not depend on duplicated hidden defaults.

This document defines ownership. It deliberately does not copy the current numerical values of mutable thresholds or calibration constants; inspect the resolved YAML/product provenance for the exact values used in a run.

## Ownership rule

| Owner | What belongs here | What does not belong here |
| --- | --- | --- |
| `config.yaml` | processing/scientific recipe chosen for a run | station history, instrument calibration facts, physical equations |
| `station.yaml` | station/site/instrument reality, dated history, calibration identity, station-derived climatology/mapping, explicitly labeled provisional instrument estimates | generic processing policy, algorithmic equations |
| Python modules | equations, physical constants, validation logic, typed contracts and implementation mechanics | mutable station facts or silent scientific recipe defaults |
| external scientific source | ancillary data content such as ERA5/radiosonde | MILGRAU processing configuration |
| NetCDF provenance | resolved record of what actually ran | secrets, machine-local absolute paths, transient cache mechanics |

A value should have one authoritative owner. Copying the same scientific meaning into multiple layers creates ambiguous provenance and is avoided.

## `config.yaml` — processing/scientific recipe

`config.yaml` answers: **How should this measurement be processed?**

Typical responsibilities include:

- processing/incremental policy;
- input/output directory roles;
- Level 0 acquisition QA policy;
- Level 1 algorithm settings and atmosphere-source priority;
- optional external-service settings that are not credentials;
- Level 2 wavelengths requested;
- temporal block duration;
- gluing QA/search settings;
- Rayleigh-reference search/QA settings;
- productive KFS mode and Monte Carlo settings;
- cloud-screening enablement/configuration;
- visualization/QA settings.

Scientific stage readers are intentionally strict: missing productive settings should fail rather than silently manufacture an algorithmic default.

### Values deliberately excluded

Examples that should not migrate into `config.yaml` merely for convenience:

- station coordinates/altitude/timezone;
- dated laser/instrument periods;
- channel dead time/bin shift/calibration facts;
- telescope/FOV/beam/divergence/alignment values used to describe instrument overlap;
- station SCC channel IDs;
- station-derived lidar-ratio climatology when it represents the station record;
- CDS/API secrets.

## `station.yaml` — observational reality

`station.yaml` answers: **What instrument/site reality applies to this measurement time?**

Typical responsibilities include:

- station identity and geolocation;
- historical instrument periods and profile IDs;
- laser/acquisition facts tied to those periods;
- channel definitions and calibration parameters;
- dead time/bin shifts and other characterized instrument properties;
- instrument calibration identity;
- SCC mapping/IDs;
- station-derived lidar-ratio climatology/assumptions where this is part of the station scientific record;
- overlap receiver/transmitter geometry, including explicitly labeled estimates while characterization is pending.

Station history is date-resolved before processing. Downstream code should consume the resolved station profile instead of rediscovering history independently.

For overlap, the station may define a clearly labeled transmitter fallback when a historical profile lacks a more specific geometry. Profile-specific geometry takes precedence. The resolver records whether the effective values came from the profile or the station fallback. A fallback remains an estimate; using it does not promote the geometry to a calibration.

## Python — scientific implementation

Python owns behavior that must be tested and versioned as code rather than edited as site configuration:

- atmosphere and Rayleigh equations;
- lidar inversion equations;
- numerical kernels;
- generic geometrical-overlap equations and numerical integration;
- signal/error propagation formulas;
- support/missing-value logic;
- config validation and station/profile fallback resolution mechanics;
- channel/source selection mechanics;
- product contracts/schema assembly;
- provenance algorithms such as content identity;
- stable enums/flag meanings.

A physical constant or equation coefficient that is universal to the implemented model belongs in code. A characterized instrument parameter belongs in the station record. A provisional instrument estimate also belongs in the station record but must carry an explicit status. A scientific run choice belongs in the processing recipe.

The current overlap model is intentionally diagnostic: Python evaluates the station-owned geometry but does not apply an overlap correction to the signal. See `docs/overlap_model.md`.

## Canonical scientific owners

The detailed module inventory is in `docs/code_inventory.md`. Current productive ownership includes:

- atmosphere fallback physics — `milgrau.physics.atmosphere`;
- diagnostic geometrical overlap physics — `milgrau.physics.overlap`;
- overlap station validation/resolution — `milgrau.config.overlap`;
- Level 1 corrections — `milgrau.level1.corrections`;
- atmosphere source/materialization — `milgrau.level1.thermodynamics`;
- strict Level 2 recipe parsing — `milgrau.level2.config`;
- analog/PC gluing numerical kernel — `milgrau.level2.gluing`;
- productive signal/source selection — `milgrau.level2.signal_selection`;
- molecular/reference-search numerics — `milgrau.level2.molecular`;
- KFS inversion kernel — `milgrau.level2.kfs`;
- productive Rayleigh QA/backward retrieval — `milgrau.level2.optical_retrieval`;
- wavelength orchestration — `milgrau.level2.retrieval`;
- typed result assembly — `milgrau.level2.result_assembly`;
- Level 2 schema construction — `milgrau.level2.dataset`;
- reusable FAIR provenance — `milgrau.provenance`.

No visualization module is allowed to become a hidden scientific decision owner.

## External atmosphere sources

External data source identity is scientific provenance, while download/cache details are operations.

Current stable Level 2 provenance distinguishes:

- ERA5 — Copernicus provider + configured dataset + DOI/release identity when available;
- radiosonde — University of Wyoming upper-air service family plus separate station/time metadata;
- USSA76 — US Standard Atmosphere / standard atmosphere / edition 1976.

Transient cache filenames do not define the scientific source ID.

## Secrets and local paths

Secrets do not belong in either YAML or published product provenance. Service credentials must use the provider-supported user/environment mechanism.

Published provenance stores portable filenames/IDs and exact YAML content, not host-specific absolute paths. Operational logs may contain local paths where useful for debugging, but those paths are not scientific identity.

## Hash policy

Hashes are used only where there is a named consumer:

- `source_level1_sha256` identifies exact Level 1 bytes and protects Level 2 lineage/incremental reuse;
- `source_code_sha256` distinguishes installed MILGRAU code states sharing a package version.

Configuration hashes are intentionally not used as a substitute for readable configuration provenance. Exact processing/station YAML snapshots remain the human-readable record.

## Versioning consequences

Changing a YAML value does not automatically require changing a retrieval-method version; it normally changes the resolved run recipe and is captured in embedded YAML/provenance.

Changing a diagnostic overlap estimate in `station.yaml` therefore does not change method v3 while the curve is not used productively. Turning overlap into a productive signal correction or support decision would be a scientific semantic change and requires deliberate method/version review, tests, uncertainty treatment and provenance.

Changing a scientific equation, accepted support semantics, uncertainty interpretation or productive method policy **does** require deliberate method/version review. The current Level 2 schema version and retrieval-method version are separate so storage-only changes and scientific changes are not conflated.

Release version/DOI/tag alignment is finalized under the release-readiness gate in `tracker.md`.
