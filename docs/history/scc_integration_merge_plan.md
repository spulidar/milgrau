> **Archived after integration to `main` (11 November 2026).** This is a historical plan, not a current runbook. The active scientific/operational acceptance gates are in [the tracker](../../.github/tracker.md).

# MILGRAU SCC + atmosphere: temporary integration and merge gates

**Work branch:** `integration/scc-atmosphere-20261111`  
**Source branches:** `scc-integration-refactor` (`a287ef05`) and
`feat/cds-queue-atmosphere-fallback` (`f8c7a3db`, which contains the
current `main` scientific changes). No production branch was changed.

## Scope retained

- Scientific sessionization, persistent raw Licel header index, and
  independent SCC-only reexport of valid canonical Level 0 data.
- Opt-in SCC native staging/validation/status/upload/download, append-only
  journal, audit archives, and fail-closed uncertain remote operations.
  Never auto-submit, auto-delete, or silently replace a remote SCC record.
- Current Level 1 and Level 2 science, including productive radiosonde use,
  IGRA2 historical archive, and explicit provenance through Level 2.
- Normal atmospheric priority: validated cached ERA5 / permitted CDS request,
  then observed Wyoming, local NOAA IGRA2, and finally identified USSA76.
  `--radiosonde-only` bypasses CDS while preserving the observed fallback
  chain and identified USSA76 last resort.
- Best-effort global CDS queue signal. When queue is above the configured
  heuristic threshold, keep cached ERA5 but do not initiate new CDS jobs.
  Failure to read the public dashboard is *unknown*, never *low*.
- Optional SCC sounding is always an **observed** cached profile: never derive
  it from ERA5/USSA76. Use the LIPANCORA-validated observed launch, provider,
  cache identity and tolerance; refuse contradictory provider or station
  provenance. Existing SCC history remains immutable.

## Explicitly excluded / deferred

- Experimental ERA5 monthly GRIB planner, `cfgrib` dependency, CDS queue
  ledger/throttle and global prefetch proposed in the older SCC branch.
  The existing ERA5 IO/cache path is retained.
- Changes to canonical production `main`, remote SCC uploads/deletes,
  expensive CDS downloads, manipulation of historical archive files, or
  automatic remote retries in this integration review.

## Before any merge to main

1. **Automated gate:** run Ruff, Python 3.12/3.14 pytest on Linux/Windows and
   Core Web Explorer. Zero failures on the *final commit*. The temporary
   integration branch is included in CI push triggers during verification.
2. **Scientific gate (requires original archive):** select representative
   real sessions across instrument eras, midnight/month boundaries, and
   day/night regimes. Record raw Licel identifiers and SHA-256, L0/L1/L2
   hashes, QA selection, dark-current associations, session IDs, optical
   retrievals and atmospheric provenance. Compare before/after under identical
   inputs and atmospheric policy. Explain any material scientific difference.
3. **Atmosphere gate:** test both congested and normal queue modes, cache-hit
   ERA5, failure/timeout of dashboard, Wyoming unavailable, valid IGRA2,
   no observed sounding, and all-USSA76 fallback. Check no CDS submissions
   when `--radiosonde-only` or cache-only ERA5 is selected. The public queue
   threshold is heuristic, not a predicted job wait time.
4. **SCC gate (non-destructive first):** with `scc.enabled: true`, stage one
   known science session, run `milgrau-scc upload --dry-run`, inspect its
   manifest and native NetCDFs (sounding origin/time/vertical profile) and
   validate original canonical L0/L1/L2 SHA-256 unchanged. Read-only remote
   `status --id ID --remote` may follow when authorized. Actual remote
   POST/DELETE and replacement require **separate explicit approval**.
5. **Optional production SCC validation:** after a separately approved safe
   test ID has ELPP/ELDA completion, validate downloaded remote results and
   immutable generation checksums. Prior mocked CI evidence is insufficient.
6. **Final review:** check merge diff for credentials, local data, state/journal,
   cache, NetCDF, accidental requirements and ambiguous provenance. Confirm
   `scc.enabled: false` by default. Do not merge until scientific and
   operational gates are explicitly accepted.

## Repeatable local commands

Use a disposable workspace and preserve originals. Replace `SESSION_ID` and
`MEASUREMENT_ID` with the exact selected records:

```bash
milgrau-libids -i SESSION_ID
milgrau-lipancora -i SESSION_ID --radiosonde-only
milgrau-lebear -i SESSION_ID

# Explicit opt-in in config.yaml is required for SCC staging:
milgrau-scc upload --id MEASUREMENT_ID --dry-run
milgrau-scc status --id MEASUREMENT_ID
```

To validate ERA5 preference under normal conditions, separately run
`milgrau-lipancora -i SESSION_ID` with preserved baseline outputs; both
modes normally write the same canonical L1 name, so do not overwrite
comparison evidence. Any real remote SCC operation is out of scope here.

**Status:** integration code under review in this temporary branch; automated
and real-data validation determine final merge readiness. No new PR is
required or created by this plan.
