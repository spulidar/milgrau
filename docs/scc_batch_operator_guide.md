# SCC operator guide — session-local inputs and explicit uploads

The SCC is an **optional external service**, enabled for testing in the
`scc-integration-refactor` branch. No LIBIDS/LIPANCORA call uploads to SCC.

## Filesystem layout

A continuous scientific session can contain several day/night or UTC-month
SCC Measurement_IDs. The canonical science products remain independent.

```text
02-processed_data/spu/YYYY/MM/spu_.../
├── spu_..._L0.nc
├── spu_..._L1.nc
├── spu_..._L2.nc
├── figures/
└── scc/
    ├── scc-inputs/
    │   ├── 20240913spu1549/
    │   │   ├── 20240913spu1549.nc
    │   │   ├── lr_20240913spu15.nc
    │   │   └── manifest.json
    │   └── <other Measurement_ID>/
    └── scc-results/
        └── <Measurement_ID>/generations/remote-<hash>/
            ├── elpp.archive
            ├── elda.archive
            └── results.json
```

An optional `rs_YYYYMMDDcccNN.nc` sounding file can also be in each input.
The input directory is session-local, not `.cache/scc/pending/`.
The append-only SQLite operations journal and explicit remote archives remain
in `.state/scc/`, ignored by Git. Scientific processing does not depend on it.

**Existing legacy `.cache/scc/pending/` packages are not moved or erased
automatically.** Re-export from the canonical sessions or move verified
packages after checking manifest identity, hashes and the destination. Keep
the original local evidence until reconciled.

## Standard, non-destructive operations

```bash
milgrau-libids
milgrau-lipancora
milgrau-scc upload --dry-run
milgrau-scc status
milgrau-scc status --remote
milgrau-scc download

# Exact SCC ID:
milgrau-scc status --id MEASUREMENT_ID --remote
milgrau-scc download --id MEASUREMENT_ID

# Send one new ID; never silently replace existing remote IDs:
milgrau-scc upload --id MEASUREMENT_ID --confirm

# Send all staged absent IDs, skipping remote-existing ones:
milgrau-scc upload --confirm
```

Bulk status/download examines locally staged and journalled IDs; it does
not enumerate every record held by the remote SCC. Bulk download only
fetches remotely completed ELPP and ELDA (both status code 127).

## Explicit same-ID replacement — backup is OPTIONAL

```bash
# Inspect the remote ID (read-only):
milgrau-scc status --id MEASUREMENT_ID --remote

# Optional: manually preserve a remote snapshot before replacement:
milgrau-scc status --id MEASUREMENT_ID --remote --backup

# Intentionally delete and re-upload without requiring a remote backup:
milgrau-scc upload --id MEASUREMENT_ID --replace-existing --confirm
```

`--confirm` paired with `--replace-existing` explicitly waives the
pre-deletion backup requirement. **Existing remote inputs/results can be
irreversibly lost.** This is not an atomic operation: DELETE may succeed
and the subsequent POST may fail. Remote identity, fingerprint, staged
file checksums and durable event recording still apply.

With no `--id`, `upload --confirm --replace-existing` can replace
**every staged ID also present remotely**. Use a single ID first if
that is not what you intend.

A response lost after DELETE or POST is *uncertain*, not proof of
success/failure. Ordinary `--confirm` does not silently clear a prior
uncertain journal entry or retry its destructive operation. In
particular, no-backup replacement cannot use the archive-based
recovery command. Inspect SCC remotely and reconcile that ID.

## Sounding, historical maintenance and cleanup

New packets can be finalized after LIPANCORA using cached *observed*
radiosondes. Filenames must match the shorter SCC formats:
`lr_YYYYMMDDcccNN.nc` and `rs_YYYYMMDDcccNN.nc`.
Neither ERA5 nor USSA76 is used as a fabricated observation.

```bash
# Legacy unsubmitted staged inputs, when applicable:
milgrau-scc sounding --dry-run
milgrau-scc sounding --confirm

# Remove verified, downloaded input package only if desired:
milgrau-scc cleanup --id MEASUREMENT_ID --dry-run
milgrau-scc cleanup --id MEASUREMENT_ID --confirm
```

Results in `session/scc/scc-results/` are not removed by SCC input cleanup.
Configuration `scc.cleanup.policy: manual` retains staged input by default.
See [technical SCC details](scc_native_client.md) for rare recovery and
legacy migration procedures.
