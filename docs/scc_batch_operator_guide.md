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
particular, an interrupted **DELETE** can be reconciled without a backup
*only* if the same journal revision records the explicit no-backup waiver,
its remote fingerprint and the uncertain DELETE outcome. After independently
checking the SCC portal and confirming API absence for that ID, use:

```bash
milgrau-scc upload --id MEASUREMENT_ID --resume-after-delete --confirm
```

This does **not** repeat DELETE; it performs at most one new POST and verifies
the resulting remote record. If the ID is present, staged hashes differ,
the waiver/audit trail is missing or the SCC lookup fails, it aborts without
uploading. Any uncertain POST remains blocked; never repeat `upload --confirm`
to guess the outcome. The historical second-POST repair options have been
removed; an uncertain POST must be independently investigated rather
than resubmitted through a generic retry switch.

## Simplified SCC CLI and HTTP configuration

The three supported commands are `upload`, `status`, and `download`.
For remote HTTP requests, `config.yaml` has an explicit read-only retry
configuration:

```yaml
scc:
  enabled: true
  config_file: "scc.yaml"
  state_dir: ".state/scc"
  http:
    timeout_seconds: 60
    get_retries: 2
```

`get_retries` applies **only to GET**, not to login, upload, or DELETE
POSTs. `scc.upload.automatic`, `scc.upload.retries` and
`--continue-on-error` have been removed. Every batch stops on its
first operational SCC/journal error. Local input validation still checks
each staged package before making remote requests.

## Soundings and operational input retention

New packets are finalized automatically after LIPANCORA when observed
radiosondes are enabled in `scc.yaml`. Filenames must match SCC's
`lr_YYYYMMDDcccNN.nc` and `rs_YYYYMMDDcccNN.nc` formats.
Only independently verified observations may be included; neither ERA5
nor USSA76 may be fabricated as a measured sounding.

There are three main commands: `upload`, `status`, and `download`.
The former `sounding`, `sounding-repair`, `upload --migrate-cache`,
`upload --repair-lr-cache` and incident-only retry options are removed.
Use `milgrau-scc upload --dry-run` for a read-only validation of the
current session inputs. Only `upload --id ID --resume-after-delete --confirm`
remains as guarded reconciliation of a genuinely uncertain DELETE;
uncertain POST outcomes never trigger automatic retries.

Inputs in `<Session_ID>/scc/scc-inputs/<Measurement_ID>/` are now
**persistent**, not cache. The SCC CLI never deletes them after upload
or download, and neither `cleanup` nor `download --keep-cache` exists.

If local disk space requires manual input removal, identify the **exact**
session and Measurement_ID, confirm the remote status and full journal
(including any uncertain POST/DELETE), and preserve any needed hashes,
manifest, scientific provenance and historical evidence first. Do not
delete an input needed for retry/reexport/reconciliation. Only the
operator may remove that specific `scc-inputs/<Measurement_ID>/` directory.
Never treat `.state/scc/`, `scc-results/`, canonical L0/L1/L2, figures or
raw Licel measurements as removable SCC input cache.

See [technical SCC details](scc_native_client.md) for rare recovery and
legacy migration procedures.
