# SCC operator guide — current workflow

The SCC adapter manages locally staged measurements in
`.cache/scc/pending/<Measurement_ID>/`. Canonical LIBIDS L0/L1/L2,
source Licel files, downloaded SCC products, and the append-only
`.state/scc` journal are independent of the disposable input cache.
No `milgrau-scc` command uploads automatically: an explicit
`--confirm` is required.

## Standard flow

```bash
cd ~/milgrau
git status --short
git switch scc-integration-refactor
git pull --ff-only
source .venv/bin/activate
python -m pip install -e ".[scc]"

# New sessions, only when required:
milgrau-libids
milgrau-lipancora

# Preflight is entirely offline:
milgrau-scc upload --dry-run
milgrau-scc status

# Read-only remote check for an exact ID:
milgrau-scc status --id MEASUREMENT_ID --remote

# One new/absent ID (preferred for an SCC sounding smoke test):
milgrau-scc upload --id MEASUREMENT_ID --confirm

# All pending: skip existing remote IDs; invalid local IDs are reported
# and excluded. Does NOT replace an existing SCC measurement:
milgrau-scc upload --confirm

# Opt-in: after a remote problem, proceed to other independent IDs.
# The failed/uncertain ID is NOT retried; review the per-ID journal:
milgrau-scc upload --confirm --continue-on-error
```

A remote-existing ID is skipped with an ordinary batch upload, even
if its local submission journal is uncertain; skipping is strictly
read-only and does not resolve the uncertainty. A corrupt packet no
longer prevents unrelated, valid packets from being processed, but the
overall batch exit status is nonzero if any ID failed. By default a
remote/POST/DELETE error stops the live batch; opting into
`--continue-on-error` only isolates that failure and continues with
other IDs. **No POST/DELETE is ever blindly retried.**

## Same-ID replacement is a separate, destructive operation

```bash
milgrau-scc status --id MEASUREMENT_ID --remote
milgrau-scc status --id MEASUREMENT_ID --remote --backup
milgrau-scc upload --id MEASUREMENT_ID --replace-existing --confirm
```

The replacement creates and verifies a local archive of the existing
SCC record and retrievable results **before modifying the durable
journal revision and before any remote deletion**. If staged content
matches the archived remote inputs, the operation is a no-op. Otherwise
it deletes and reuploads the **same SCC ID** using a non-atomic
server-side sequence. A failure or ambiguous remote outcome means
**inspect that ID**; the journal prevents a repeat POST or DELETE.
The remote record can disappear after a successful DELETE if a later
upload fails. Do not use batch replacement before completing a
successful controlled single-ID replacement on the live SCC.

An existing accepted/downloaded journal record is not silently
reset because somebody used `--replace-existing`; a new explicit
revision starts only after the remote snapshot is independently
verified. Remote absence during a replacement does not authorize
blindly creating a new measurement.

## Sounding and validation

The SCC form requires the following auxiliary names:

- LR: `lr_YYYYMMDDcccNN.nc` (example `lr_20241008spu16.nc`)
- Sounding: `rs_YYYYMMDDcccNN.nc` (example `rs_20241008spu16.nc`)

`<Measurement_ID>.nc` remains the raw SCC NetCDF name.
`Sounding_File_Name`, `LR_File_Name`, file content and manifest
SHA-256 must agree. With sounding enabled, a **new** LIBIDS packet is
finalized by LIPANCORA from the same validated observed radiosonde
in its scientific L1 QA. No ERA5/USSA76 model profile is fabricated.

Historical pending packets do not retroactively receive soundings
simply because LIPANCORA was rerun. For **unsubmitted** old packets:

```bash
milgrau-scc sounding --dry-run
milgrau-scc sounding --confirm
milgrau-scc upload --dry-run
```

Older filenames or interrupted one-time migrations may need the
isolated offline maintenance commands documented in
[`scc_native_client.md`](scc_native_client.md).
Never change a journalled/uncertain package's bytes or SHA-256
manually to bypass its known provenance.

## Result download and retained cache

```bash
milgrau-scc status --id MEASUREMENT_ID --remote
milgrau-scc download --id MEASUREMENT_ID --keep-cache
```

Keep `scc.cleanup.policy: manual` in your **local**
`config.yaml`; local configurations previously using
`after_download` override repository defaults. `--keep-cache`
also prevents cleanup for this download. Never call
`milgrau-scc cleanup --id ID --confirm` unless explicitly
choosing to remove that verified input cache.

SCC products are saved under each scientific session's
`scc-results/<ID>/generations/` and do not rely on the pending cache.

## Incident diagnosis

```bash
milgrau-scc status --id MEASUREMENT_ID --remote
```

Remote status is read-only, and **remains available even when the
local staged file is invalid**. The command reports the local
validation problem and exits nonzero while still showing the remote
record. Inspect `.state/scc` submission history and archive before
a replacement/recovery. A local HTML validation rejection or uncertain
HTTP response is *not* proof the SCC accepted or rejected the POST.

For historic details from the 2026-10-17 LR/sounding filename
migration and the first interrupted replacement attempts, refer to
[the SCC integration tracker](../.github/tracker.md) and the
append-only local journal, not to instructions for the current normal run.
