# SCC native client — Phases 12.4–12.6

MILGRAU uses its own minimal HTTPS transport, **not** the historical
`scc_access` package. The native client uses the publicly documented SCC
Django Quick upload form and read-only measurement API. Remote protocol
acceptance with the SPU account has **not yet been demonstrated**.

Official SCC user documentation:
https://docs.scc.imaa.cnr.it/en/latest/data_processing/uploading_ancillary.html

Public protocol reference for the SCC form/API fields:
https://repositories.imaa.cnr.it/public/scc_access/file/4ef6f2102a61/scc_access/scc_access.py

## Installation and ownership

Normal L0/L1/L2 MILGRAU installation never requires the SCC HTTP stack.
For SCC remote operations:

```bash
python -m pip install -e ".[scc]"
```

`scc.yaml` contains the public server address only, at
`server.base_url` (must be an HTTPS origin without embedded credentials).
The default is `https://scc.imaa.cnr.it/`. This can be changed to the
specific deployed SCC HTTPS origin after verification.
`config.yaml` enables the optional integration with `scc.enabled: true`
and controls `scc.upload.timeout_seconds` and `scc.upload.retries`.
Upload retries apply **only to safe GET requests**; multipart POSTs are
never blindly retried.

Web-login credentials must be provided in the **local environment**:

- `MILGRAU_SCC_USERNAME`
- `MILGRAU_SCC_PASSWORD`

If the site also requires HTTP Basic authentication, additionally use:

- `MILGRAU_SCC_BASIC_USERNAME`
- `MILGRAU_SCC_BASIC_PASSWORD`

### HTTP 401 before SCC website login

The SCC access tool's public documentation distinguishes **two independent
credential pairs**: `basic_credentials` for the HTTP access gate and
`website_credentials` for the Django login. A 401 response **when GETting
`/accounts/login/`** happens before website credentials are submitted;
it is not evidence that the Django username/password are wrong.

For a safe shell session in zsh, after entering website credentials, request
HTTP Basic values separately from the operator/system administrator:

```bash
read -r "MILGRAU_SCC_BASIC_USERNAME?SCC HTTP Basic user: "
read -rs "MILGRAU_SCC_BASIC_PASSWORD?SCC HTTP Basic password: "
echo
export MILGRAU_SCC_BASIC_USERNAME MILGRAU_SCC_BASIC_PASSWORD

# Read-only login + measurement query: NO UPLOAD.
milgrau-scc status --id 20240913spu1549 --remote

# Clear both pairs after the test.
unset MILGRAU_SCC_USERNAME MILGRAU_SCC_PASSWORD
unset MILGRAU_SCC_BASIC_USERNAME MILGRAU_SCC_BASIC_PASSWORD
```

If the user has no separate HTTP Basic account, first confirm the correct SCC
HTTPS address and ask its administrator for the HTTP-layer credentials. Do
**not** assume the website username/password are interchangeable with Basic.
Never disable HTTPS certificate validation, share passwords in logs/screenshots
or place secrets in committed YAML. A 401 without a Basic
`WWW-Authenticate` challenge can also be an upstream proxy/WAF policy
or wrong endpoint, so verify rather than guessing.

Never put these values in `scc.yaml`, `config.yaml`, GitHub, a NetCDF,
an issue, a screenshot, or a copied command containing literal secrets.
Use a restricted SCC upload-only user where possible. TLS certificate
verification is **always on**.

## Four operator commands

```bash
# Upgrade older local raw filenames and fix deployed SCC LR form validation.
milgrau-scc upload --migrate-cache
milgrau-scc upload --repair-lr-cache
milgrau-scc upload --dry-run

# Explicit single-ID upload-only retry after local LR filename correction,
# original SCC backup validation and portal/API absence; NOT batch-safe:
milgrau-scc upload --id 20240913spu1549 --retry-after-lr-fix --confirm

# After a confirmed portal/API absence and recovery_upload_uncertain,
# one explicitly authorized upload-only retry (never deletes):
milgrau-scc upload --id 20240913spu1549 --retry-after-upload --confirm

# Validate every staged packet, offline; no SCC credentials or network needed.
milgrau-scc upload --dry-run

# Validate one specific SCC Measurement_ID locally.
milgrau-scc upload --id 20240913spu1549 --dry-run

# Conservative sequential live batch, one SCC ID at a time:
# Submit remote-absent IDs; skip every already-existing remote ID.
milgrau-scc upload --confirm

# Explicitly opt in to replacing existing remote IDs in batch
# ONLY after the single-ID destructive trial has been approved and verified.
milgrau-scc upload --confirm --replace-existing

# Confirm a single real upload, or exact-ID guarded destructive replacement.
milgrau-scc upload --id 20240913spu1549 --confirm
milgrau-scc upload --id 20240913spu1549 --replace-existing --confirm

# Local pending files and integrity.
milgrau-scc status

# Read-only remote state for one ID (requires credentials).
milgrau-scc status --id 20240913spu1549 --remote

# Read-only SCC ELPP/ELDA download for an owned, journalled upload.
milgrau-scc download --id 20240913spu1549

# For an older remote measurement never submitted by MILGRAU,
# give its verified scientific session ID explicitly. This never enables cleanup.
milgrau-scc download --id 20240913spu1549 --session spu_20240913-1549Z_20240913-1750Z

# Keep cached raw/LR even after a successful confirmed-owned download.
milgrau-scc download --id 20240913spu1549 --keep-cache

# Preview or explicitly remove only eligible pending staging bytes.
milgrau-scc cleanup --id 20240913spu1549 --dry-run
milgrau-scc cleanup --id 20240913spu1549 --confirm
```

The initial Phase 12.4 only implemented single-ID upload. In Phase 12.7,
the explicit `upload --confirm` command now supports a **sequential batch**
of all pending local packages. Local integrity preflight covers the whole
batch before any network work. By default, pre-existing remote IDs are
**skipped without deletion or journal ownership claims**. Only
`upload --confirm --replace-existing` authorizes a batch to replace
remote-existing IDs after verified original-data backups. A failed,
interrupted or uncertain operation stops subsequent IDs; partial successes
remain recorded in the durable journal, and no network POST is retried
automatically. A final summary reports submitted, replaced, identical,
skipped, failed and unprocessed counts. Single `--id` operations remain
available and are the required first live replacement test.

For a current operator workflow and exact runbook, read
[batch submission and first replacement test](scc_batch_operator_guide.md).
Setting `scc.upload.automatic: true` **does not** cause LIBIDS to perform
a POST.

## Local preflight

`upload --dry-run` checks the immutable staged package manifest's SHA-256,
file sizes, SCC `Measurement_ID`, SCC configuration ID, channel_ID existence,
LR filename correspondence, eligible optical product IDs, dimensions and
numeric LR profiles. A corrupt/mismatched packet is rejected without network
requests or reading credentials.

The native client logs in with Django CSRF protection and optional HTTP Basic
credentials, checks the read-only
`api/v1/measurements/<Measurement_ID>/` endpoint for an existing record,
and submits the original NetCDF using multipart `data` alongside an optional
`lidar_ratio_file`. It treats the upload as confirmed **only** when a
subsequent read-only API lookup finds the measurement. HTTPS/TLS verification
is enabled, HTTP redirects are restricted to the configured origin, unsafe
307/308 POST repeats are forbidden, and no form credentials are written
to outputs.

## Phase 12.5 — Durable tracking and same-ID replacement

The SCC `Measurement_ID` **does not change** when a replacement uses
different source timestamps/profiles or a new climatological LR. The native
client follows the documented `scc_access --force_upload` semantics:
**backup original remote data → delete old SCC record → upload new bytes with
the same ID**. SCC does not offer transactional overwrite, so this flow is
intrinsically non-atomic.

A private persistent SQLite journal lives in
`.state/scc/submissions.sqlite`, explicitly **outside** `.cache/scc` and
excluded by Git. It records scientific session, channel configuration,
product IDs, source segments and periods, hashes of raw/LR NetCDF,
transitions and timestamps, remote identity, archive location, and
the last error. An exclusive per-ID lock prevents concurrent local uploads.
A stale lock after a crash **must be investigated before manual removal**.

A safe read-only preflight exists **before** any destructive replacement.
It authenticates and downloads/verifies the remote raw and available products
into persistent state, but never modifies the server:

```bash
milgrau-scc status --id 20240913spu1549 --remote --backup
```

If this fails because an original auxiliary LR is unavailable, do not use
`--replace-existing` until a reliable SCC LR retrieval method has been
established. The backup operation can be run repeatedly and does not
change scientific L0/L1/L2. The archived data is retained outside cache.

A replacement is never automatic. It requires precisely:

```bash
milgrau-scc upload --id 20240913spu1549 --replace-existing --confirm
```

**Do not run that command against live SCC until the backup protocol and
permissions have been verified on an approved test measurement.** It is a
destructive replacement operation; `--confirm` authorizes deletion of the
existing SCC record after all safeguards. No other SCC records or related
measurements are intentionally deleted.

The replacement workflow:
1. Validate the complete pending local package and its SHA-256 checksums.
2. Query SCC and reject queued/active processing or unexpected remote
   changes. An absent ID is not treated as a replacement.
3. Download and verify the old raw NetCDF or ZIP, and each available remote
   result archive (ELPP, ELDA, etc.) into
   `.state/scc/archives/<Measurement_ID>/<timestamp>/`.
   If the previous raw file references an external LR, archive it too.
   If SCC does not expose a verifiable LR download, **refuse deletion**.
4. Recheck the original SCC record fingerprint; compare the actual old
   NetCDF data/period/LR to local data. Provably identical content is a
   **no-op**, without deletion or second upload. Uncertain equivalence is
   *not* assumed identical.
5. Only then POST to SCC's administrative
   `admin/database/measurements/<ID>/delete/` page using the documented
   `not_delete_related` value. Query API for confirmed disappearance.
   Upload the new raw NetCDF and optional LR with the **same ID**, then
   require API confirmation.
6. Record every transition in SQLite. After a crash, deletion timeout or
   uncertain upload, the state becomes `uncertain` or is left in an
   interrupted state. The command must never automatically retry it.

A pre-existing SCC record is **not** considered to have been uploaded by
MILGRAU merely because remote `status` shows ELPP/ELDA results.
`milgrau-scc status` shows the local queue and durable journal;
`status --id ID --remote` shows safe remote metadata and local state
without exposing the full station object or credentials.

**Limitations requiring live verification:** the official SCC version may
return raw files or ancillary LR downloads in a different container/schema.
If the original data cannot be archived and verified locally, the client
stops before deletion. Upload of a new record and admin-delete endpoints are
mock-tested, but have not been exercised on the live SCC in this phase.
The SQLite journal cannot prevent independent SCC portal operators from
editing the measurement during the narrow interval after its final read;
there is no server-side atomic compare-and-swap. Use a restricted operator
maintenance window for destructive replacements.

Phase 12.6 implements verified read-only download and cache-only cleanup.
Old SCC backups and journal entries are persistent evidence and are never
removed by cache cleanup.

## LIBIDS → LIPANCORA → SCC observed sounding (opt-in in scc.yaml)

In the current SCC-enabled example, `scc_policy.sounding.enabled: true`.
`milgrau-libids` generates a pending raw `<Measurement_ID>.nc`
(`Molecular_Calc=0`) plus climatological
`lr_YYYYMMDDcccNN.nc` and a manifest with
`sounding_stage: awaiting_lipancora`. Such a packet is **not
uploadable yet**. Raw NetCDF creation remains within LIBIDS, independent
of the canonical scientific L0.

`milgrau-lipancora` then carries out its normal L1 thermodynamics and
radiosonde QA. After successfully writing L1 (or checking an already-current
L1), it can attach **only the same observed Wyoming radiosonde actually
selected and verified by LIPANCORA**: an existing observation CSV and
provenance JSON are read from the existing
`.cache/radiosonde/YYYY/MM/` location. It does not request data from
the network in this SCC packaging step, synthesize radiosondes from
ERA5/USSA76, or modify the scientific L1 itself.

For an eligible observed sounding, the SCC-only raw NetCDF is updated
with `Molecular_Calc=1` and
`Sounding_File_Name=rs_YYYYMMDDcccNN.nc`. A new
`rs_YYYYMMDDcccNN.nc` with station-relative altitude (m),
temperature (°C), pressure (hPa), original synoptic launch
time and radiosonde launch-site metadata is added. SHA-256 hashes in
the manifest are updated. The change occurs via an isolated temporary
copy and verified directory swap. The preimage stays safe during the
transaction; on failure the previous pending package is restored.
An existing `.state/scc` SQLite submission row (even pending,
failed or uncertain) blocks any alteration to the package.

When no matching valid observed sounding is available, the policy
`missing_policy: skip` finalizes the packet as
`sounding_stage: skipped_missing`, retaining `Molecular_Calc=0`,
or `missing_policy: fail` blocks completion until resolved.
`sounding_stage: ready` includes the verified `rs_YYYYMMDDcccNN.nc` file.
`milgrau-scc upload --dry-run` prevents uploads while a packet
is still `awaiting_lipancora`.

```bash
milgrau-libids
milgrau-lipancora
milgrau-scc upload --dry-run
milgrau-scc upload --confirm   # ONLY after an explicit operator decision
```

Only **new SCC packets staged by the enabled pipeline** enter this
two-stage handoff. Existing SCC archives, accepted measurements and
previously created pending input files are not silently changed or
resubmitted. To skip LIPANCORA sounding staging for new packets, set
`scc_policy.sounding.enabled: false`.

The current template's `config.yaml scc.cleanup.policy` is `manual`,
so verified result downloads keep the input cache by default. Existing
local configs still set to `after_download` should be explicitly edited
or call `milgrau-scc download --id ID --keep-cache`.

**Live sounding acceptance by SCC has not yet been tested.**
First validate one new measurement and the remote processing output
before authorizing a multi-ID batch.

**SCC sounding filename requirement:** Quick Upload rejects the complete
Measurement_ID suffix in a sounding filename. For example,
`20241008spu1650` is uploaded with `rs_20241008spu16.nc`
(not `rs_20241008spu1650.nc`). The sounding's basename is
validated at package creation and again immediately before upload.
For locally prepared packets created before this fix, manually renaming
the `rs_...` file **alone is insufficient**: its
`Sounding_File_Name` raw-NetCDF attribute, the corresponding
`manifest.json.files` key, and the raw/sounding SHA-256 and file-size
entries must also be consistent. Do not edit an accepted or journalled
submission to force a retry.

## Incident: mixed SCC batch failure after manual sounding rename

After the October 18 batch, two **independent** states must be kept
separate:

- `20241008spu1650` (previously skipped because it exists remotely)
  has a **local** legacy sounding filename/manifest checksum mismatch.
  Do not run the 18-ID `--replace-existing --confirm` batch merely to
  bypass this error.
- `20241220spu0414` attempted an SCC Quick Upload POST which returned
  to the HTML form rather than a confirmed measurement. The client
  correctly journalled `upload_result_uncertain` and stopped the
  remaining 15 IDs. The exact remote outcome must be checked through
  `milgrau-scc status --id 20241220spu0414 --remote` and SCC portal
  before any further remote attempt. Local filename repair does **not**
  clear that durable uncertainty or authorize a second POST.

For locally inconsistent **unjournalled** packets from the old
`rs_<Measurement_ID>.nc` generation, the repair accepts either an optional
single `--id`, or (by default) **all pending IDs**:

```bash
# Preview all 18 staged packets without changing anything:
milgrau-scc sounding-repair --dry-run

# Fix every eligible staged packet, without remote network operations:
milgrau-scc sounding-repair --confirm

# Or use the same operation on one specific measurement:
milgrau-scc sounding-repair --id 20241008spu1650 --dry-run
milgrau-scc sounding-repair --id 20241008spu1650 --confirm

# Always check SCC package integrity offline before any remote action:
milgrau-scc upload --dry-run
```

The bulk command does **not stop** when one ID has a mismatch or an SCC journal
record. It prints `REPAIRABLE`, `REPAIRED`, `CURRENT`, `BLOCKED` or `ERROR`
for **each** ID, with a count summary. When one or more packets return
`ERROR`, the process exits nonzero **after checking every ID**. Journalled
IDs that need correction are `BLOCKED` but do not prevent updating unrelated
unjournalled packets. This removes the *whole-batch* barrier without erasing
scientific provenance or resetting a potentially uncertain remote POST.
An already valid journalled packet is reported as `CURRENT`; no changes are made.

The repair accepts only the specific old sounding name and requires
the old SHA-256 and byte size **still match the original manifest**,
even if the operator has already moved the sounding to the short name
manually. It validates the raw NetCDF's `Measurement_ID` and
`Molecular_Calc`, and allows only an old or new
`Sounding_File_Name` matching that measurement. It creates a
candidate staging directory and changes just the filename reference
and raw file hash; the observed sounding and LR bytes are unchanged.
It verifies the complete candidate before publication, moves the old
folder into `.cache/scc/legacy-sounding/<ID>` as a durable preimage,
and performs an atomic rename with rollback if validation fails.
**Any existing SCC journal row** (including uncertain, accepted and
downloaded) blocks this operation. If the raw NetCDF has already
been manually edited without updating the manifest checksum, this
repair deliberately fails closed: inspect/recover the original copy,
rather than overwriting unverified provenance.

Even after the local dry-run is green, **do not repeat the batch until
the `20241220spu0414` uncertain remote upload has been explicitly
reconciled**. An unconfirmed POST should not be blindly retried.

## Explicit historical backfill (old pending SCC packets after LIPANCORA)

Older pending packets staged before the two-stage LIBIDS/LIPANCORA
handoff have `sounding_stage: not_requested` (or no stage entry).
Running LIPANCORA does **not** implicitly change them. To prepare
such packets from **existing** validated L1 observations without
reprocessing LIBIDS/LIPANCORA or contacting SCC:

```bash
# Preview each historical Measurement_ID and its source L1/observational QA:
milgrau-scc sounding --dry-run

# Optional single-ID preview:
milgrau-scc sounding --id 20241008spu1650 --dry-run

# Explicitly backfill every eligible historical pending packet, offline:
milgrau-scc sounding --confirm

# Final integrity validation (shows rs_YYYYMMDDcccNN.nc for eligible packets):
milgrau-scc upload --dry-run
```

The new `sounding` subcommand has only **dry-run** and **confirm**;
neither opens the SCC network client nor uploads measurements. In
preview, `READY` means L1 identifies a radiosonde observation that
can be retrieved from the corresponding Wyoming CSV/JSON cache
with matching station and target UTC time. The synoptic selection
uses the **scientific session midpoint**, matching LIPANCORA's QA
reference; it does not blindly compare against the first laser profile. `SKIP` means no eligible
observed sounding, no L1 or a packet already finalized; it is
left byte-for-byte intact. `BLOCKED` means any durable SQLite
submission state already exists for the ID; it is never modified.
`ERROR` identifies invalid/inconsistent scientific provenance or
staging and prevents changes.

With `--confirm`, only READY unsubmitted packets are revised,
using the atomic temporary directory/rollback safeguards. The
scientific L0, L1, original Licel inputs, L1 atmospheric data,
and LR file remain unchanged. The SCC-specific raw `<ID>.nc`
gains `Sounding_File_Name=rs_YYYYMMDDcccNN.nc` and `Molecular_Calc=1`;
the new observed sounding and revised raw hashes are recorded in
`manifest.json`. Its original SCC catalog hash remains recorded;
the backfill's current catalog hash is stored separately to avoid
misrepresenting its original production provenance.

**This is a local preparation step, not remote validation.**
IDs already submitted (including the accepted
`20240913spu1549`) are protected. Some of the remaining historical
packets may have no existing radiosonde observation; do not invent
a sounding or force it into the cache. Test *one* READY/new remote ID
after the local preview/backfill, before running any bulk SCC upload.

## Phase 12.6 — Verified results and disposable cache cleanup

SCC results are explicitly **external scientific outputs**, not replacements
for MILGRAU L0/L1/L2 or the station's own inversion. Download saves the
read-only SCC processed result archives under the original scientific session:

```text
02-processed_data/spu/YYYY/MM/<Session_ID>/
    <Session_ID>_L0.nc
    scc-results/
        <Measurement_ID>/
            generations/
                remote-<creation-hash>/
                    elpp.archive
                    elda.archive
                    cloudmask.archive
                    ...
                    results.json
```

An upload-specific result folder is created atomically after *all* advertised
completed archives are obtained and validated. The remote record must have
finished ELPP and ELDA (SCC result code 127 in the confirmed installation);
missing URLs, corrupt/missing products, unfinished processing, unsafe redirects
or a mismatched configuration fail without publishing an incomplete folder.
The result manifest contains remote creation time, processing outcomes,
configuration ID, linked scientific Session_ID, download links and SHA-256
hashes. Existing results are verified and **never silently overwritten** by a
new remote generation under the same Measurement_ID.

`download` can retrieve an existing SCC portal measurement without having
uploaded it from MILGRAU using `--session <Session_ID>`, provided that session
already has a canonical scientific L0. Its manifest explicitly records
`uploaded_by_milgrau: false`; it **cannot** authorize cache deletion.

The journal now retains `results_path`, `results_manifest_sha256` and new
`downloaded`/`cleaned` states. Old Phase 12.5 SQLite databases are migrated
in place without deleting entries. The cache-cleanup gate requires:

- the durable journal to confirm MILGRAU's own uploaded generation using the
  SCC remote `creation_date` and configuration ID;
- verified completed ELPP and ELDA archives and all other advertised finished
  products, with their SHA-256 values rechecked against `results.json`;
- the results manifest's checksum/path matching the durable journal;
- the original pending raw NetCDF and any LR matching their submitted hashes.

When a local `config.yaml` sets `scc.cleanup.policy: after_download` (not the current example default),
an owned, newly downloaded and verified packet triggers automatic **local-only**
cleanup of `.cache/scc/pending/<Measurement_ID>/`. Use `download --keep-cache`
to suppress that automatic cleanup. To exercise the same gate separately,
use `cleanup --id ID --dry-run` then `cleanup --id ID --confirm`.
`manual` policy also suppresses automatic deletion. Cleanup never touches
Licel files, the scientific NetCDFs, retained SCC outputs, old remote
replacement archives or `.state/scc/submissions.sqlite`. Failures and
uncertain upload states are protected.

A crash between the local directory removal and the SQLite `cleaned`
transition may leave a journal state of `downloaded` with no cache folder.
The command fails closed and requires manual inspection; it never silently
deletes an arbitrary replacement folder.

**Live validation pending:** no real SCC result was downloaded by this new
12.6 command in development and no remote upload/delete occurred. Test first
with a known existing remote measurement using `download --id ... --session`
which is read-only and does **not** clean the pending upload package.

## What happens when the network fails?

If an upload loses its response, the SCC may have received it even when the
client sees a timeout. No automatic retry is performed. The operator must use
`milgrau-scc status --id ... --remote` (or the portal) to reconcile the
outcome before considering another upload. An existing remote ID is never
deleted/replaced.

**Successful upload is recorded as accepted in the durable journal**, but
does not immediately remove staged files. Remote processing completion is
observed through read-only status checks; Phase 12.6 downloads verified
SCC products and removes only eligible disposable staging files.
An uploaded `<Measurement_ID>.nc` packet must not be assumed processed by ELPP/ELDA.

## Test evidence and outstanding validation

Offline mocked-server tests cover CSRF login, HTTP Basic and website
credentials, exactly one multipart POST, LR attachment, remote duplicate
detection, ambiguous POST failures, unsafe redirects, and validation
without credentials. Tests do not substitute for a real SCC server response.

Do not execute `--confirm` against the live SCC until the account,
permissions, upload form behavior, and the station's NetCDF inputs have been
checked together with the operator. Initial remote validation should submit
**one** deliberately chosen measurement, not all archived packets.
