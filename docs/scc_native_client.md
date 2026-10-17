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
# Upgrade older local pending caches offline, no server requests.
milgrau-scc upload --migrate-cache

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

When `config.yaml` sets `scc.cleanup.policy: after_download` (current default),
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
