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
and controls `scc.http.timeout_seconds` and `scc.http.get_retries`.
The latter retries **only read-only HTTP GETs**, on transient errors:
it never repeats a Quick Upload POST, authentication POST or DELETE.
The previous `scc.upload.automatic`, `scc.upload.timeout_seconds` and
`scc.upload.retries` configuration keys have been removed.

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

## Three primary operator commands

```bash
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
# give the verified scientific session ID explicitly; inputs remain untouched.
milgrau-scc download --id 20240913spu1549 --session spu_20240913-1549Z_20240913-1750Z

# A confirmed-owned download also preserves the inputs:
milgrau-scc download --id 20240913spu1549
```

The initial Phase 12.4 only implemented single-ID upload. In Phase 12.7,
the explicit `upload --confirm` command now supports a **sequential batch**
of all pending local packages. Local integrity preflight covers the whole
batch before any network work. By default, pre-existing remote IDs are
**skipped without deletion or journal ownership claims**. Only
`upload --confirm --replace-existing` authorizes a batch to replace
remote-existing IDs without requiring original-data backups when `--confirm` is passed. A failed,
interrupted or uncertain operation stops subsequent IDs; partial successes
remain recorded in the durable journal, and no network POST is retried
automatically. A final summary reports submitted, replaced, identical,
skipped, failed and unprocessed counts. Single `--id` operations remain
available and are the required first live replacement test.

For a current operator workflow and exact runbook, read
[batch submission and first replacement test](scc_batch_operator_guide.md).
There is no automatic SCC submission flag. LIBIDS and LIPANCORA never
initiate SCC HTTP uploads. Remote POST/DELETE operations require
`milgrau-scc upload --confirm` and remain fail-stop on uncertainty.
`--continue-on-error` has been removed: a remote/journal error halts
the batch before later IDs are processed.

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
**optional operator-requested backup → delete old SCC record → upload new bytes
with the same ID**. SCC does not offer transactional overwrite, so this flow is
intrinsically non-atomic.

A private persistent SQLite journal lives in
`.state/scc/submissions.sqlite`, explicitly **outside** processed session products and
excluded by Git. It records scientific session, channel configuration,
product IDs, source segments and periods, hashes of raw/LR NetCDF,
transitions and timestamps, remote identity, archive location, and
the last error. An exclusive per-ID lock prevents concurrent local uploads.
A stale lock after a crash **must be investigated before manual removal**.

A safe, optional read-only backup can be requested **before** replacement.
It authenticates and downloads/verifies the remote raw and available products
into persistent state, but never modifies the server:

```bash
milgrau-scc status --id 20240913spu1549 --remote --backup
```

Remote backup is optional when the operator uses `--replace-existing --confirm`.
Skipping it may permanently destroy prior SCC inputs/results; the backup
command remains a separate, repeatable read-only option.

A replacement is never automatic. It requires precisely:

```bash
milgrau-scc upload --id 20240913spu1549 --replace-existing --confirm
```

**This command explicitly waives the backup requirement.** It is a
destructive replacement operation; `--confirm` authorizes deletion of the
existing SCC record after all safeguards. No other SCC records or related
measurements are intentionally deleted.

The replacement workflow:
1. Validate the complete pending package and its SHA-256 checksums.
2. Query SCC and reject queued/active processing or unexpected changes.
   An absent ID is not treated as a replacement.
3. With `--replace-existing --confirm`, record the explicit no-backup waiver,
   check the remote fingerprint and local hashes **before** beginning a new
   journal revision. When an archive is requested separately, download and
   verify the old raw NetCDF/ZIP, available remote products and external LR
   into `.state/scc/archives/<Measurement_ID>/<timestamp>/`. If the archive
   cannot be verified, the archived flow refuses deletion. An identical
   verified archive is a no-op.
4. Issue the guarded administrative DELETE with `not_delete_related` and
   confirm remote absence. POST the new packet with the **same ID** exactly
   once, then require authoritative API confirmation.
5. Record each phase in SQLite. An unknown DELETE/POST outcome remains
   `uncertain`; ordinary uploads never retry it automatically.
6. An operator can recover an *uncertain DELETE* using `upload --id ID
   --resume-after-delete --confirm` only after checking remote absence,
   immutable staged identity and either a verified archive or the explicit,
   same-revision no-backup waiver. This submits at most one POST, never DELETE.
   An uncertain POST requires separate explicit reconciliation.

A pre-existing SCC record is **not** considered to have been uploaded by
MILGRAU merely because remote `status` shows ELPP/ELDA results.
`milgrau-scc status` shows the local queue and durable journal;
`status --id ID --remote` shows safe remote metadata and local state
without exposing the full station object or credentials.

**Limitations requiring live verification:** the official SCC version may
return raw files or ancillary LR downloads in a different container/schema.
If an archive was requested and cannot be verified locally, its guarded
workflow stops before deletion; an explicit no-backup replacement may instead
accept the documented irreversible risk. Upload and admin-delete endpoints are
mock-tested, but have not been exercised on the live SCC in this phase.
The SQLite journal cannot prevent independent SCC portal operators from
editing the measurement during the narrow interval after its final read;
there is no server-side atomic compare-and-swap. Use a restricted operator
maintenance window for destructive replacements.

Phase 12.6 implements verified read-only downloads. Session-local SCC
inputs, old remote backups and journal entries are retained independently;
there is no automatic or CLI-mediated SCC input cleanup.

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

Session-local SCC inputs are persistent. Downloads never remove them.
The former `scc.cleanup.policy`, `download --keep-cache` and
`milgrau-scc cleanup` interfaces no longer exist. Remove legacy
`scc.cleanup` settings from local `config.yaml` before running commands.

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

## Current SCC sounding preparation

For newly generated inputs, LIBIDS stages `sounding_stage: awaiting_lipancora`
only when observed soundings are enabled. LIPANCORA finalizes those inputs
automatically, using independently verified local radiosonde observations
and the corresponding session's scientific provenance. The
`missing_policy: skip` and `fail` options remain supported. Packages
already recorded in the SCC journal cannot be altered by finalization.

The historical `milgrau-scc sounding` and `sounding-repair` commands,
`upload --migrate-cache`, `upload --repair-lr-cache`, and the
incident-specific `--retry-after-upload`/`--retry-after-lr-fix` options
were removed. They must not be used on existing jobs. `upload --dry-run`
validates current `<Measurement_ID>.nc`, `lr_YYYYMMDDcccNN.nc` and optional
`rs_YYYYMMDDcccNN.nc` names. Recreate incompatible, unsubmitted
historical inputs through explicit session SCC re-export after inspecting
the journal and original science; do not rewrite submitted/uncertain inputs
or silently resume historical network POSTs.

## Phase 12.6 — Verified SCC results and persistent inputs

SCC results are explicitly **external scientific outputs**, not replacements
for MILGRAU L0/L1/L2 or the station's own inversion. Download saves the
read-only SCC processed result archives under the original scientific session:

```text
02-processed_data/spu/YYYY/MM/<Session_ID>/
    <Session_ID>_L0.nc
    scc/
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

The journal retains `results_path`, `results_manifest_sha256` and the
historical `downloaded`/`cleaned` states. Existing SQLite journals
continue to be readable; the historical `cleaned` state is **not**
an instruction to delete anything. Download verification validates
the result archives and their checksums, independently of all local input
directory lifecycle decisions.

There is no cleanup subcommand, automatic deletion on download or
configuration policy for deleting inputs. All staged input folders are
preserved by the upload/status/download workflow. For operator-managed
disk cleanup, inspect the exact Measurement_ID, remote state, pending
journal operations, file hashes and future re-export needs. Only then may
the operator independently remove an exact
`<Session_ID>/scc/scc-inputs/<Measurement_ID>/` folder; retain
`.state/scc/`, `scc-results/`, raw Licel and all scientific files.

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
SCC products without deleting session-local SCC inputs.
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
