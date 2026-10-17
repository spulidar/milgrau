# SCC native client — Phase 12.4

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
# Validate every staged packet, offline; no SCC credentials or network needed.
milgrau-scc upload --dry-run

# Validate one specific SCC Measurement_ID locally.
milgrau-scc upload --id 20240913spu1549 --dry-run

# Confirm a single real upload. Requires environment credentials.
milgrau-scc upload --id 20240913spu1549 --confirm

# Local pending files and integrity.
milgrau-scc status

# Read-only remote state for one ID (requires credentials).
milgrau-scc status --id 20240913spu1549 --remote

# Reserved and safe: explicitly refuse to run until Phases 12.5–12.6.
milgrau-scc download
milgrau-scc cleanup
```

There is deliberately no batch live upload, remote delete, force-replace or
automatic upload in Phase 12.4. The explicit `--id ... --confirm` command
is necessary to authorize one real network submission. Setting
`scc.upload.automatic: true` **does not** cause LIBIDS to perform a POST.

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

`download` and `cleanup` remain explicitly **disabled** until Phase 12.6.
Old SCC backups and journal entries are persistent evidence and must not
be removed by cache cleanup.

## What happens when the network fails?

If an upload loses its response, the SCC may have received it even when the
client sees a timeout. No automatic retry is performed. The operator must use
`milgrau-scc status --id ... --remote` (or the portal) to reconcile the
outcome before considering another upload. An existing remote ID is never
deleted/replaced.

**Successful upload is recorded as accepted in the durable journal**, but
still does not remove staged files. Remote processing completion is observed
through read-only status checks; download and cleanup are Phase 12.6.
An uploaded `raw.nc` packet must not be assumed processed by ELPP/ELDA.

## Test evidence and outstanding validation

Offline mocked-server tests cover CSRF login, HTTP Basic and website
credentials, exactly one multipart POST, LR attachment, remote duplicate
detection, ambiguous POST failures, unsafe redirects, and validation
without credentials. Tests do not substitute for a real SCC server response.

Do not execute `--confirm` against the live SCC until the account,
permissions, upload form behavior, and the station's NetCDF inputs have been
checked together with the operator. Initial remote validation should submit
**one** deliberately chosen measurement, not all archived packets.
