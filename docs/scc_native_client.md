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

## What happens when the network fails?

If an upload loses its response, the SCC may have received it even when the
client sees a timeout. No automatic retry is performed. The operator must use
`milgrau-scc status --id ... --remote` (or the portal) to reconcile the
outcome before considering another upload. An existing remote ID is never
deleted/replaced.

**Successful upload does not remove or mark a pending package as completed.**
Its durable lifecycle, idempotent reconciliation and status registry belong
to Phase 12.5. Download and cleanup remain blocked until Phase 12.6.
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
