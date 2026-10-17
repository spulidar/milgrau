# SCC batch operator guide — same-ID replacement (Phase 12.7)

**Scope:** launch all locally staged SCC packets sequentially, optionally replace
remote-existing measurements without changing their Measurement_ID, download
products, and **preserve all staging cache**. This is an experimental Phase 12.7
workflow until native multipart upload and delete/re-upload have been validated
on the live SCC. The read-only status, backup and six-product download for
`20240913spu1549` were validated with a real SPU login.

## Safety first

- The default `upload --confirm` without `--id` processes all pending
  measurements **sequentially**. It **skips remote-existing** IDs: it does not
  replace, delete, download or reprocess them.
- Only `upload --confirm --replace-existing` opts in to *batch* deletion and
  replacement of existing SCC IDs. It requires a locally verified backup
  of the existing remote measurement and every retrievable prior output,
  including original external LR if referenced. It preserves the remote
  Measurement_ID but removes/reprocesses the previous SCC record.
- A whole-batch **offline preflight** of the staged NetCDF packages precedes
  all network calls. Any corruption aborts before upload. During a live batch,
  errors/ambiguous outcomes stop subsequent IDs; there are NO automatic
  POST/DELETE retries and NO automatic continuation of `uncertain` records.
  Previously successful IDs are not rolled back.
- Upload does not clean cache. Set `scc.cleanup.policy: manual` in
  `config.yaml` AND pass `download --keep-cache`. Never call
  `cleanup --confirm` in this workflow.
- Live **same-ID** deletion/re-upload has not yet been verified end-to-end on
  real SCC. It is **not atomic**; if deletion succeeds and upload fails, the
  original SCC record can disappear until recovered. Only test first against
  one expressly approved measurement whose SCC history may be replaced,
  with independently preserved backup and a known restoration plan.
- `scc.upload.automatic: false` remains false. Neither LIBIDS nor the MILGRAU
  scientific L0/L1/L2 processing launches SCC uploads automatically.

## Prepare (Arch Linux, zsh)

Inside the existing `scc-integration-refactor` checkout:

```bash
cd ~/milgrau
git status --short
git branch --show-current
# Commit/stash local config changes as appropriate BEFORE git pull.
git pull --ff-only
source .venv/bin/activate
python -m pip install -e ".[scc]"
```

Ensure `scc.enabled: true`, `scc.state_dir: ".state/scc"` and
`scc.cleanup.policy: "manual"` in your **local** `config.yaml`.
Point `scc.yaml server.base_url` to the tested HTTPS SCC origin.

Export the two independent credential pairs if the deployment requires both;
**never paste secrets into tickets, output logs or GitHub**:

```zsh
read -r "MILGRAU_SCC_USERNAME?SCC website user: "
read -rs "MILGRAU_SCC_PASSWORD?SCC website password: "
echo
read -r "MILGRAU_SCC_BASIC_USERNAME?SCC HTTP Basic user: "
read -rs "MILGRAU_SCC_BASIC_PASSWORD?SCC HTTP Basic password: "
echo
export MILGRAU_SCC_USERNAME MILGRAU_SCC_PASSWORD
export MILGRAU_SCC_BASIC_USERNAME MILGRAU_SCC_BASIC_PASSWORD
```

Omit the Basic pair completely if your SCC deployment doesn't require it.

Validate and inventory without modifying SCC:

```bash
milgrau-scc upload --dry-run
milgrau-scc status
milgrau-scc status --id 20240913spu1549 --remote
milgrau-scc status --id 20240913spu1549 --remote --backup
```

## First controlled live **single-ID** replacement

For the example ID `20240913spu1549` we already verified a real SCC
read-only archive and downloaded six processor-result groups.
**It is not a disposable ID**. Only use it when you have explicit authority
to replace its existing SCC results and know how to restore the archived
original if a POST fails. Prefer a disposable test measurement in the same
local pending inventory instead.

1. Confirm the local package validates and its `raw.nc` references the
   intended LR file with `milgrau-scc upload --id ID --dry-run`. The output
   must show the correct day/night configuration and LR auxiliary.
2. Confirm the remote is not queued or processing:
   `milgrau-scc status --id ID --remote`.
3. Make a **new read-only backup** and confirm it says `VERIFIED`:
   `milgrau-scc status --id ID --remote --backup`.
   Keep the archive outside `.cache` and optionally copy it to an
   independently backed-up location. Review the previous raw/outputs, not
   only the archive-manifest SHA-256.
4. Only after accepting the real non-atomic data-loss risk, submit **one ID**
   under the same Measurement_ID:
   `milgrau-scc upload --id ID --replace-existing --confirm`.
5. Inspect the CLI output and durable journal; on timeout/uncertain/failed,
   **STOP**. Do not rerun `--confirm` blindly or attempt batch. Use
   `milgrau-scc status --id ID --remote` and the preserved archive to
   coordinate manual recovery with the SCC operator.
6. When accepted, poll read-only until ELPP and ELDA both equal 127 and
   queued/running are False:
   `milgrau-scc status --id ID --remote`.
7. Download without allowing local staging removal:
   `milgrau-scc download --id ID --keep-cache`.
   Confirm `SCC downloads VERIFIED` and `upload_owned=True`, and
   check the new directory under `scc-results/ID/generations/`. Earlier
   downloaded generations remain separate.
8. Check `milgrau-scc status --id ID` and ensure
   `.cache/scc/pending/ID/raw.nc` and the original LR auxiliary are present.
   Confirm the new remote result and LR are scientifically correct.

## Incident recovery: DELETE returned unknown, SCC ID now absent

**Observed on the real SCC (2026-10-17):** Single-ID replacement of
`20240913spu1549` returned `Deletion POST ... UNKNOWN outcome`. The
subsequent authenticated `status --remote` reported `NOT FOUND`.
The local journal retained `state=uncertain` and a verified original
archive at
`.state/scc/archives/20240913spu1549/20261017T212943822169Z`.
All 19 staged packets continued passing the offline dry run.
**The original SCC record appears to have been deleted, but the new
raw/LR was never sent by the interrupted replacement workflow.**
The precise network/transport root cause is not yet established.

Do **not** repeat `--replace-existing --confirm`, do **not** submit the
19-packet batch, and do not manually edit the SQLite journal. Backup of a
now-absent remote measurement naturally fails; that is not a failure of
the already stored local archive. First verify the original archive locally
(and independently retain a copy):

```bash
python - <<'PY'
from pathlib import Path
from milgrau.config.loader import load_config
from milgrau.scc.backup import verify_archive
from milgrau.scc.state import SubmissionStore, state_root

with SubmissionStore(state_root(load_config())) as store:
    record = store.get("20240913spu1549")
    assert record and record["state"] == "uncertain", record
    saved = verify_archive(
        Path(record["archive_path"]),
        record["archive_manifest_sha256"],
    )
    print("ARCHIVE VERIFIED:", record["archive_path"])
    print("Original files:", sorted(saved["files"]))
    print("History:", [(event["action"], event["state"])
                       for event in store.events("20240913spu1549")[-5:]])
PY
```

Then use `milgrau-scc status --id 20240913spu1549 --remote` again
(read-only) to confirm the remote ID is **still absent**. The SCC
operator should verify the deleted record is not merely transiently
invisible or being restored through the portal.

Only if you decide to **finish the intended replacement** (rather
than restore the archived original), the exact-ID, explicit recovery
operation is:

```bash
milgrau-scc upload --id 20240913spu1549 --resume-after-delete --confirm
```

This command works **only** for an `uncertain` journal whose most recent
event is `delete_result_uncertain`, with matching immutable local
raw/LR checksums, verified old remote archive and a fresh SCC 404.
It never deletes anything. It requests **one** new raw/LR upload to
the currently absent ID, then requires API confirmation. It cannot
run in a batch, with `--replace-existing`, or without both `--id`
and `--confirm`. A failed/unknown recovery upload records
`recovery_upload_uncertain` and **cannot be automatically retried**;
inspect the portal and coordinate with the SCC operator.

After an accepted response, wait for ELPP/ELDA=127 and run
`milgrau-scc download --id 20240913spu1549 --keep-cache`.
Verify `upload_owned=True` and climatological LR acceptance on SCC.
The original remote archives and previous six-product results remain
locally preserved.

**Do not infer the delete POST's underlying exception type from the
initial message:** new code surfaces only its transport exception class
for *future* failed deletion attempts; it cannot retroactively diagnose
this already completed request.

## Follow-up: recovery upload returned HTTP 200 but SCC API did not confirm

**Observed 2026-10-17:** The operator authorized a single
`upload --id 20240913spu1549 --resume-after-delete --confirm`, and
the resulting live multipart upload returned
`Upload response ... was not confirmed by the SCC API; remote state is
uncertain. Do not repeat the POST.` This indicates the client executed
the upload request, but **does not confirm SCC accepted raw.nc or the
LR auxiliary**. The journal now blocks another automatic recovery
(`recovery_upload_uncertain`).

Check **read-only** `milgrau-scc status --id 20240913spu1549 --remote`
and the actual SCC portal, without any new upload or delete. In the
event the API still reports 404, also inspect the server's ancillary
LR registry with a read-only API GET for
`lr_20240913spu1549.nc`; the authoritative `scc_access` client
looks up already registered ancillary files before attaching them,
whereas the current MILGRAU client always includes the locally staged
LR. A duplicate LR record or SCC NetCDF validation rejection could
explain the failed submission, but has **not** been proven here. Avoid
silently using an already-registered LR based on filename alone.

The official `scc_access` source distinguishes a submission that
remains on the Quick upload form (potential validation rejection)
from a redirect to measurement details and documents that the API
may show temporary 404 immediately after acceptance. MILGRAU now
does up to three **read-only** API lookups after a single POST and
separately identifies a Quick-form response. This only improves
future diagnostics; the HTML form response from this already
completed attempt is not recorded and cannot be recreated
without another upload.

**Never reset the journal, delete the archive, or run either batch**
until the previous attempt is resolved. If the remote ID is present,
verify the new generation, upload timestamp, configuration and
external LR before formally reconciling journal ownership. If absent,
investigate SCC upload form validation and server logs in cooperation
with its administrators before considering another manually
authorized upload.

## Read-only finding: uploaded raw NetCDF name was wrong

The official SCC input format requires the **uploaded filename**
`<Measurement_ID>.nc`, while the original MILGRAU native client
submitted the disposable internal filename `raw.nc`. The native
client has now been fixed to send the same staged NetCDF bytes using
`20240913spu1549.nc` as the multipart filename, without renaming or
changing any cached bytes. The previous POST response HTML was not
retained; this may explain server-side form rejection, but cannot prove
the historic failure. SCC `LR_Input=0` for supplied external LR was
also checked against the exporter policy.

**Do not repeat the prior `--resume-after-delete` command:** the
journal now has `recovery_upload_uncertain`, and the command is
intentionally limited to a different prior event. An eventual second
POST must be designed as a separately authorized **single-ID**
reconciliation of the failed upload, contingent on confirming remote
absence through API **and portal**, original archive integrity, and
the identity/configuration/source checks. No automatic retry, reset of
`.state/scc/submissions.sqlite`, additional deletion or 19-ID batch
is allowed.

## October 17 controlled retry — switch ALL local staged raw files to ID.nc

The operator confirmed in the SCC **website as well as the API**
that `20240913spu1549` remains absent after a
`recovery_upload_uncertain` event. The original network upload
returned HTTP 200 but was not indexed by SCC. The client previously
transmitted the correct multipart *name* without storing the same
name locally. The new rule is **consistent on disk and in transit**:
`.cache/scc/pending/<Measurement_ID>/<Measurement_ID>.nc`, alongside
`lr_<Measurement_ID>.nc` and `manifest.json`. Existing scientific
sessions and old remote backup archives are unchanged.

**Required local migration before any further POST:**

```bash
milgrau-scc upload --migrate-cache
milgrau-scc upload --dry-run
```

Migration is fully offline and operates on all existing pending
packages; it first verifies the old raw/LR NetCDF and manifest hashes,
copies the raw bytes to the ID-named file, atomically updates the
existing manifest's file mapping and only then removes the old
`raw.nc` duplicate. It leaves original NetCDF bytes, LR inputs,
scientific L0/L1/L2, old remote archives and the durable
`.state/scc` SQLite journal untouched. The command can be
repeated safely if interrupted; conflicts or tampering fail closed.
A single package can also be migrated using
`milgrau-scc upload --id ID --migrate-cache`.
**The cache is retained** under the same `pending/ID` directory.

After `19/19` local preflight succeeds and the operator confirms
the remote ID is *still* absent from both portal and API, a **single**
explicit upload-only retry is available for precisely the old
`recovery_upload_uncertain` event:

```bash
milgrau-scc upload --id 20240913spu1549 --retry-after-upload --confirm
```

Unlike the prior `--resume-after-delete` and
`--replace-existing` commands, `--retry-after-upload` cannot
delete anything. It performs a fresh API 404, original backup archive
verification, per-file SHA-256, scientific period, configuration, LR
and journal history check **before requesting one upload POST**.
A successful new SCC record must be confirmed by the authoritative
API to register `accepted`. If the operation fails or becomes
uncertain, the last journal action changes to
`manual_corrected_upload_uncertain` and the command refuses
to run again. No automatic retries, batch mode or further deletion
are permitted. If the remote measurement reappears, the command
stops before uploading.

Continue only after a successful single-ID trial with
`milgrau-scc status --id 20240913spu1549 --remote`,
wait for ELPP=127 and ELDA=127, verify SCC's accepted LR
and `milgrau-scc download --id 20240913spu1549 --keep-cache`.
Do not run the 19-ID bulk upload until this trial proves the actual
SCC server accepts the raw and LR inputs.

## Actual root cause found in the SCC Quick Upload form (2026-10-17)

An **operator-run manual upload** reproduced the SCC validation message:

> Submit a lidar ratio file with filename format lr_YYYYMMDDcccNN.nc

The earlier MILGRAU artifact `lr_20240913spu1549.nc` includes HHMM
(four digits) rather than SCC's requested two-digit `NN` suffix.
That is a confirmed **upload-form validation failure for this deployment**;
other NetCDF validation failures cannot be ruled out until the LR name
passes. The generic SCC input-document examples use `lr_<Measurement_ID>.nc`,
so the stricter two-digit filename rule is **deployment-specific** and
comes directly from the form being used.

MILGRAU now assigns the two-digit `NN` from the measurement's UTC
starting hour (stable for a given SCC Measurement_ID). For example:

```text
.cache/scc/pending/20240913spu1549/
    20240913spu1549.nc
    lr_20240913spu15.nc
    manifest.json
```

The name must also match the **`LR_File_Name` global attribute**
inside the raw NetCDF, so the raw NetCDF's SHA-256 changes even though
the lidar signal, temporal bins and LR numeric content do not. Another
measurement starting in the same hour on the same day at the same
station is rejected at staging rather than reusing an auxiliary filename.

**Required offline repair** (no SCC GET, POST or DELETE):

```bash
milgrau-scc upload --repair-lr-cache
milgrau-scc upload --dry-run
```

The repair validates old hashes, copies the original package to
`.cache/scc/legacy-lr/<Measurement_ID>/`, updates only
`LR_File_Name` inside a separate raw NetCDF, renames the LR file
without modifying its bytes, and publishes the repaired manifest.
For the old journal in `uncertain`, it keeps the original SCC backup,
failure state and all events, recording a new
`local_lr_filename_repair` event containing before/after raw
checksums. **Neither archive nor original cache evidence is deleted.**
This is an explicit file revision, unlike the earlier raw *basename-only*
migration whose hashes stayed unchanged.

After 19/19 offline validations succeed, confirm the ID is absent
on SCC's **website and API** and verify whether
`lr_20240913spu15.nc` is already registered as an ancillary on the
site. Do **not** overwrite or duplicate an already-OK ancillary.
The new **one-time upload-only** retry can then be explicitly
authorized for the single affected ID, never for a batch:

```bash
milgrau-scc upload --id 20240913spu1549 --retry-after-lr-fix --confirm
```

It requires the exact `local_lr_filename_repair` journal event,
matching fixed hashes/period/configuration and old remote archive,
then reconfirms the remote ID is missing. It does not send DELETE.
Any failure or ambiguous result becomes
`manual_lr_fixed_upload_uncertain`, blocking another retry. If SCC
accepts the corrected files, inspect ELPP/ELDA processing and confirm
that the external climatological LR is used. Download only when
completed, with `--keep-cache`. The previous official SCC results
generation and original remote backup remain retained locally.

Do not start the bulk 19-ID upload before this single-ID test
is verified end to end.

## Only after the single-ID trial passes

```bash
# Stage validation (no network).
milgrau-scc upload --dry-run

# Conservative batch: upload absent IDs, skip existing SCC records.
milgrau-scc upload --confirm

# Destructive batch, ONLY after single-ID replacement passes,
# every existing SCC record is eligible and operator-approved,
# and restoration is understood:
milgrau-scc upload --confirm --replace-existing
```

The batch summary contains `total`, `submitted`, `replaced`,
`identical`, `skipped_existing`, `failed`, and `unprocessed`.
On the first error, no further IDs are touched, even if some earlier
submissions completed. **Do not blindly rerun a failed destructive batch.**

After SCC finishes a journal-confirmed upload, use per-ID read-only status,
then `milgrau-scc download --id ID --keep-cache`. Phase 12.7 does **not**
implement an automatic download-all command; repeat the download command per
ID, and do not download unfinished ELPP/ELDA products.

All SCC-generated results remain external artifacts separate from canonical
MILGRAU NetCDFs. The original Licel input, staged raw + climatological LR,
SCC archives and durable journal are all retained.

## Operator checks after every trial

```bash
milgrau-scc status
milgrau-scc upload --dry-run
ls -la .cache/scc/pending/
ls -la .state/scc/archives/
```

**Important:** repeated `upload --confirm --replace-existing` can reprocess
previously uploaded IDs. Use the batch replacement operation only when you
intend to replace *all* remote-existing pending measurements, not as an
idempotent routine upload. When unsure, use the non-replacing batch first.
