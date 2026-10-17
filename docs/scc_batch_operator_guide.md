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
