# SCC operator guide — preparing and comparing external lidar processing

The **Single Calculus Chain (SCC)** is an optional external processing system. MILGRAU can prepare SCC-compatible lidar input packages, monitor processing and download completed products for comparison.

**MILGRAU Level 0/1/2 scientific NetCDFs are independent of the SCC.** Processing a session with LIBIDS/LIPANCORA does not upload it to an external server.

## 1. Enable the optional workflow

Install the additional SCC HTTP client only if needed:

~~~bash
python -m pip install -e ".[scc]"
~~~

In `config.yaml`:

~~~yaml
scc:
  enabled: true
  config_file: "scc.yaml"
  state_dir: ".state/scc"
  http:
    timeout_seconds: 60
    get_retries: 2
~~~

The default is `enabled: false`. The separate `scc.yaml` contains SCC server, product and channel identifiers; check it against the actual instrument era and remotely configured SCC products. It contains **no credentials**. Authentication is supplied separately through the supported environment-based mechanism.

The SCC input file identity is not the same as the MILGRAU continuous session identifier. A long session may span solar regimes or months and produce several SCC `Measurement_ID` packages.

## 2. Prepare scientific data locally

Process the selected session as usual:

~~~bash
milgrau-libids -i SESSION_ID
milgrau-lipancora -i SESSION_ID
~~~

If SCC was enabled only after a valid canonical Level 0 already existed, reexport the SCC inputs without rewriting the canonical scientific Level 0:

~~~bash
milgrau-libids -i SESSION_ID --scc-reexport
~~~

Do not combine `--scc-reexport` with `--force`. SCC reexport is an interoperability operation, not a new scientific acquisition or a license to modify the original Level 0.

The local package can contain:

| SCC member | Meaning |
| --- | --- |
| `MEASUREMENT_ID.nc` | SCC-compatible raw lidar measurement |
| `lr_YYYYMMDDcccNN.nc` | Lidar-ratio ancillary input where applicable |
| `rs_YYYYMMDDcccNN.nc` | Observed radiosonde ancillary input, when available and validated |
| `manifest.json` | Local package identity, source and file checksums |

The selected SCC configuration/product identifiers and available instrument channels determine which files can be prepared.

## 3. Soundings: observation only

An SCC sounding is **not** synthesized from ERA5 or USSA76. If observational sounding use is enabled in `scc.yaml`, its preparation is finalized after LIPANCORA verifies the appropriate observation.

The preferred observational sequence is **Wyoming → local NOAA IGRA2**. The SCC ancillary retains the actual observed launch time, station identity and provider. It is linked to the observation used by Level 1 QA, including verified cache identity; a changed/mismatched observation is rejected rather than silently substituted.

When no eligible observed sounding exists, the configured SCC missing-sounding policy applies. A normal MILGRAU Level 1 that used USSA76 is **not** evidence that a measured sounding exists for SCC.

## 4. Inspect and validate local SCC inputs

Run a fully local preflight first:

~~~bash
milgrau-scc upload --dry-run
milgrau-scc upload --id MEASUREMENT_ID --dry-run
milgrau-scc status
~~~

The dry run validates staged packages **without contacting the SCC server**. Local status shows the known staged/journal records; it is not a complete inventory of all remotely hosted SCC measurements.

Local packages remain in the scientific session directory:

~~~text
02-processed_data/spu/YYYY/MM/SESSION_ID/
  SESSION_ID_L0.nc
  SESSION_ID_L1.nc
  SESSION_ID_L2.nc
  scc/
    scc-inputs/
      MEASUREMENT_ID/
        MEASUREMENT_ID.nc
        lr_YYYYMMDDcccNN.nc
        rs_YYYYMMDDcccNN.nc   # if an observed sounding was validated
        manifest.json
    scc-results/
      MEASUREMENT_ID/
        generations/
~~~

Not every package contains all optional ancillaries. A prepared package does **not** prove that SCC has received or processed it.

## 5. Remote status, explicit upload and results

Remote interaction requires authorized credentials and a correctly configured SCC endpoint.

Read-only checks:

~~~bash
milgrau-scc status --id MEASUREMENT_ID --remote
~~~

Upload **one** validated measurement only after confirming the remote ID and authorizing the action:

~~~bash
milgrau-scc upload --id MEASUREMENT_ID --confirm
~~~

A batch upload without `--id` may act on all eligible staged packages; do not use it as a substitute for first checking one ID.

Once the SCC has completed its required processing stages, download available results:

~~~bash
milgrau-scc download --id MEASUREMENT_ID
~~~

Downloaded results and their provenance are stored under the session's `scc/scc-results/` directory. Successful transport/processing status does not automatically establish numerical agreement with MILGRAU.

The SCC includes stages such as ELPP (preprocessing) and ELDA (optical retrieval). A remote record may be present without both stages having completed successfully.

## 6. Protect remote and local records

- An SCC upload is **explicit**, never triggered by ordinary LIBIDS/LIPANCORA/LEBEAR processing.
- Do not replace an existing remote ID simply because a new local file has the same name. Replacement may delete existing inputs/results and cannot be assumed reversible.
- A network interruption during POST or DELETE may leave the outcome uncertain. Inspect the remote state and recorded evidence before any recovery.
- Do not manually remove local journals, historical archives or canonical L0/L1/L2 as if they were disposable caches.
- Keep SCC input manifests and downloaded generations when they support scientific comparisons or operational recovery.

The advanced [SCC protocol and incident reference](scc_native_client.md) documents rare replacement and uncertain-state recovery. Those actions should not be used in a routine scientific workflow.

## 7. Comparing MILGRAU and SCC scientifically

A valid comparison should document:

1. The same raw observation and time interval.
2. The SCC configuration and channel/product mapping.
3. Dark current, background and instrument correction assumptions.
4. Actual atmospheric source, observation date and sounding height coverage.
5. Lidar ratio, molecular reference and extinction assumptions.
6. Wavelength, temporal averaging, vertical resolution and supported altitude.
7. Both products' uncertainty and quality flags.

SCC and MILGRAU do not necessarily implement identical reductions or inversion assumptions. Differences are scientific questions to investigate, not automatically software failures.

## Before reporting successful SCC validation

Confirm that the relevant ELPP/ELDA stages completed for the selected remote ID, downloaded files are present and interpretable, and source/manifest identities match the intended observation. A local dry run or a remote upload acknowledgement alone does not close this validation.

See [Processing levels](processing_levels.md), [Scientific quality](scientific_quality.md) and the [SCC interoperability reference](scc_level0_interoperability.md).
