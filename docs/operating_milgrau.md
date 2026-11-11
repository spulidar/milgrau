# Operating MILGRAU — reproducible user workflows

This guide describes the productive MILGRAU commands and their scientific meaning. It complements the [first-run guide](getting_started.md). The commands have been checked against the current `main` CLI implementation; the example session ID is illustrative, **not evidence of a completed real-data experiment**.

## Set up and identify the dataset

~~~bash
git clone https://github.com/spulidar/milgrau.git
cd milgrau
git switch main
python -m pip install -e .
~~~

Use Python 3.12 or newer. Confirm `config.yaml` points to the correct Licel raw-data tree (normally `01-data`), a writable processed-data tree (normally `02-processed_data`) and a historically correct `station.yaml`. Refer to [Configuration](configuration_ownership.md) for the scientific effect of each setting.

Inspect help before applying a destructive reprocessing flag:

~~~bash
milgrau-libids --help
milgrau-lipancora --help
milgrau-lebear --help
milgrau-inspect --help
~~~

## Data selectors: session, local date or path

For the three science commands, `-i` / `--input` can specify a complete scientific session, a **station-local calendar date** as `YYYYMMDD`, or an existing path. Without `-i`, the respective stage discovers eligible inputs.

| Input | Example | Meaning |
| --- | --- | --- |
| Entire discovered input set | `milgrau-libids` | Find available raw acquisitions |
| Complete session | `-i spu_20250511-0012Z_20250511-0737Z` | Process that continuous UTC session |
| Local civil date | `-i 20250510` | Sessions intersecting that station-local calendar date |
| Existing path | `-i /data/lidar/2025/05` | Choose available raw or processed files relevant to the stage |

A local date is **not** a UTC date or a 00–06/06–12/12–18/18–24 publication partition. Selection by exact session ID or local date must match sessions in the discovered inventory. Path selectors must exist.

A session is defined by acquisition continuity and the applicable historical instrument configuration; its `spu_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ` ID uses UTC minute precision. The underlying file retains more precise acquisition times.

## Standard Level 0 → Level 1 → Level 2 operation

~~~bash
milgrau-libids
milgrau-lipancora
milgrau-lebear
~~~

To process one known session:

~~~bash
milgrau-libids -i spu_20250511-0012Z_20250511-0737Z
milgrau-lipancora -i spu_20250511-0012Z_20250511-0737Z
milgrau-lebear -i spu_20250511-0012Z_20250511-0737Z
~~~

- LIBIDS standardizes the original Licel acquisitions and writes the canonical `_L0.nc`.
- LIPANCORA calculates corrected/RCS signals and materializes the hourly molecular state in `_L1.nc`.
- LEBEAR calculates conditional elastic aerosol optical products with uncertainty and validity information in `_L2.nc`.

A successful execution status is not a scientific quality certificate. Inspect acquisition QA, actual atmosphere, optical reference, valid altitude and missing data.

## Choosing an atmosphere

~~~bash
milgrau-lipancora -i SESSION_ID
milgrau-lipancora -i SESSION_ID --cds-queue-threshold 8000
milgrau-lipancora -i SESSION_ID --no-cds-queue-check
milgrau-lipancora -i SESSION_ID --radiosonde-only
~~~

The ordinary preference is **ERA5 → Wyoming radiosonde → NOAA IGRA2 → USSA76**. An already validated ERA5 cache hour remains eligible when the public CDS queue is congested; the queue indicator is a heuristic, not a guaranteed waiting time.

The explicit `--radiosonde-only` flag avoids ERA5, then tries Wyoming, the locally available IGRA2 archive and an identified USSA76 last resort. It cannot be combined with `--figures-only`.

The same canonical L1 filename is used for different atmosphere selections. **Preserve prior files before comparing alternate scientific recipes.** Rerun L2 after an intentional change of L1.

## Selecting Level 2 times and segments

~~~bash
milgrau-lebear -i SESSION_ID --regime night
milgrau-lebear -i SESSION_ID --segment seg01
milgrau-lebear -i SESSION_ID --regime night --time-window-utc 01:00 03:00
~~~

`--regime` and `--segment` are mutually exclusive. `--regime` may select disjoint pieces of a session; `--segment` selects one contiguous solar segment. `--time-window-utc` accepts two UTC times in HH:MM or ISO format and can further restrict either selection.

These are **derived analysis selections**, not new raw-acquisition sessions.

## Incremental processing, --force and figure-only

`processing.incremental: true` allows existing valid/current products to be reused. It is not a claim that instrument calibration or the physical model has been independently revalidated.

~~~bash
milgrau-libids -i SESSION_ID --force
milgrau-lipancora -i SESSION_ID --force
milgrau-lebear -i SESSION_ID --force
~~~

Use `--force` intentionally, after backing up any scientific baseline to a different location. It can replace a product with the same canonical name.

Only refresh LIPANCORA figures:

~~~bash
milgrau-lipancora --figures-only -i SESSION_ID
milgrau-lipancora --figures-only -i SESSION_ID --force
~~~

In figure-only mode, `--force` regenerates figures; it does **not** recalculate the Level 1 science NetCDF.

## Files, figures and inspection

The canonical organization is based on the **UTC start year and month of the continuous session**:

~~~text
02-processed_data/
  spu/
    YYYY/
      MM/
        SESSION_ID/
          SESSION_ID_L0.nc
          SESSION_ID_L1.nc
          SESSION_ID_L2.nc
          figures/
          scc/                  # only when applicable
~~~

Use `milgrau-inspect SESSION_ID` for a text summary and `milgrau-explorer` for read-only figures and fields. Figure time/height aggregation affects display only; it does not alter scientific NetCDF resolution.

The `figures/` layout and derived L2 filenames can vary by stage, wavelength and time/regime selection. Do not treat the diagram above as a literal listing of every generated file.

## Optional SCC reexport and remote operation

`scc.enabled: false` is the default. After deliberately enabling and validating `scc.yaml`, SCC-only restaging can be performed without rebuilding canonical L0:

~~~bash
milgrau-libids -i SESSION_ID --scc-reexport
milgrau-scc upload --id MEASUREMENT_ID --dry-run
~~~

`--scc-reexport` and `--force` cannot be combined. A dry run checks the staged package locally; it does not upload. A separate explicit command/confirmation is required before actual remote submission. Review [SCC operations](scc_batch_operator_guide.md).

## Suggested reproducibility record

Retain the exact source Licel files, session ID and UTC range; station profile/calibration; full relevant YAML recipe; source-specific atmospheric identities; file hashes; quality warnings; reference and lidar-ratio assumptions; selected wavelengths/segments; and the final valid altitude/uncertainty support.

The purpose is not merely to reproduce a file, but to reproduce the **physical assumptions and observations** from which its reported aerosol properties were derived.

See [Level 0](libids.md), [L0 variables](level0_schema.md), [Atmospheric profiles](atmospheric_profiles.md) and [Scientific quality](scientific_quality.md).
