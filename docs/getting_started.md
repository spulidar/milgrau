# Getting started with MILGRAU

This guide is for researchers processing existing Licel lidar observations, not software developers.

## 1. Prepare the environment

Use Python 3.12 or newer. In a working folder:

~~~bash
git clone https://github.com/spulidar/milgrau.git
cd milgrau
git checkout integration/scc-atmosphere-20261111
python -m pip install -e .
~~~

The integrated Explorer is included. Only if you intend to interact with the remote Single Calculus Chain, install the optional SCC requirements:

~~~bash
python -m pip install -e ".[scc]"
~~~

For command-specific options, use `milgrau-libids --help`, `milgrau-lipancora --help`, `milgrau-lebear --help` or `milgrau-scc --help`.

## 2. Supply the measurements and the scientific recipe

The default `config.yaml` reads:

- `directories.raw_data`: the original Licel observations (default `01-data`).
- `directories.processed_data`: scientific outputs (default `02-processed_data`).
- `station_config`: the station catalog (default `station.yaml`).
- `processing.incremental`: reuse existing valid products where applicable.
- `level0`, `level1` and `level2`: stage-specific scientific and processing parameters.

`station.yaml` describes station identity, instrument history, dated instrument configurations, channel calibration and physical measurement context. Verify that it describes the actual period being processed; defaults are not a substitute for traceable instrument measurements. A historical instrument configuration change may start a new session.

The optional `scc.yaml` provides SCC channel/product identifiers. It is read only when `scc.enabled: true` is explicitly selected in `config.yaml`. The default remains **false**.

CDS credentials, if ERA5 network downloads are needed, must be configured through the supported external `cdsapi` mechanism (usually `~/.cdsapirc`), not inserted in the repository YAML or NetCDF metadata.

## 3. Choose an observation

MILGRAU organizes raw acquisitions into **continuous scientific sessions**. A canonical ID looks like:

`spu_20250511-0012Z_20250511-0737Z`

The ID records session start/end in UTC (minute precision). A session may span midnight or one or more solar regimes. Distinct solar segments remain recorded inside the session.

CLI `-i` selectors accept a session ID, station-local date in `YYYYMMDD` form, or an explicit file/directory path, depending on the processing stage. A date selects intersecting sessions, not a fixed publication window.

## 4. Process Level 0, Level 1 and Level 2

To process available input data:

~~~bash
milgrau-libids
milgrau-lipancora
milgrau-lebear
~~~

To process one already identified session:

~~~bash
milgrau-libids -i spu_20250511-0012Z_20250511-0737Z
milgrau-lipancora -i spu_20250511-0012Z_20250511-0737Z
milgrau-lebear -i spu_20250511-0012Z_20250511-0737Z
~~~

LIBIDS produces the standardized acquisition `_L0.nc`; LIPANCORA generates corrected signals and time-resolved atmosphere in `_L1.nc`; LEBEAR generates `_L2.nc` with optical retrieval and validity information. Stage figures are created alongside their scientific NetCDF outputs.

### Atmosphere when ERA5 is congested

In ordinary processing, the configured order is **ERA5 → Wyoming → IGRA2 → USSA76**. Valid cached ERA5 can still be used during a high CDS queue; new CDS requests are avoided in the high-queue mode. The public queue number is a heuristic indicator, not a reliable wait-time estimate.

For an explicit run **without ERA5/CDS**, use:

~~~bash
milgrau-lipancora -i SESSION_ID --radiosonde-only
~~~

This attempts Wyoming, then a locally available NOAA IGRA2 historical archive, then explicitly identified USSA76 when no acceptable observation exists. The configured radio-only search window is 12 hours; observing times and source identities are stored in Level 1.

The default queue threshold is 8,000 globally queued requests and can be modified with `--cds-queue-threshold N`. The experimental public dashboard read can be disabled with `--no-cds-queue-check`; an unreadable dashboard is recorded as **unknown**, not as a low queue.

**Scientific caution:** ERA5, measured soundings and USSA76 can yield different molecular conditions. Preserve reference outputs before switching atmosphere modes: the modes use the same canonical Level 1 filename, and a changed Level 1 may require reprocessing downstream Level 2.

## 5. Select Level 2 observations

~~~bash
milgrau-lebear -i SESSION_ID --regime night
milgrau-lebear -i SESSION_ID --segment seg01
milgrau-lebear -i SESSION_ID --time-window-utc 01:00 03:00
~~~

A solar regime can contain multiple disjoint segments; `seg01` identifies one contiguous segment. These selections produce derived views/products; they do not redefine the original continuous acquisition.

## 6. Reuse or refresh outputs

With incremental processing enabled, current scientific products may be skipped. To explicitly reprocess, use the stage's `--force` option after preserving any needed scientific baseline.

To regenerate only Level 1 figures:

~~~bash
milgrau-lipancora --figures-only -i SESSION_ID
~~~

Add `--force` to regenerate all selected figures; figure-only processing does not recompute Level 1.

## 7. Locate and inspect the results

Products normally appear under:

~~~text
02-processed_data/spu/YYYY/MM/SESSION_ID/
  SESSION_ID_L0.nc
  SESSION_ID_L1.nc
  SESSION_ID_L2.nc
  figures/
  scc/                    # when SCC is enabled
~~~

Use `milgrau-inspect SESSION_ID` to inspect availability, session metadata and product summaries. Use `milgrau-explorer` and open `http://127.0.0.1:8765` for read-only scientific visualization.

A successful command is **not** a scientific quality certificate. Check valid support, provenance, gaps, background/saturation diagnostics, molecular reference and retrieval flags before interpreting or publishing results. See [Scientific quality](scientific_quality.md).

## 8. Optional SCC workflow

Set `scc.enabled: true` only after reviewing the configured station and `scc.yaml` catalog. SCC inputs are prepared locally while processing; they are not remotely submitted by LIBIDS or LIPANCORA. Validate offline first:

~~~bash
milgrau-scc upload --dry-run
milgrau-scc status
~~~

Remote upload, remote status, download and same-ID replacement have distinct operational consequences. Review the [SCC operator guide](scc_batch_operator_guide.md) before enabling them. Never infer that a locally prepared package has already been accepted by the SCC.

## Further reading

[LIBIDS](libids.md) · [LIPANCORA](lipancora.md) · [Atmospheric profiles](atmospheric_profiles.md) · [LEBEAR](lebear.md) · [Explorer](explorer.md)
