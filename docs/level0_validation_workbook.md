# Level 0 verification workbook — synthetic evidence and real-observation protocol

This workbook makes LIBIDS validation reproducible, without claiming that generic automated tests establish the full scientific correctness of SPU-Lidar's historical archive.

## Evidence classes

| Label | What it proves | What it does not prove |
| --- | --- | --- |
| **Structural/automated** | CLI, parser and NetCDF contracts behave as asserted on synthetic inputs | Historical instrument calibration or all observed acquisitions |
| **Mathematical cross-check** | Solar calculations agree within a stated tolerance with an independently implemented approximate formula | Exact ephemeris accuracy or observed cloud/solar brightness |
| **Scientific real-data comparison** | Same Licel inputs, settings and instrument context give documented physically interpretable outputs | Complete generalization to every historical configuration |

## A. Offline synthetic example: parse an analog and PC Licel file

The test fixture in `tests/test_licel.py` constructs a complete minimal binary Licel sample with two active channels: `532.AN` and `532.PC`. The 532 nm analog channel uses 12-bit ADC, a 500 mV DAQ range, 100 accumulated laser shots and the raw integer array `[100, 200, 300, 400]`.

For an analog sample of 100 counts, the parser computes

$$
V_{\mathrm{AN}}=
\frac{100}{100}\frac{500\ \mathrm{mV}}{2^{12}-1}
\approx 0.1221\ \mathrm{mV}.
$$

The photon-counting channel retains its input raw integer counts `[1, 2, 3, 4]` as numerical sample values rather than applying the analog ADC conversion.

Run the fixture tests in a development install:

~~~bash
python -m pip install -e ".[dev]"
pytest -q tests/test_licel.py tests/test_level0_quality.py
pytest -q tests/test_inventory.py tests/test_solar.py tests/test_level0_netcdf.py
pytest -q tests/test_documentation_operational_contracts.py
~~~

These are **synthetic, offline fixtures**, not a real SPU observation. They exercise channel parsing, acquisition QA, cross-midnight/session boundaries, nearby and missing dark associations, NetCDF writing and solar segmentation.

## B. Prepare a historical observation for independent comparison

Use *one known continuous SPU session* from each relevant historical instrument era, including:

1. Legacy pre-SCC configuration (if original source files survive).
2. APEL 2017.
3. Raman 2018.
4. MerionC 2024+.

Include at least one continuous session crossing midnight or a UTC month; one day/night transition with several profiles; and one case with dark current inside/outside the configured 12-hour window.

**Do not invent session IDs.** Obtain exact IDs from LIBIDS inventory or existing processed products. Use distinct baseline and candidate output roots or preserve immutable copies before reprocessing. Work from read-only raw sources.

For each selected case record:

| Field | Evidence to capture |
| --- | --- |
| Observation | Full canonical session ID and exact UTC start/end |
| Source | Original Licel path list, count and SHA-256 per file |
| Station context | Profile ID, calibration ID, instrument/channel set, vertical bin widths |
| QA | Accepted/rejected paths and per-file reason; nominal shots/durations |
| Sessionization | Actual gap values, boundaries, local-date overlap |
| Dark current | Associated files, distance from session and missing channels |
| Solar | Geometric midpoint elevation, regime transition and segment boundaries |
| Weather | Open-Meteo observation hours, missing-value count and scalar medians |
| L0 output | Filename, product SHA-256, dimensions, variables and flags |
| Baseline comparison | Numerical differences, expected change, explanation and reviewer/date |

### Capture hashes and preserve the baseline

With a known canonical NetCDF file, use Python (platform independent) to calculate its SHA-256:

~~~python
from hashlib import sha256
from pathlib import Path

path = Path("02-processed_data/spu/YYYY/MM/SESSION_ID/SESSION_ID_L0.nc")
digest = sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(path.name, digest.hexdigest())
~~~

Repeat for original Licel source files and any comparison product. A changed NetCDF file hash alone does not prove a changed scientific signal: re-encoding or metadata order may differ.

To inspect the Level 0 contract, use:

~~~bash
milgrau-inspect SESSION_ID --validate
milgrau-inspect SESSION_ID --full
~~~

The first command validates the **current structural contract**. The second prints a more extensive listing; inspect channel units, solar segment boundaries and provenance manually.

### Compare the baseline and candidate scientifically

For an accepted real sample, inspect equal dimensions and channel order; compare `Raw_Lidar_Data`, shots, bin resolution and observation timing under the *same* parser/configuration. Differences should be recorded with absolute and relative tolerances appropriate to the variable, not papered over by a universal arbitrary threshold.

Verify that changing from a full raw header scan to the persistent SQLite **header cache** does not change session identities, QA or canonical NetCDF scientific values. Record first-scan and repeat-scan wall times on the same hardware; a historical benchmark is not a new benchmark.

SCC reexport can be checked separately with `milgrau-libids -i SESSION_ID --scc-reexport`, with `scc.enabled: true` explicitly set. It must not rewrite the original canonical Level 0. Verify the original checksum before/after; avoid remote upload.

## C. Field review sign-off template

| Category | Reference | Result | Reviewer |
| --- | --- | --- | --- |
| Raw Licel integrity and header validity | Source hashes and headers | *Pending* | |
| Historical profile/calibration identity | Station catalog | *Pending* | |
| Acquisition QA and source rejection | QA output logs | *Pending* | |
| 30-minute continuity and local-day selection | Actual header times | *Pending* | |
| Dark-current association | Actual dark acquisition | *Pending* | |
| Solar midpoints and regime transition | Independent ephemeris | *Pending* | |
| Weather availability / missing values | Open-Meteo hourly source | *Pending* | |
| Raw tensor, shots and channel units | Baseline-vs-new L0 | *Pending* | |
| Header-cache scientific equivalence | First/second full run | *Pending* | |
| SCC staging retained canonical L0 | Before/after L0 hash | *Pending* | |

Until these rows are filled from actual observations, **do not mark the real-data scientific-equivalence check complete** in the tracker.

See [LIBIDS](libids.md), [Solar regime verification](solar_regimes.md), [Level 0 variables](level0_schema.md), [Scientific quality](scientific_quality.md).


## D. Read-only check for historical solar labels

Because pandas may store datetime values in microseconds or nanoseconds, older Level 0 products can carry different solar elevations even when their session timestamps are identical. A November 2026 correction normalized the astronomical Julian-day input to nanoseconds.

Use the following **read-only** script to compare the saved solar attributes with the current `main` calculation. Replace the path with an actual preserved NetCDF file. It does not change the original file:

~~~python
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from milgrau.physics.solar import solar_elevation_deg

path = Path("02-processed_data/spu/YYYY/MM/SESSION_ID/SESSION_ID_L0.nc")
with xr.open_dataset(path) as ds:
    reference = pd.Timestamp(ds.attrs["measurement_start_time"])
    start_seconds = np.asarray(ds["Raw_Data_Start_Time"].values[:, 0], dtype=float)
    stop_seconds = np.asarray(ds["Raw_Data_Stop_Time"].values[:, 0], dtype=float)
    middle = reference + pd.to_timedelta((start_seconds + stop_seconds) / 2, unit="s")
    expected = solar_elevation_deg(
        middle,
        float(ds.attrs["Latitude_degrees_north"]),
        float(ds.attrs["Longitude_degrees_east"]),
    )
    stored = np.asarray(ds["solar_elevation_deg"].values, dtype=float)
    threshold = float(ds.attrs["Solar_Day_Night_Threshold_deg"])
    predicted_regime = np.where(expected >= threshold, "day", "night")
    saved_regime = np.asarray(ds["solar_regime"].values).astype(str)

    print("Session:", ds.attrs["Session_ID"])
    print("Maximum solar-elevation difference [degree]:", np.nanmax(abs(expected - stored)))
    print("Changed day/night classifications:", int(np.sum(predicted_regime != saved_regime)))
~~~

A nonzero difference may reflect a prior software version or a real provenance/input discrepancy. **Do not automatically reprocess or delete previous L0/L1/L2 results** based on the printout. First retain hashes, date-resolved station coordinates, installed software identity and original source files, then perform the documented scientific comparison.

If segment classifications change, Level 2 regime/segment selections and SCC day/night partitions may also change. Such changes should be analyzed before interpreting formerly published optical products.

## E. Next step once historical data are available

Select the priority cases from section B, complete the evidence table, and sign off on the scientific differences independently. The exact result of this workbook must never be replaced by an assertion that CI or an independent mathematical approximation proved historical instrument accuracy.
