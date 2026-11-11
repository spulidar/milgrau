# Solar elevation and day/night segments — Level 0 scientific validation

LIBIDS assigns a geometric solar elevation and a day/night label to each accepted lidar acquisition. These are **calculated observational descriptors**, not measurements of sunlight, clouds or aerosol content.

## Physical model and time used

The production Level 0 path evaluates the solar position at each acquisition's **UTC midpoint**,

$$
t_{\mathrm{mid}}=t_{\mathrm{start}}+
\frac{t_{\mathrm{stop}}-t_{\mathrm{start}}}{2}.
$$

The `milgrau.physics.solar.solar_elevation_deg` calculation uses a Julian-century solar-geometry approximation based on NOAA-style solar-position equations. It computes geometric solar-center elevation without atmospheric-refraction correction.

The geometry follows

$$
\cos\theta_z=\sin\varphi\sin\delta+
\cos\varphi\cos\delta\cos h,
\qquad e=90^\circ-\theta_z,
$$

where `φ` is station latitude, `δ` solar declination, `h` solar hour angle, `θz` zenith angle and `e` the resulting geometric elevation. Longitude is measured positive east, as in the station catalog.

The configured regime is

$$
\mathrm{regime}(t)=
\begin{cases}
\mathrm{day},&e(t)\ge e_{\mathrm{threshold}},\\
\mathrm{night},&e(t)<e_{\mathrm{threshold}}.
\end{cases}
$$

The distributed threshold is **−3.0°**. This is a project-specific analysis threshold, **not the conventional astronomical sunrise/sunset horizon**, and does not include measured refraction or weather.

References: [NOAA Solar Calculation Details](https://gml.noaa.gov/grad/solcalc/solareqns.PDF) and [NOAA solar geometry calculation](https://gml.noaa.gov/grad/neubrew/SolarCalc.jsp). The simplified NOAA fractional-year equation is also used as an *independent approximate numerical reference* in `tests/test_documentation_operational_contracts.py`.

## One session can contain several segments

Each time an ordered sequence of accepted measurement profiles changes from day to night, or vice versa, Level 0 assigns a **new contiguous segment**:

| Profile order | Solar regime | Segment |
| --- | --- | --- |
| 1 | day | `seg00` |
| 2 | night | `seg01` |
| 3 | night | `seg01` |
| 4 | day | `seg02` |

A later second night receives a **different segment ID**, not `seg01`. Consequently, a single `--regime night` selection in Level 2 can combine disjoint night segments, but `--segment seg01` selects one interval only.

Segment start/end timestamps use the first profile start and last profile end of that contiguous portion. The original scientific session ID remains continuous; no 6-hour publication-window splitting occurs here.

## How to inspect real observations

Check these Level 0 variables, rather than guessing the label from local wall-clock time:

| Field | Interpretation |
| --- | --- |
| `solar_elevation_deg(time)` | Geometric solar-center elevation at acquisition midpoints [degree] |
| `solar_regime(time)` | `day` or `night`, based on the configured threshold |
| `segment_id(time)` | Contiguous regime label |
| `Segment_Label(segments)` | Ordered segment labels |
| `Segment_Regime(segments)` | Corresponding regime |
| `Segment_Start_Time_UTC(segments)` | Segment acquisition start, Unix seconds UTC |
| `Segment_End_Time_UTC(segments)` | Segment acquisition end, Unix seconds UTC |

The global attributes `Solar_Day_Night_Threshold_deg` and `Solar_Position_Algorithm` record the applied rule. The station site coordinates and timezone come from the dated station metadata.

## Numerical and logical regression evidence

- `tests/test_solar.py` already checks day/night at SPU, the −3.0° boundary and contiguous repeated regimes.
- `tests/test_documentation_operational_contracts.py` compares MILGRAU's solar elevations at equinox/solstice times and multiple latitudes to a **separately implemented** NOAA fractional-year approximation. The 2° test tolerance reflects differences between these approximate methods and does not establish sub-degree astronomical accuracy.
- The same added test checks that `seg00/01/02/03` represent successive contiguous regimes, including returns to a previous regime.

**Pandas-unit regression discovered during this review.** The current code now normalizes
`DatetimeIndex` to nanoseconds before deriving Julian day; the first
independent tests revealed errors of many degrees for microsecond-backed
timestamps. The physical solar model and the configured −3° regime threshold
were not changed. An earlier product may have recorded different segment
labels, depending on how its pandas datetime index was represented.

**Remaining field-validation task:** select a real, dated SPU measurement spanning the configured solar transition, inspect acquisition midpoints and station-profile selection, compare against an independent solar ephemeris and record the numerical discrepancy. Use the [read-only historical audit procedure](level0_validation_workbook.md#d-read-only-check-for-historical-solar-labels) before considering reprocessing. This requires actual source observations and is not closed by automated synthetic tests.

## Canonical solar calculation

The productive solar-context calculation is `milgrau/level0/processing.py::_annotate_solar_context`, which uses the physical solar-position functions in `milgrau/physics/solar.py` and the **acquisition midpoint**. A former unreferenced duplicate, `milgrau/level0/solar.py`, computed elevation at the acquisition **start**; it was retired during the November 2026 cleanup.

The removal of that unused alternative does **not** change the productive solar equations, its midpoint convention or its −3° threshold. The separate pandas microsecond/nanosecond correction is documented above and **can** affect historical outputs.

See [LIBIDS](libids.md), [Level 0 variables](level0_schema.md) and [Cleanup review](cleanup_review.md).
