# MILGRAU processing levels — scientific overview

MILGRAU processes atmospheric lidar observations through three scientific levels. Each level adds interpretation while retaining the link to the original continuous acquisition and relevant quality evidence.

| Stage | Main input | What it establishes | Principal output |
| --- | --- | --- | --- |
| **LIBIDS — Level 0** | Original Licel measurements and instrument record | Continuous session, standardized channels, acquisition metadata and checks | `SESSION_L0.nc` |
| **LIPANCORA — Level 1** | Valid Level 0 and instrument/atmospheric context | Corrected lidar signal, uncertainty and time-resolved molecular atmosphere | `SESSION_L1.nc` |
| **LEBEAR — Level 2** | Valid Level 1 | Conditional elastic aerosol backscatter/extinction and retrieval validity | `SESSION_L2.nc` |

The output of a level is not automatically evidence that every channel, wavelength or altitude is scientifically valid. Each stage records status, diagnostic and source information for interpretation.

## Continuous session and solar segments

A canonical scientific observation is a continuous session:

`spu_20250511-0012Z_20250511-0737Z`

Its UTC start/end times are based on raw acquisition continuity and applicable station/instrument history. The session can cross midnight or a six-hour publication boundary.

Day/night are determined from configured **geometric solar elevation** and saved for the actual lidar measurements. Contiguous regimes receive `segXX` identifiers. A later Level 2 selection of one regime or segment **does not change the original session identity**.

## Level 0 — LIBIDS

**Scientific question:** what was measured, when, by which instrument/channel, and with which acquisition context?

LIBIDS reads raw Licel measurements, identifies continuous sessions, applies acquisition-quality checks, associates applicable dark-current records and maps measured channels to historical instrument information. It preserves solar regime/segment identity and available surface weather without inventing missing measurements.

Level 0 does **not** calculate aerosol optical retrievals. Its NetCDF is the standardized source for subsequent corrected signals.

See [LIBIDS user and scientific guide](libids.md).

## Level 1 — LIPANCORA

**Scientific question:** what corrected lidar signal and molecular environment can be justified from these acquisitions?

Level 1 applies the implemented instrumental and background corrections, computes corrected/range-corrected signals and propagates available measurement uncertainties. Relevant diagnostics include detector/dead-time flags, background robustness and channel correction status.

It also materializes temperature and pressure on the lidar grid at hourly atmosphere times. The current preference is **ERA5 → Wyoming → NOAA IGRA2 → USSA76**, subject to configured time windows and the operational public CDS queue heuristic. Each actual source and any USSA76 extension is identified.

Level 1 is the **scientific source of the molecular atmosphere**. Level 2 consumes it rather than downloading a new one.

See [LIPANCORA](lipancora.md) and [Atmospheric profiles](atmospheric_profiles.md).

## Level 2 — LEBEAR

**Scientific question:** given the corrected signals, molecular state, reference and lidar-ratio assumptions, which elastic aerosol optical profiles are supported?

LEBEAR forms time blocks with consistent scientific segments, selects or glues appropriate signals, evaluates a molecular Rayleigh reference and performs a two-sided Klett–Fernald–Sasano inversion with selection-aware Monte Carlo diagnostics.

The present method saves aerosol backscatter, conditional aerosol extinction, scattering ratio and molecular fields, along with reference selection, branch endpoints, uncertainty and temporal-support information. Unsupported values remain missing rather than being extrapolated across invalid gaps.

The assumed lidar ratio affects the extinction result. A high reference or a finite scattering ratio alone does not establish a valid aerosol retrieval.

See [LEBEAR](lebear.md) and the [Level 2 variable dictionary](level2_schema.md).

## Figures, inspection and visualization

LIPANCORA and LEBEAR generate scientific figures alongside their products. Use `milgrau-inspect SESSION_ID` for a text overview and `milgrau-explorer` for local, read-only scientific exploration.

Figures may reduce display resolution and should not replace full-resolution NetCDF values or scientific support flags. The original dataset is not altered when you change visualization settings.

## Provenance and reproducibility

Saved products identify the observation, session, applicable station/calibration context, scientific recipe, software state and upstream products as available. Level 2 identifies the exact Level 1 file used; atmospheric sources are selected in Level 1 and propagated.

For reproducible comparisons, preserve originals, record processing settings, validate the source time/height mapping and compare the same observation/segment/altitude support under compatible assumptions.

## Optional external SCC workflow

SCC uses its own product identifiers, channel mappings and input formats. When enabled, LIBIDS/LIPANCORA can stage **separate SCC interoperability files** beside the canonical session. SCC processing does not redefine MILGRAU Level 0, Level 1 or Level 2. Remote SCC uploads are explicitly operator-initiated.

See the [SCC operator guide](scc_batch_operator_guide.md).

## Interpreting scientific quality

Compare **availability**, **numeric finiteness**, **retrieval support** and **scientific validity** separately. Check any instrumental assumption, atmosphere source, background quality, reference and uncertainty before interpreting optical results.

A staged or successfully executed product is a necessary operational milestone, not by itself a sufficient scientific validation. Refer to [Scientific quality and interpretation](scientific_quality.md).
