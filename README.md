# MILGRAU

**Multi-Indexed LALINET GeneRAlized and Unified Algorithm**

*Scientific processing of atmospheric lidar observations.*

MILGRAU is a scientific processing suite developed for atmospheric lidar measurements, with emphasis on the historical SPU-Lidar dataset operated by IPEN/USP in São Paulo, Brazil.

The software processes raw Licel acquisitions into standardized measurements, corrected lidar signals, and aerosol optical products, preserving the information needed to interpret and reproduce the scientific results.

## Scientific processing workflow

MILGRAU has three main processing levels:

| Level | Program | Scientific purpose |
| --- | --- | --- |
| Level 0 | LIBIDS | Organize, validate and standardize lidar measurements |
| Level 1 | LIPANCORA | Apply instrumental corrections and characterize the molecular atmosphere |
| Level 2 | LEBEAR | Retrieve aerosol optical properties using elastic lidar measurements |

Each processing level produces a NetCDF dataset containing scientific results, relevant quality diagnostics and provenance information.

### Level 0 — LIBIDS

LIBIDS reads raw Licel measurements and organizes them into continuous observation sessions.

Its main tasks include:

- Identifying measurements and their acquisition periods.
- Applying acquisition-quality checks.
- Associating appropriate dark-current measurements.
- Identifying the instrument configuration and available channels.
- Recording observation times and solar conditions.
- Preserving raw signals and their measurement context.

**Scientific output:** a standardized Level 0 NetCDF dataset that preserves the measurement information required for subsequent processing.

### Level 1 — LIPANCORA

LIPANCORA converts Level 0 measurements into instrumentally corrected lidar signals.

Processing includes instrumental corrections, background estimation, uncertainty propagation and range correction.

It also constructs time-resolved temperature and pressure profiles on the lidar altitude grid.

**Scientific output:** corrected signals, range-corrected signals (RCS), measurement uncertainties, atmospheric profiles and quality diagnostics.

### Level 2 — LEBEAR

LEBEAR uses corrected signals and the molecular atmosphere to retrieve aerosol optical properties.

The current method uses a two-sided elastic Klett–Fernald–Sasano inversion, with explicit molecular-reference selection and uncertainty evaluation.

The main scientific products include:

- Aerosol backscatter coefficients.
- Aerosol extinction coefficients conditional on the assumed lidar ratio.
- Scattering ratio.
- Molecular backscatter and extinction.
- Retrieval validity and uncertainty diagnostics.

**Scientific output:** time- and altitude-resolved aerosol optical products, including their assumptions, uncertainty information and valid retrieval coverage.

## Atmospheric information

The molecular atmosphere used in lidar inversion is determined during Level 1 processing.

The configured source preference is:

**ERA5 → Wyoming radiosonde → NOAA IGRA2 → US Standard Atmosphere 1976 (USSA76).**

ERA5 is preferred. Validated ERA5 data already in cache remain usable even during CDS congestion.

MILGRAU can inspect the public CDS queue indicator before requesting missing ERA5 data. When the reported queue is high, it avoids new CDS requests and uses available observational or standard-atmosphere fallbacks.

The `--radiosonde-only` option bypasses ERA5 and uses Wyoming, followed by IGRA2 and USSA76 if required.

The scientific products record the actual atmospheric source, radiosonde provider, temporal information and standard-atmosphere contribution. See [Atmospheric profiles](docs/atmospheric_profiles.md) for selection criteria and scientific limitations.

## Installation

MILGRAU requires **Python 3.12 or newer**.

Clone the repository and use the current **`main`** branch:

~~~bash
git clone https://github.com/spulidar/milgrau.git
cd milgrau
git switch main
python -m pip install -e .
~~~

Optional remote SCC support:

~~~bash
python -m pip install -e ".[scc]"
~~~

Configure `config.yaml` and `station.yaml` before processing. See [Getting started](docs/getting_started.md).

## Running MILGRAU

### Process all available measurements

Run the programs in order:

~~~bash
milgrau-libids
milgrau-lipancora
milgrau-lebear
~~~

### Process one observation session

For example, `spu_20250511-0012Z_20250511-0737Z`:

~~~bash
milgrau-libids -i spu_20250511-0012Z_20250511-0737Z
milgrau-lipancora -i spu_20250511-0012Z_20250511-0737Z
milgrau-lebear -i spu_20250511-0012Z_20250511-0737Z
~~~

### Use Level 1 without ERA5

~~~bash
milgrau-lipancora -i SESSION_ID --radiosonde-only
~~~

### Regenerate Level 1 figures without recalculating scientific data

~~~bash
milgrau-lipancora --figures-only -i SESSION_ID
~~~

### Inspect observations

~~~bash
milgrau-inspect SESSION_ID
milgrau-explorer
~~~

The Explorer offers read-only visualization of processed observations, atmospheric profiles and available scientific products.

## Scientific products and organization

Outputs are organized by station, year, month and continuous observation session. A typical session contains:

- `SESSION_L0.nc` — standardized lidar acquisition.
- `SESSION_L1.nc` — corrected lidar signals and atmospheric state.
- `SESSION_L2.nc` — aerosol optical products.
- `figures/` — scientific and diagnostic visualizations.
- `scc/` — optional SCC input packages and downloaded results.

The original session identity is maintained across processing levels. An acquisition may cross midnight; fixed publication windows do not define scientific sessions.

## Scientific quality and interpretation

MILGRAU records diagnostic information to examine:

- Instrumental and background corrections.
- Availability and provenance of atmospheric data.
- Molecular-reference suitability.
- Valid altitude ranges.
- Data gaps and unsupported retrieval regions.
- Retrieval uncertainties and scientific assumptions.

Elastic extinction depends on the assumed aerosol lidar ratio. Retrieved optical quantities and their uncertainties must be interpreted in light of the documented inversion assumptions. Successfully running a command alone does not establish the scientific validity of every altitude or observation.

## SCC interoperability

MILGRAU supports optional interoperability with the **Single Calculus Chain (SCC)**.

Scientific Level 0, Level 1 and Level 2 products remain independent of SCC processing. When explicitly enabled, MILGRAU can prepare SCC-compatible inputs, attach validated observational radiosonde profiles, inspect remote status and retrieve available results.

Remote uploads require an explicit operator command; scientific processing never uploads automatically. See the [SCC operator guide](docs/scc_batch_operator_guide.md).

## Documentation

The same scientific Markdown in [`docs/`](docs/) powers the [Lotus Docs + Hugo website](docs/site_hugo.md). Generated website files are temporary and are not a second documentation source. Publication is automated through GitHub Pages when enabled.

Start with the [Documentation guide](docs/README.md). Dedicated user and scientific guides cover:

- [Installation and first processing run](docs/getting_started.md).
- [Operational commands and selectors](docs/operating_milgrau.md).
- [Scientific configuration reference](docs/configuration_reference.md).
- [Scientific terminology and units](docs/terminology_units.md).
- [Level 0 — LIBIDS](docs/libids.md).
- [Level 0 NetCDF variable dictionary](docs/level0_schema.md).
- [Solar regime interpretation](docs/solar_regimes.md).
- [Level 0 validation and historical audit](docs/level0_validation_workbook.md).
- [Level 1 — LIPANCORA](docs/lipancora.md).
- [Level 2 — LEBEAR](docs/lebear.md).
- [Atmospheric sources and provenance](docs/atmospheric_profiles.md).
- [Scientific quality, uncertainties and interpretation](docs/scientific_quality.md).
- [SCC interoperability](docs/scc_batch_operator_guide.md).
- [Explorer and visualization](docs/explorer.md).

Detailed [Level 2 variable definitions](docs/level2_schema.md) and [scientific references](docs/scientific_traceability.md) remain available for deeper study.

## Citation

If you use MILGRAU for scientific work, consult [CITATION.cff](CITATION.cff) for the appropriate software citation and version information.

## License

See [LICENSE](LICENSE).
