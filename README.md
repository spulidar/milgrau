# MILGRAU 🌩️

**Multi-Indexed LALINET GeneRAlized and Unified algorithm**

MILGRAU is a Python-based atmospheric lidar processing suite developed for the **SPU-Lidar Station** at **IPEN/USP, São Paulo, Brazil**.

The system is designed to process raw Licel lidar measurements into physically traceable atmospheric products, from raw signal standardization to range-corrected signals and, under development, Level 2 aerosol optical-property retrievals.

---

# Installation

## 1. Install requirements

MILGRAU requires:

- Git
- Python 3.12 or newer and pip
- venv support for virtual environments

---

### Debian / Ubuntu

```bash
sudo apt update
sudo apt install git python3 python3-pip python3-venv
```
---

### Arch / CachyOS / Manjaro

```bash
sudo pacman -Syu
sudo pacman -S git python python-pip
```

---

### Fedora

```bash
sudo dnf install git python3 python3-pip
```

---

### macOS

First, install Homebrew if it is not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Git and Python:

```bash
brew install git python
```

---

### Windows

On Windows, use **PowerShell**.

Use `winget` or install Git and Python manually using the official installers:

Git: https://git-scm.com/download/win

Python: https://www.python.org/downloads/windows/

When installing Python, make sure to enable:

```text
Add python.exe to PATH
```

---

### Notes about Python commands

Depending on the operating system, the Python command may be either `python` or `python3`, use whichever one works on your machine.


---

## 2. Download MILGRAU

Clone the repository:

```bash
git clone https://github.com/spulidar/milgrau.git
cd milgrau
```

## 3. Create a virtual environment

Inside the MILGRAU directory:

```bash
python -m venv .venv
```

Activate the environment:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```
---

## 4. Install MILGRAU

For users who only want to run the program:

```bash
pip install .
```

### Developer mode

Use this option if you are going to edit the MILGRAU source code:

```bash
pip install -e ".[dev]"
```

This installs MILGRAU in editable mode and includes development dependencies such as `pytest`.

Run the test suite with coverage using the same command as CI:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider \
  --cov=milgrau \
  --cov-report=term-missing \
  --cov-report=xml
```

The terminal report shows the current baseline. `coverage.xml` is generated for CI publication; no minimum percentage is enforced yet.

---

## Update MILGRAU

To download new changes from GitHub, run in the milgrau repository:

```bash
git pull
```

If the package was not installed in developer mode, reinstall it:

```bash
pip install .
```

---

## Basic usage

Activate the virtual environment whenever you want to use MILGRAU:

```bash
cd milgrau
source .venv/bin/activate
```

After installation and set up, you can run the modules:

```bash
milgrau-libids
milgrau-lipancora
milgrau-liracos
milgrau-lebear
```

Pipeline commands use consistent process exit codes: `0` when all inputs succeed or are skipped, `1` for a mixed batch with recoverable failures, and `2` when every input fails or a fatal error stops execution.

Logging verbosity is controlled independently with `processing.console_level` and `processing.file_level` (for example, `DEBUG`, `INFO`, or `WARNING`). Reconfiguring a MILGRAU logger replaces only handlers created by MILGRAU; externally attached capture or telemetry handlers are preserved.

The complete configuration inventory is in [docs/configuration_inventory.md](docs/configuration_inventory.md). Keys marked `DORMANT` are validated and preserved but do not affect pipelines; their decisions and required approvals are recorded in [docs/configuration_dormant_decisions.md](docs/configuration_dormant_decisions.md). Incremental reuse and invalidation are specified in [docs/incremental_provenance.md](docs/incremental_provenance.md). The internal Level 2 result model is documented in [docs/retrieval_contract.md](docs/retrieval_contract.md), and its reproducible performance protocol and current baseline are in [docs/level2_benchmarks.md](docs/level2_benchmarks.md). The approved, not-yet-implemented temporal schema v2 decision is recorded in [docs/level2_temporal_representation_proposal.md](docs/level2_temporal_representation_proposal.md).

LEBEAR writes a temporary NetCDF beside the destination and replaces the final file atomically only after a successful write. Optional Level 2 QA runs afterward; a plotting failure is reported separately and does not invalidate or remove the completed NetCDF product. Set `visualization.level2_qa.enabled` to enable or disable that stage explicitly.


# About MILGRAU

## Scientific purpose

Elastic and Raman lidar systems measure atmospheric backscattered radiation as a function of time and range. These raw signals are affected by instrumental response, photon-counting limitations, analog offsets, background radiation, dark current, incomplete overlap, clouds and atmospheric variability.

MILGRAU organizes the treatment of these signals into processing levels:

 **Level 0** — raw Licel measurements are parsed, quality-controlled and standardized into NetCDF files compatible with SCC-style processing.
 
 **Level 1** — instrumental corrections are applied and Range Corrected Signals are generated with propagated uncertainties.
 
 **Level 2** — aerosol optical properties will be retrieved through molecular calibration, signal gluing and Klett-Fernald-Sasano inversion.

---

## Processing modules

| Module | Level | Scientific role |
|---|---:|---|
| **LIBIDS** | Level 0 | Parses raw Licel binary files, builds measurement inventories, associates dark-current acquisitions, applies basic acquisition quality control and writes SCC-compatible Level 0 NetCDF files. |
| **LIPANCORA** | Level 1 | Applies detector and instrumental corrections, propagates uncertainties, computes corrected signal and Range Corrected Signal, estimates PBL height and integrates radiosonde thermodynamic information. |
| **LIRACOS** | Visualization | Generates Range Corrected Signal quicklooks, mean profiles and uncertainty bands for Level 1 scientific inspection. |
| **LEBEAR** | Level 2 | Under development. Intended for Analog/Photon Counting gluing, Rayleigh molecular calibration, cloud screening and Klett-Fernald-Sasano aerosol backscatter/extinction retrieval. |

---


## Level 0 — LIBIDS

LIBIDS converts raw Licel measurements into standardized Level 0 NetCDF files.

Raw-data discovery is read-only: spurious extensions are detected and logged but never moved or deleted during inventory construction. Maintenance code must call `quarantine_file(s)` or `delete_file(s)` from `milgrau.io.filesystem` explicitly; both APIs report structured, idempotent outcomes.

Main tasks:

- scan raw measurement folders;
- identify valid measurement and dark-current files;
- parse Licel headers and binary payloads;
- classify acquisitions into `am`, `pm` and `nt`;
- associate dark-current profiles with nearby measurements;
- reject invalid laser-shot acquisitions;
- fetch or fallback to surface meteorological metadata;
- write SCC-compatible Level 0 NetCDF products.

The Level 0 product stores raw lidar data as:

```text
Raw_Lidar_Data(time, channels, points)
```

and includes metadata such as:

```text
Measurement_ID
System
Latitude_degrees_north
Longitude_degrees_east
Accumulated_Shots
RawData_Start_Date
RawData_Start_Time_UT
RawData_Stop_Time_UT
Temperature_C
Pressure_hPa
CloudCover_percent
Source_File_Count
Source_Files
```

When available, dark-current measurements are stored as:

```text
Background_Profile(time_bck, channels, points)
```

---

## Level 1 — LIPANCORA

LIPANCORA transforms Level 0 raw signals into corrected Level 1 lidar products.

The correction sequence includes:

1. dark-current subtraction;
2. photon-counting normalization;
3. non-paralyzable dead-time correction;
4. bin-shift alignment;
5. sky-background subtraction;
6. uncertainty propagation;
7. range correction.

The Level 1 NetCDF explicitly stores both the corrected signal and the Range Corrected Signal:

```text
corrected_signal(time, channel, altitude)
corrected_signal_error(time, channel, altitude)

range_corrected_signal(time, channel, altitude)
range_corrected_signal_error(time, channel, altitude)
```

This distinction is scientifically important.

`corrected_signal` is the lidar signal after instrumental corrections but before multiplication by range squared.

`range_corrected_signal` is defined as:

```text
RCS(z) = corrected_signal(z) · z²
```

and is the primary signal used for Level 1 visualization and future Level 2 inversion.

LIPANCORA also enriches the Level 1 product with atmospheric diagnostics:

```text
PBL_Height_km(time)
Radiosonde_Temperature_K(radiosonde_altitude)
Radiosonde_Pressure_hPa(radiosonde_altitude)
```

when radiosonde data are available.

---

## Atmospheric diagnostics

### Planetary Boundary Layer

The PBL height is estimated from the vertical gradient of a smoothed Range Corrected Signal profile. The method searches for the strongest physically meaningful negative gradient within a configured altitude interval.

Typical configuration keys:

```yaml
physics:
  pbl_min_search_m: 500.0
  pbl_max_search_m: 4000.0
  pbl_smooth_bins: 15
```

### Tropopause

When radiosonde data are available, MILGRAU estimates:

- **Cold Point Tropopause**;
- **Lapse Rate Tropopause**, following a WMO-style lapse-rate criterion.

These diagnostics are stored as global Level 1 attributes and can be overlaid on visual products.

---

## Molecular atmosphere and Rayleigh calculations

The Level 2 development path depends on a molecular reference profile.

MILGRAU includes routines to compute molecular backscatter and extinction from pressure and temperature profiles:

```text
β_mol(z, λ)
α_mol(z, λ)
```

where pressure and temperature may come from radiosonde data or a fallback standard atmosphere.

For Rayleigh scattering, the molecular lidar ratio is treated as:

```text
S_mol = 8π / 3 sr
```

The molecular profile is required for Rayleigh calibration and for the Klett-Fernald-Sasano inversion.

---

## Signal gluing

LEBEAR will use Analog and Photon Counting channels together when both are available for the same wavelength.

The gluing procedure is designed to search for a stable overlap region where the two signals are physically compatible. The selection criteria include:

- high Pearson correlation;
- stable linear mapping between Analog and Photon Counting signals;
- acceptable relative intercept;
- low residual spread;
- compatible min/max envelope after scaling.

The glued signal is intended to preserve near-field analog dynamic range while retaining photon-counting sensitivity at higher altitudes.

---

## Klett-Fernald-Sasano inversion

The planned Level 2 retrieval uses a Klett-Fernald-Sasano-type inversion to estimate aerosol optical properties from calibrated elastic lidar signals.

The expected Level 2 products include:

```text
aerosol_backscatter(time, wavelength, altitude)
aerosol_backscatter_error(time, wavelength, altitude)

aerosol_extinction(time, wavelength, altitude)
aerosol_extinction_error(time, wavelength, altitude)
```

The inversion will use:

- molecular backscatter and extinction;
- calibrated Range Corrected Signal;
- configured aerosol lidar ratios;
- molecular reference altitude selection;
- Monte Carlo perturbations for uncertainty estimation.

---

## Configuration

MILGRAU is configured through `config.yaml`.

The configuration file contains:

- directory paths;
- station metadata;
- instrument constants;
- channel-specific dead-time corrections;
- bin-shift values;
- background offsets;
- PBL search limits;
- radiosonde settings;
- visualization settings;
- Level 2 inversion parameters;
- monthly aerosol lidar ratios;
- cloud-screening parameters.

Physical constants, correction thresholds and processing options should be configured externally and not hardcoded in processing scripts.

Example channel configuration:

```yaml
physics:
  channels:
    "532.PC": [0.0035, -3, 0.0015]
    "532.AN": [0.0000,  6, 0.0000]
```

where the values are:

```text
[deadtime_us, bin_shift, background_offset]
```

---
