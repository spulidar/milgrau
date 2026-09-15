# CF-aligned metadata policy

MILGRAU uses the Climate and Forecast (CF) Metadata Conventions as an interoperability guide for its NetCDF products, but it does **not** currently claim formal CF compliance.

## Current policy

- Use CF-compatible coordinate, unit, flag, missing-value, and descriptive-metadata patterns whenever they fit the lidar product semantics.
- Do not invent a CF `standard_name` when no appropriate standard name exists; use clear `long_name`, units, and MILGRAU-specific metadata instead.
- Keep instrument- or method-specific provenance and diagnostics as explicit non-standard metadata where necessary. CF permits additional non-standard attributes when they do not conflict with CF semantics.
- Do **not** write a global `Conventions = "CF-..."` attribute unless MILGRAU intentionally decides to make and maintain a formal compliance claim for that product schema.
- Therefore, current MILGRAU products should be described as **CF-aligned** or **using CF metadata conventions**, not as **CF-compliant**.

The released CF 1.13 specification requires a conforming CF-1.13 file to identify itself with a global `Conventions` attribute containing `CF-1.13`. Omitting that declaration is an intentional choice not to claim formal conformance.

## Validation tooling

The project pins IOOS Compliance Checker 6.1.0 in the optional `validation` dependency group:

```bash
pip install -e ".[validation]"
```

The current checker has built-in CF tests through CF 1.11. It can still be used as a strong metadata QA tool even though MILGRAU does not make a formal CF compliance claim:

```bash
compliance-checker --test=cf:1.11 path/to/product.nc
```

Checker output must be interpreted scientifically. A warning is not automatically a MILGRAU defect, and passing the checker is not evidence that the lidar retrieval physics is correct. Any accepted exception should be documented rather than hidden by inventing metadata.

## Scope

CF-oriented QA concerns metadata and data-model interoperability, including coordinate semantics, units, time representation, flags, missing values, and variable descriptions. It does not validate Klett-Fernald-Sasano physics, lidar-ratio assumptions, overlap characterization, Rayleigh-reference quality, detector corrections, or uncertainty-model correctness.

Formal CF compliance may be reconsidered later as a release-governance decision. If that policy changes, the selected CF version, `Conventions` declaration, checker evidence, and schema implications must be reviewed together.
