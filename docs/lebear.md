# LEBEAR — Level 2: aerosol optical retrieval

LEBEAR derives elastic-lidar aerosol optical properties from validated Level 1 signals and the molecular atmosphere. The present scientific retrieval uses a **two-sided Klett–Fernald–Sasano (KFS) inversion**.

## What the retrieval can and cannot determine

The elastic lidar equation may be expressed schematically as

$
P(r)=\frac{C\,O(r)}{r^2}\,\beta(r)\,
\exp\left[-2\int_0^r\alpha(s)\,\mathrm{d}s\right].
$

Here $P(r)$ is received signal, $C$ an instrument constant, $O(r)$ the overlap function, $\beta(r)$ total backscatter, and $\alpha(r)$ total extinction. Range $r$ denotes distance along the optical path; the processed altitude coordinate is reported relative to the lidar.

The total contributions contain molecular and aerosol terms:

$
\beta=\beta_{\mathrm{mol}}+\beta_{\mathrm{aer}},\qquad
\alpha=\alpha_{\mathrm{mol}}+\alpha_{\mathrm{aer}}.
$

For an elastic retrieval, the aerosol lidar ratio $S_{\mathrm{aer}}=\alpha_{\mathrm{aer}}/\beta_{\mathrm{aer}}$ and the reference boundary are **assumptions/conditions of the inversion**, not independent measurements of aerosol extinction. The resulting extinction cannot be described as Raman-measured extinction.

## Inputs and run commands

LEBEAR requires a validated `SESSION_L1.nc` containing corrected/RCS signals, uncertainty information and the materialized temperature/pressure atmosphere.

~~~bash
milgrau-lebear
milgrau-lebear -i SESSION_ID
milgrau-lebear -i SESSION_ID --regime night
milgrau-lebear -i SESSION_ID --segment seg01
milgrau-lebear -i SESSION_ID --time-window-utc 01:00 03:00
~~~

The selection options create derived Level 2 products and keep the source session identity. A day/night regime can contain multiple disjoint segments, whereas one `segXX` is contiguous.

## Scientific sequence

1. **Time and segment selection.** Valid Level 1 observations are grouped into configured time blocks without mixing different scientific segments.
2. **Channel preparation.** Where both analog and photon-counting signals are suitable, a documented gluing procedure constructs a usable signal; explicitly permitted single-channel fallback is identified.
3. **Input screening.** Signal, uncertainty and other retrieval input requirements are checked.
4. **Molecular atmosphere.** Level 1 temperature and pressure are interpolated in time for each block, and molecular Rayleigh backscatter and extinction are evaluated. LEBEAR does not download meteorology itself.
5. **Reference selection.** Candidate molecular-reference regions are evaluated for usable path, signal and Rayleigh-consistency diagnostics.
6. **Background and normalization.** Residual background and molecular normalization are fitted and their uncertainty information retained.
7. **Elastic inversion.** KFS integration proceeds backward below the selected reference and forward above it, within physically and numerically supported ranges.
8. **Uncertainty evaluation.** The implemented selection-aware Monte Carlo workflow propagates measurement variation through background fitting, reference reselection and inversion.
9. **Results and figures.** Optical products, validity information, errors, diagnostics and plots are saved.

A reference chosen high in the atmosphere is not necessarily more molecular or scientifically better. The reference is evaluated through explicit quality measures.

## Two-sided KFS and supported altitude

The backward and forward branches begin from the selected molecular reference. A branch stops at the first unsupported cell or numerical failure; a finite diagnostic value above a gap does not restore a valid continuous inversion path.

The file stores validity and endpoint information separately for backward and forward branches. **Do not assume that the retrieval extends to a fixed height (for example, 20–25 km) simply because the altitude grid extends there.**

The optical result is conditional on its selected reference, assumed lidar ratio and available signal support. Close-to-instrument overlap limitations and detector saturation require independent instrument evidence.

## Products

The Level 2 NetCDF can report:

| Quantity | Scientific meaning |
| --- | --- |
| Aerosol backscatter coefficient | Retrieved aerosol contribution to the lidar backscatter |
| Aerosol extinction coefficient | Derived aerosol extinction under the assumed lidar ratio |
| Scattering ratio | Ratio of total to molecular backscatter; a diagnostic, not proof of valid aerosol inversion |
| Molecular backscatter/extinction | Calculated from Level 1 atmospheric state at each retrieval block |
| Temporal support and valid fraction | Fraction/count of measurements or realizations supporting the reported value |
| Uncertainty estimates and quantiles | Dispersion from the documented Monte Carlo and aggregation procedures |
| Reference and branch diagnostics | Evidence about selected reference, validity and usable altitude range |

For precise NetCDF variable names, coordinates and units, consult the [Level 2 variable guide](level2_schema.md).

## Interpreting uncertainties and missing data

- One-sigma signal uncertainty and the set of selected usable samples matter for averaging.
- Monte Carlo estimates are **conditional** on the implemented input-noise and nuisance assumptions; they are not total uncertainty from every possible physical systematic error.
- Unsupported cells remain `NaN`. They must not be interpreted as zero aerosol.
- Temporal means may combine unequal numbers of valid blocks at different heights; inspect `temporal_support_fraction` and counts.
- A finite scattering ratio or molecular profile does not alone certify a finite, valid aerosol optical retrieval.
- Product completeness can be partial by wavelength or time block. Check status and recorded failure reasons.

## What to inspect in a scientific example

1. Confirm session date, segment and wavelengths.
2. Verify detector/correction diagnostics inherited from Level 1.
3. Inspect the source of molecular temperature and pressure for the selected block.
4. Check signal selection/gluing and residual-background behavior.
5. Review chosen molecular-reference range, boundary condition and stability.
6. Compare the valid backward/forward endpoints against the altitude range used in your scientific interpretation.
7. Interpret backscatter/extinction **together with assumed lidar ratio**, uncertainty and support.
8. If comparing to SCC or other chains, first ensure comparable signals, atmosphere, time averaging and inversion assumptions.

## References and limitations

The original inversion methods and Rayleigh-scattering references are listed in [Scientific traceability](scientific_traceability.md). Important unresolved physical questions—including independent overlap/saturation characterization and the provenance of monthly lidar-ratio assumptions—remain open scientific-evidence tasks rather than solved by this documentation.

For general acceptance criteria, see [Scientific quality](scientific_quality.md).
