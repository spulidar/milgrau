# Geometrical overlap model

MILGRAU contains a first-order geometrical overlap model to keep near-field support scientifically explicit while the SPU-Lidar overlap is being characterized experimentally.

## Status and scope

The current model is **diagnostic only**. It does not divide Level 0/Level 1 signals by an overlap curve, does not modify the productive Klett–Fernald retrieval, and does not by itself declare low-altitude aerosol products scientifically supported.

Station/instrument values live in `station.yaml`. Python contains only the generic geometry and validation/resolution logic. A profile may provide its own transmitter geometry; otherwise the resolver uses the explicitly declared station fallback and records `parameter_source=station_fallback`.

The current model identifier is:

`coaxial_uniform_disk_geometric_v1`

Its assumptions are:

- coaxial transmitter and receiver axes;
- parallel axes (zero relative tilt);
- circular telescope aperture;
- circular uniform-intensity laser footprint;
- linear beam-radius growth from the configured full-angle divergence;
- receiver FOV represented as a full angular field with a focal-plane field stop;
- no central obstruction, real beam-shape structure, vignetting, defocus, filter-angle effect, or alignment drift.

Any non-zero configured axis separation or tilt is rejected for this model rather than silently ignored.

## First-order geometry

For telescope diameter `D_T`, laser diameter at launch `D_L`, full receiver FOV `Psi_T`, and full laser divergence `Psi_L`, the coaxial parallel-axis full-overlap range is

`R_full = (D_T + D_L) / (Psi_T - Psi_L)`

when `Psi_T > Psi_L`. If `Psi_T <= Psi_L`, this idealized geometry has no finite exact-full-overlap range.

This is the coaxial specialization of the emitter–receiver geometry described by Di Paolantonio, Dionisi, and Liberti (2022), who define the laser divergence and telescope FOV as full opening angles and give the corresponding full-overlap expression for parallel axes.

The modeled curve is more informative than the single full-overlap range. At each range, MILGRAU computes the fraction of the physical telescope aperture accepted by the receiver angular field for each point in the circular laser footprint, then area-averages that fraction over the laser disk using deterministic Gaussian quadrature.

The focal length is retained even though the configured angular FOV is sufficient for the first-order curve. It provides the useful paraxial consistency relation

`field_stop_diameter ~= focal_length * FOV_full_angle`.

## Current SPU provisional values

The current `station.yaml` deliberately records uncertain values as uncertain rather than promoting them to calibrated constants:

- telescope diameter: 0.30 m;
- focal length: 1.50 m;
- receiver FOV: 0.10 mrad, provisionally interpreted as **full angle**;
- laser beam diameter: about 0.04 m;
- laser divergence: less than 0.10 mrad; the present diagnostic conservatively uses 0.10 mrad as an unverified upper-bound model value;
- transmitter/receiver geometry: provisionally coaxial and parallel;
- reported full overlap: about 500 m AGL, retained as unverified historical/operator evidence.

Under exactly those assumptions, the first-order model does **not** reproduce the reported 500 m full-overlap statement. Because the configured FOV and model divergence are both 0.10 mrad, the analytic model has no finite exact-full-overlap range. The modeled overlap fraction at 500 m is approximately 0.028. With focal length 1.5 m and a 0.10 mrad full FOV, the implied focal-plane field-stop diameter is approximately 0.15 mm.

This disagreement is intentionally preserved as a diagnostic. It may indicate an incorrect FOV convention/value, an over-conservative divergence estimate, an inaccurate beam-diameter estimate, a different optical definition of the quoted overlap, or a configuration/alignment difference between the historical statement and the present instrument. MILGRAU must not tune parameters merely to force agreement with 500 m.

## Updating after an experimental campaign

The preferred campaign should establish, with traceable uncertainty where practical:

1. actual receiver FOV and whether the documented number is full or half angle;
2. physical field-stop/diaphragm diameter and effective focal length;
3. beam diameter after the beam expander, with a stated beam-width convention;
4. divergence by wavelength and whether it is full or half angle;
5. transmitter–receiver lateral separation and angular misalignment;
6. telecover/alignment mapping and/or another independent overlap retrieval such as horizontal or Raman-based characterization;
7. temporal stability after alignment interventions.

After characterization, update `station.yaml` (preferably a dated station/profile configuration if the optical state changes). A productive overlap correction must be a separate, explicitly versioned scientific change with uncertainty/support tests; changing the diagnostic metadata alone must not silently alter Level 1 signals.

## Relationship to retrieval support

A finite KFS value is not automatically a scientifically supported aerosol value. Until overlap is characterized, the very-low-altitude finite output remains algorithmic output only. The future altitude-resolved retrieval support contract must carry both a lower instrument-validity boundary and the upper inversion/reference boundary.

The present 500 m statement is therefore evidence for planning and QA, not yet a hard productive cutoff and not a correction curve.

## References

- Di Paolantonio, M., Dionisi, D., and Liberti, G. L. (2022), *A semi-automated procedure for the emitter–receiver geometry characterization of motor-controlled lidars*, Atmospheric Measurement Techniques, 15, 1217–1231. DOI: `10.5194/amt-15-1217-2022`.
- Comerón, A. et al. (2023), *An explicit formulation for the retrieval of the overlap function in an elastic and Raman aerosol lidar*, Atmospheric Measurement Techniques, 16, 3015–3026. DOI: `10.5194/amt-16-3015-2023`.
