# KFS boundary-condition R&D

## Scope

MILGRAU productive Level 2 remains schema v3 / retrieval method v4. The productive backward KFS boundary is the exact measured RCS bin at the selected Rayleigh-like reference altitude, with configured `aerosol_ref_fraction: 0.0`.

This document records why that aerosol-free reference assumption is now treated as a distinct R&D uncertainty dimension. It does **not** define a new selector, threshold, correction or productive method.

## Why Rayleigh-window QA is not the same as boundary validation

The elastic KFS inversion requires a total-backscatter boundary at one exact altitude. In productive method v4 this is supplied as

`beta_total(ref) = beta_mol(ref)`.

The Rayleigh catalogue tests whether the measured-to-molecular signal relationship across a window satisfies configured shape/validity criteria. Those tests are useful for rejecting obvious poor reference regions, but they do not independently observe `beta_aer(ref)`.

A broad, smooth aerosol contribution can remain approximately molecular in local window shape while `beta_aer(ref)` is non-zero. Because the Rayleigh calibration factor is itself obtained under the molecular-reference interpretation, using that same factor as proof of molecular purity would be circular.

## Controlled evidence

`docs/regression_baselines/p5_4_synthetic_residual_aerosol_boundary_sweep_20260917.json` isolates this mechanism with noiseless synthetic truth. Broad residual aerosol is prescribed directly at the reference altitude while the Rayleigh-like window continues to pass the current shape QA. Using the true total-backscatter boundary recovers the synthetic lower-column truth; forcing the aerosol-free boundary produces an increasing lower-column bias as residual aerosol increases.

`docs/regression_baselines/p5_4_synthetic_boundary_residual_noise_lr_20260917.json` separates signal noise and lidar-ratio mismatch. Signal noise broadens profile error but does not remove the systematic boundary bias. Lidar-ratio error can either amplify or partially compensate boundary bias, so apparent lower-column agreement under one lidar-ratio assumption cannot validate the boundary by itself.

`tests/test_kfs_residual_aerosol_boundary_rnd.py` keeps these mechanisms executable. It also contains a counterexample for boundary-placement stability: localized contamination can be exposed by moving the reference, but broad contamination can remain placement-stable while the lower column stays biased. Placement stability is therefore diagnostic, not a purity certificate.

## Real-data sensitivity is not an aerosol estimate

`docs/regression_baselines/p5_4_campaign_boundary_fraction_sensitivity_20260917.json` evaluates the 150 successful campaign block/wavelength retrievals under caller-declared scenarios

`beta_total(ref) = beta_mol(ref) * (1 + f)`.

The scenario fraction `f` is **not inferred from the measurement and is not assigned a probability**. The experiment only measures how strongly the retrieved lower column depends on an unobserved boundary residual. Even small declared fractions produce material changes in many real profiles, showing that the boundary assumption is scientifically consequential independently of high-altitude path support.

`milgrau.level2.boundary_sensitivity.boundary_fraction_sensitivity_profiles` exists only to make this scenario analysis reproducible. It returns one deterministic backward-KFS profile for each explicitly supplied fraction. It deliberately provides no score, preferred scenario or decision.

## What the current elastic diagnostics can and cannot do

Across the heterogeneous observational campaign, existing candidate slope, variance, SNR, diagnostic cost and path completeness do not consistently rank lower-column sensitivity to a higher accepted boundary. Reweighting those fields into a new composite score is therefore not supported by current evidence.

The useful interpretation is narrower:

* shape QA can reject clearly non-Rayleigh-like windows;
* SNR/dependence diagnostics characterize precision/support;
* path diagnostics determine whether backward inversion is numerically supported;
* none of those establishes the physical statement `beta_aer(ref) = 0`.

## Raman-companion opportunity

The SPU station metadata identifies Raman companion wavelengths and the current `spu-merionc-2024` night SCC configuration includes `387.AN/PC` and `530.AN/PC`. The supplied `20250629sant` Level 1 contains corrected 387/530 signals with useful block SNR through the altitude range relevant to P5.4.

This is a promising route to independent boundary validation, but MILGRAU does not yet document enough companion-channel physics to derive a defensible Raman extinction or boundary constraint. Before using those channels scientifically, the project must establish at minimum:

* Raman species / scattering-process identity for each companion channel;
* effective detection wavelength and receiver/filter spectral response;
* required molecular cross-section and temperature dependence;
* calibration and overlap semantics relevant to the companion channel;
* aerosol-wavelength relationship required by any Raman extinction formulation;
* uncertainty propagation and validation against a trusted external implementation or controlled truth.

The mere presence of a channel or high SNR is not a validated Raman product.

## Method-v5 gate

A method-v5 high-column proposal is not authorized until the project can either constrain the boundary condition with independent information or carry its uncertainty/sensitivity honestly enough to preserve scientific meaning in the lower column. Vertical aggregation, a higher candidate altitude, temporal persistence or a more elaborate Rayleigh score cannot substitute for that requirement.
