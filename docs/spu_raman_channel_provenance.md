# SPU Raman companion-channel provenance

## Purpose

This note records traceable evidence for SPU non-elastic detection channels relevant to future MILGRAU boundary-condition validation. It does **not** create a Raman Level-2 product and does not change productive elastic retrieval method v4.

The guiding rule is conservative: a historical receiver/filter number is not promoted into the active `spu-merionc-2024` profile unless current-instrument evidence shows that the value still applies.

## Current repository state

`station.yaml` profile `spu-merionc-2024` applies from 2024-09-10 onward and documents a 100 Hz Quantel MerionC laser. Its night SCC configuration includes `387.AN`, `387.PC`, `530.AN` and `530.PC` in addition to the elastic channels. The current profile now carries a `raman_detection` metadata block that records channel identity separately from unresolved spectral-response metadata.

The supplied `20250629sant_level1_rcs.nc` contains corrected 387 and 530 analog/photon-counting channels with useful block SNR through the P5.4 altitude range. See `docs/regression_baselines/p5_4_raman_companion_feasibility_20250629.json`.

## Evidence hierarchy

### Current-system identity evidence

Silva et al. (2025) describe the current SPU system as a 100 Hz multiwavelength Raman lidar with elastic detection at 355/532/1064 nm and Raman-shifted detection at 387 nm (nitrogen), 530 nm (nitrogen) and 408 nm (water vapor). The paper reports 7.5 m spatial resolution and complete overlap at 300 m a.g.l.

Source:

* Silva, G. M. da et al. (2025), *Long-Range Plume Transport from Brazilian Burnings to Urban São Paulo: A Remote Sensing Analysis*, Atmosphere 16(9), 1022. DOI: `10.3390/atmos16091022`.

This source supports **channel identity and current 100 Hz system context**. It does not publish the active MerionC effective filter response/passband needed for quantitative Raman inversion.

### Historical SPU receiver evidence

Earlier peer-reviewed SPU descriptions identify the same nominal Raman channels and describe 530 nm specifically as rotational Raman nitrogen:

* Moreira, G. de A. et al. (2019), *Analyzing the atmospheric boundary layer using high-order moments obtained from multiwavelength lidar data: impact of wavelength choice*, Atmospheric Measurement Techniques 12, 4261–4276.
* Pallotta, J. V.; de Carvalho, S. A.; Lopes, F. J. D. S.; Cacheffo, A.; Landulfo, E.; Barbosa, H. M. J. (2023), *Collaborative development of the Lidar Processing Pipeline (LPP) for retrievals of atmospheric aerosols and clouds*, Geoscientific Instrumentation, Methods and Data Systems 12, 171–185. DOI: `10.5194/gi-12-171-2023`.

The public SPU Lidar Station instruments page also reports historical interference-filter values near 387 and 530 nm. However, that page simultaneously reports receiver/FOV/overlap/resolution details that are not demonstrably identical to the post-2024 MerionC profile. Therefore those historical filter values are provenance evidence only and are **not** copied into active MerionC metadata.

## Current MerionC metadata status

| Detection channel | Elastic companion | Species / interpretation | SCC 1046 night exposure | Current spectral-response status |
| --- | --- | --- | --- | --- |
| 387 nm | 355 nm | N2 vibrational Raman | AN 4076 / PC 4077 | current instrument evidence required |
| 408 nm | 355 nm | H2O vibrational Raman | not exposed in SCC 1046 | current instrument evidence required |
| 530 nm | 532 nm | N2 rotational Raman | AN 4075 / PC 4074 | current instrument evidence required |

The 408-nm row records an explicit metadata discrepancy: the 2025 current-system description lists 408 nm, but the repository's current night SCC configuration 1046 does not expose a 408 channel. This may reflect SCC configuration scope rather than absent hardware; it must not be guessed.

`tests/test_spu_raman_metadata_contract.py` guards the current contract: while spectral-response status is unresolved, no numerical MerionC `passband_fwhm_nm` or `effective_detection_wavelength_nm` is allowed to appear silently in the profile.

## What is established

The following statements are currently defensible:

* 387 nm is a nitrogen Raman companion of 355 nm;
* 530 nm is a nitrogen rotational-Raman companion associated with 532 nm;
* 408 nm is a water-vapor Raman companion of 355 nm in published SPU system descriptions;
* current night SCC mapping exposes 387 and 530 AN/PC channels;
* corrected 387/530 Level-1 signals exist in the supplied 20250629 case and have useful high-altitude SNR;
* none of those facts alone establishes a quantitative Raman extinction/backscatter retrieval or validates `beta_aer(ref)=0`.

## Metadata still required before a Raman boundary constraint

Before MILGRAU uses 387/530 quantitatively to constrain aerosol extinction or elastic boundary residual aerosol, the active MerionC receiver path should establish:

* effective detection wavelength and passband / spectral transmission response;
* filter manufacturer/model or measured transmission curve and whether the receiver changed at the 2024 transition;
* relevant Raman cross-section and temperature dependence for the chosen channel physics;
* relative calibration and overlap behavior of Raman and elastic paths;
* treatment of rotational-Raman spectral response at 530 nm;
* any Ångström or wavelength-conversion assumption required to map Raman extinction to the emitted elastic wavelength;
* uncertainty propagation and controlled synthetic validation before real-data retrieval;
* independent comparison with a trusted Raman/optical chain when available.

Until these items are resolved, Raman-channel presence and SNR are **feasibility evidence only**. They must not be used as an undocumented cloud veto, molecular-purity flag, inferred boundary correction or productive uncertainty term.
