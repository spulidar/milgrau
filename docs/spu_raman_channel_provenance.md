# SPU Raman companion-channel provenance

## Purpose

This note records published semantic evidence for SPU non-elastic detection channels relevant to future MILGRAU boundary-condition validation. It does not create a Raman Level-2 product and does not change productive elastic retrieval method v4.

## Current repository state

`station.yaml` profile `spu-merionc-2024` applies from 2024-09-10 onward and documents a 100 Hz Quantel MerionC laser. Its night SCC configuration includes `387.AN`, `387.PC`, `530.AN` and `530.PC` in addition to the elastic channels. The general SCC policy also records Raman companion wavelengths, but the repository does not yet encode scattering species, effective filter response or Raman retrieval physics.

The supplied `20250629sant_level1_rcs.nc` contains corrected 387 and 530 analog/photon-counting channels with useful block SNR through the P5.4 altitude range. See `docs/regression_baselines/p5_4_raman_companion_feasibility_20250629.json`.

## Published channel identity

A 2025 SPU study describing the current 100 Hz system states that the station uses six detection wavelengths: the elastic 355, 532 and 1064 nm channels and Raman-shifted channels at 387 nm (nitrogen), 530 nm (nitrogen) and 408 nm (water vapor).

Source:

* Silva, G. M. da et al. (2025), *Long-Range Plume Transport from Brazilian Burnings to Urban São Paulo: A Remote Sensing Analysis*, Atmosphere 16(9), 1022. DOI: `10.3390/atmos16091022`.

Earlier peer-reviewed SPU instrumentation descriptions provide the same channel identities and describe 530 nm specifically as rotational Raman scattering from nitrogen:

* Moreira, G. de A. et al. (2019), *Analyzing the atmospheric boundary layer using high-order moments obtained from multiwavelength lidar data: impact of wavelength choice*, Atmospheric Measurement Techniques 12, 4261–4276. The paper describes 387 nm as the nitrogen-shifted companion of 355 nm, 408 nm as the water-vapor-shifted companion of 355 nm, and 530 nm as rotational Raman shifting from 532 nm by nitrogen.
* Freudenthaler, V. et al. / LPP collaboration (2023), *Collaborative development of the Lidar Processing Pipeline (LPP) for retrievals of atmospheric aerosols and clouds*, Geoscientific Instrumentation, Methods and Data Systems 12, 171–203, likewise describes SPU detection at elastic 355/532/1064 nm plus nitrogen Raman 387/530 nm and water-vapor 408 nm.

The SPU Lidar Station instrument page also publishes the same species assignments and historical interference-filter bandwidths. Those filter values are useful provenance evidence but are **not copied into current MerionC metadata here**, because the repository does not yet demonstrate that the receiver/filter assembly and effective spectral response were unchanged across the 2024 instrument-profile transition.

## What is established

For R&D planning, the following channel identities have adequate published support:

| Detection channel | Emitted companion | Published interpretation | Current use in MILGRAU |
| --- | --- | --- | --- |
| 387 nm | 355 nm | Raman-shifted nitrogen | Level-1 signal retained; no productive Raman retrieval |
| 408 nm | 355 nm | Raman-shifted water vapor | Level-1 signal may exist; not part of current MerionC SCC night mapping |
| 530 nm | 532 nm | rotational Raman nitrogen | Level-1 signal retained; no productive Raman retrieval |

These identities do not by themselves establish quantitative Raman extinction/backscatter retrieval semantics.

## Metadata still required before a Raman boundary constraint

Before MILGRAU uses 387/530 quantitatively to constrain `beta_aer(ref)` or aerosol extinction, the current instrument profile should explicitly establish:

* effective detection wavelength and passband for the active MerionC receiver path;
* filter/transmission response and whether it changed at the 2024 transition;
* relevant Raman cross-section / temperature dependence and molecular species treatment;
* relative calibration and overlap behavior of the Raman and elastic paths;
* any Ångström or other wavelength relationship required to convert Raman extinction to the emitted wavelength;
* uncertainty propagation and a controlled synthetic plus external-chain validation.

Until those items are resolved, Raman-channel presence and SNR are feasibility evidence only. They must not be used as an undocumented cloud veto, molecular-purity flag or productive boundary correction.
