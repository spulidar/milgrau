# Level 1 scientific validation workbook — historical SPU observations

This workbook defines the evidence needed to complete the LIPANCORA scientific validation beyond automated tests. It provides reproducible, **non-destructive** comparisons and does **not** claim to have executed processing on historical SPU-Lidar observations.

## Priority acquisition samples

The smallest useful first batch is **one complete continuous session with its raw Licel files**, a dated station profile, and its associated dark acquisitions if any. To assess historical behavior, repeat later for the four catalog eras:

| Instrument era | Catalog interval | Preferred scientific scenario |
| --- | --- | --- |
| Legacy | Before 2017-03-08 | Missing or provisional channel calibration; possible no SCC support |
| APEL | 2017-03-08 to 2018-03-18 | Analog and photon-counting signal agreement, with associated dark |
| Raman | 2018-03-19 to 2024-09-09 | Mixed channels, far-range background, and a day/night transition |
| MerionC | From 2024-09-10 | Current 100 Hz laser profile, high rates, calibration/saturation flags |

Select real sessions by available original evidence; the examples in other guides are illustrative and **must not be assumed to exist**. For first acceptance, one well-documented session is better than an unreviewed large archive.

## Required input package

A representative experiment benefits from:

1. **Original Licel files** covering a complete continuous session, including acquisition start/stop times and all channels.
2. **Associated dark-current acquisitions**, or confirmation that none were taken. Do not substitute invented or arbitrary dark data.
3. The exact **`station.yaml`** and **`config.yaml`** used for that processing; provide the actual historical calibration ID, not just the current defaults.
4. Any preserved **`SESSION_L0.nc` and `SESSION_L1.nc`** from the earlier processing version, plus their software revision and SHA-256 if available.
5. Optional but useful: a session log, quicklook and known issue/warning observations. External ERA5/IGRA2 archives are only needed if we will analyze the atmospheric fallback; do not send authentication credentials.

An existing previously generated L0/L1 without the raw Licel input is sufficient to inspect **stored outputs and provenance**, but **not** enough to prove parser/processing reproducibility from original acquisitions.

## Run plan (use a disposable data root)

1. Record hashes of all original raw files. Keep them read-only and never run destructive quarantine on them.
2. Use separate baseline and candidate output roots; do not let `--force` overwrite earlier scientific evidence.
3. Run LIBIDS with the appropriate dated station catalog and Level 0 config; inspect L0 QA, channels, dark-current sources and solar labels.
4. Run LIPANCORA on the same canonical session and record the exact calibration and atmosphere selection policy.
5. Evaluate the raw versus pre-background and fully corrected signal, 1σ uncertainty, clipping and saturation characterization, background quality, PBL availability, and actual thermodynamic source by atmosphere hour.
6. Compare baseline and candidate on matched native channels, time/altitude grids and source conditions; explain each nontrivial difference.
7. Run optional alternate atmosphere modes **only after preserving the first candidate output**. Do not compare different ERA5/radiosonde/USSA76 settings as if the molecular science were unchanged.
8. Inspect LEBEAR only after Level 1 is accepted, keeping the original L1 identity and quality flags in the comparison.

Suggested operator commands:

~~~bash
milgrau-libids -i SESSION_ID
milgrau-lipancora -i SESSION_ID
milgrau-inspect SESSION_ID --validate
milgrau-inspect SESSION_ID --full
milgrau-lipancora --figures-only -i SESSION_ID
~~~

`milgrau-inspect --validate` checks stored product structure; it does not perform a physical detector calibration or prove that a weak-signal aerosol profile is valid.

## Scientific QA worksheet

| Check | Required evidence | Status |
| --- | --- | --- |
| Source integrity | Raw Licel list, counts, SHA-256, exact acquisition interval | Pending real data |
| Instrument calibration | Historical profile, channel-specific detector properties and unknown flags | Pending real data |
| Native grid handling | Original channel resolution and common-grid interpolation outcome | Pending real data |
| Dark current | Selected acquisitions, distance, means and standard errors | Pending real data |
| Analog uncertainty | Native corrected units, robust scale, dark contribution | Pending real data |
| PC observed count rate | Shots, native bin-time, maximum observed MHz | Pending real data |
| Dead-time clipping | Numerical denominator floor and before/after raw diagnostic | Pending real data |
| Physical saturation | Characterized rate limit or explicit *uncharacterized* flag | Pending instrument evidence |
| Bin shifts | Configured integer shifts and masked edge fractions | Pending real data |
| Far-range background | Selected physical altitude support, MAD, valid bins and outliers | Pending real data |
| RCS uncertainty | Signal uncertainty, correlated background location uncertainty, altitude range | Pending real data |
| PBL and tropopause | Available source/algorithm and supported periods | Pending real data |
| Molecular atmosphere | ERA5/Wyoming/IGRA2/USSA76 by hour, sounding age and vertical extension | Pending real data |
| Product continuity | Session, source L0, L1 hashes, warnings, accepted/failed channels | Pending real data |
| Scientific equivalence | Matched baseline/candidate numerical differences, reason and reviewer | Pending review |

No row should be marked complete solely because all synthetic CI checks passed.

## Quantities and formula checks

For a photon-counting native range bin of width \(\Delta r\), the bin-time is

$$
\\Delta t_{\\mu s}=\\frac{2\\Delta r}{c}\\times10^6.
$$

The observed rate is \(N/(N_{\\rm shots}\\Delta t_{\\mu s})\); this rate **precedes dark-current subtraction**. For a channel with calibrated nonparalyzable dead time \(\tau\), the implemented correction is the dark-subtracted rate divided by \(1-\\tau q\), subject to its numerical denominator floor.

The robust far-range background uses a per-profile median and Gaussian-consistent \(1.4826\\,\\mathrm{MAD}\). The 1σ location uncertainty uses the Gaussian median approximation and is **correlated across altitude within one channel/profile**. Examine whether the chosen far-range window is physically free of relevant atmospheric returns for the particular observation.

The output RCS is the corrected signal times \(r^2\); it is not a direct aerosol backscatter coefficient. A finite RCS at a given altitude does not establish that a two-sided KFS retrieval can be performed there.

## Report template

Record: exact session ID; source acquisition timestamps and hashes; station profile/calibration; instrument/channels; dark-current data; applied config and software SHA; weather/atmosphere source policy; L0 and L1 hashes; diagnostic plots and fields; profile-support limits; differences versus baseline with numerical tolerances; remaining uncertainty; reviewer and review date.

**Acceptance:** only after scientific review of real evidence, not an assumed successful run. See [Level 1 variable dictionary](level1_schema.md), [Scientific quality](scientific_quality.md) and [Level 0 workbook](level0_validation_workbook.md).
