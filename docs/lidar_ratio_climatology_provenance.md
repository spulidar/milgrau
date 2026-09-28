# SPU monthly aerosol lidar-ratio provenance

## Scope

MILGRAU currently uses station-owned monthly aerosol lidar-ratio values for 355, 532 and 1064 nm. These values are a productive scientific assumption because elastic aerosol extinction is conditional on the assumed lidar ratio. This note records what is and is not currently known about their origin.

No numerical lidar-ratio value is changed by this document.

## Repository genealogy

The current values in `station.yaml` were migrated from the legacy MILGRAU `config.yaml`.

The exact same 36 monthly values are already present in the oldest `config.yaml` revision recoverable through the current repository history:

* commit `96c22196c73e1ce5733e21c3877f031d8da34d97`;
* commit date: 2026-03-31;
* commit message: `refatoração milgrau`.

That revision labels the values only as `Mean Aerosol Lidar Ratios (LR) per wavelength and month (01 to 12)`. No derivation script, dataset identifier, sample period, uncertainty calculation or literature citation accompanies the table in the tracked history.

Later legacy revisions preserve the same monthly numbers. The migration into `station.yaml` therefore preserved an existing empirical recipe, but did not establish its original scientific derivation.

## Published SPU context

Published São Paulo lidar work supports the **general scientific plausibility of seasonal and event-dependent lidar-ratio variability**, but the sources reviewed for this audit do not reproduce the 36 exact values used by MILGRAU.

Relevant context includes:

* Landulfo et al. (2003), *Synergetic measurements of aerosols over São Paulo, Brazil using LIDAR, sunphotometer and satellite data during the dry season*, Atmospheric Chemistry and Physics 3, 1523–1539, DOI `10.5194/acp-3-1523-2003`. The study derives a representative dry-season lidar ratio near 532 nm from lidar/sun-photometer synergy.
* Landulfo et al. (2008), *A Four-Year Lidar–Sun Photometer Aerosol Study at São Paulo, Brazil*, Journal of Atmospheric and Oceanic Technology 25, DOI `10.1175/2007JTECHA984.1`. The study reports temporal/seasonal lidar-ratio variability at 532 nm using coincident lidar and sun-photometer observations.
* Pallotta et al. (2023), *Collaborative development of the Lidar Processing Pipeline (LPP) for retrievals of atmospheric aerosols and clouds*, Geoscientific Instrumentation, Methods and Data Systems 12, 171–185, DOI `10.5194/gi-12-171-2023`. The work demonstrates SPU Raman retrieval capability and uses explicit wavelength-dependent lidar-ratio assumptions in validation examples.

These publications provide scientific context only. They must not be cited as the numerical source of the current monthly 355/532/1064 tables unless an original derivation establishes that link.

## Current provenance status

The defensible statement is:

> The monthly 355/532/1064 lidar-ratio table is a legacy SPU/MILGRAU empirical recipe, preserved unchanged from the earliest currently tracked configuration. Its original dataset, derivation method, sample period and uncertainty calculation have not yet been recovered.

Accordingly:

* the values remain usable as the **declared productive assumption** while their limitation is explicit;
* they are not independent validation truth;
* their monthly standard deviations in `station.yaml` must not be interpreted as demonstrated climatological sampling uncertainty without source evidence;
* a future recovered derivation should identify data source, period, screening, wavelength conversion (if any), estimator and uncertainty semantics;
* replacing the table or its uncertainties would be a scientific-method change requiring new provenance and validation, not a documentation cleanup.

## Evidence needed to close this gap

Any one of the following may recover the missing lineage if it contains the actual calculation behind the table:

* original notebook/script used to compute monthly values;
* spreadsheet or exported monthly statistics with source dataset identifiers;
* thesis/article/supplement containing the exact table;
* archived AERONET/SCC/Raman processing output plus the documented aggregation procedure;
* an earlier repository or workstation snapshot predating the current Git history.

Until then, the provenance status is `legacy_empirical_recipe_scientific_derivation_unresolved`.
