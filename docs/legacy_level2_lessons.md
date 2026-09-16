# Legacy Level 2 audit: scientific lessons for the high-column redesign

This note records a scientific audit of a user-supplied historical MILGRAU archive. The archive is evidence about earlier behavior and design ideas; it is **not** a ground-truth reference implementation and must not be used as a golden numerical target for the current code.

Archive SHA-256: `b40895501375566d78c9051100d257b94a8ad1af3dc3f3f8120e6665f3271b29`

Representative historical case: `20241219nt` / 20 Dec 2024 05:21:13--06:14:39 UTC.

## Provenance limitation

The source files and saved products in the archive are not a perfectly synchronized software snapshot. For example, the included root `06-LEBEAR.py` currently says `glueflag = "yes"`, while the saved KFS plot is labelled `355 nm PC`, and the saved optical CSV name is not the filename emitted by that exact script state. Therefore the audit distinguishes source inspection from numerical reconstruction.

Despite that limitation, the saved 355 nm backscatter profile can be reproduced very closely from the archived mean 355.PC RCS using the archived KFS implementation, the December 355 nm lidar ratio, the archived U.S. Standard Atmosphere path, and the legacy post-inversion Savitzky--Golay smoothing. The reconstruction has Pearson correlation `0.999958`, RMSE `4.84e-8 m-1 sr-1`, and relative L2 difference about `1.09%` against the saved profile. This is strong behavior evidence for the historical retrieval mechanism, not proof that the archived script and product came from the same exact revision.

The archive does not contain a processed SCC/ELDA optical product for this case, so the historical statement that this behavior compared well with SCC cannot be independently quantified from this bundle alone.

## What made the legacy inversion appear to reach high altitude

The legacy path combined several mechanisms:

1. **Whole-measurement averaging before inversion.** The archived case contains 105 nominal 30 s profiles over about 53.4 min. `06-LEBEAR.py` concatenates all profiles and computes one altitude-wise mean before molecular fitting and KFS. This is an early form of the high-column/backbone concept.
2. **A nominal 15 km KFS reference.** With `ini_molref_alt = 5000 m` and `fin_molref_alt = 25000 m`, the code chooses the midpoint, approximately 15 km, as `index_reference`.
3. **A very broad reference-signal estimator.** `_get_reference_values()` smooths and takes the median of RCS over approximately 5--25 km, while molecular backscatter is taken at the exact 15 km reference bin.
4. **Two-sided KFS.** The archived KFS integrates backward below the reference and forward above it, so the returned array is numerically populated toward 30 km when the denominator remains finite.
5. **Post-inversion smoothing.** The saved optical product is smoothed with a 15-bin, third-order Savitzky--Golay filter, corresponding to about 112.5 m on the 7.5 m grid.

These choices explain why the historical plot extends to 30 km. They do **not** establish 30 km scientific support.

## What is scientifically useful and should be retained as an idea

### 1. Long-mean / high-column backbone

The strongest reusable idea is averaging more profiles before attempting a high reference. This belongs in the current P5 backbone design, but with uncertainty and temporal-support bookkeeping that the legacy code did not have.

The archived case also shows why a backbone must not simply mean "average everything". Using the current Rayleigh shape-QA concepts on the archived 355.PC profiles with an exploratory temporal standard error gives very different 20 min behavior:

- 05:20 block, 37 profiles: high-altitude molecular-like candidates exist well above 15 km and above 20 km;
- 05:40 block, 40 profiles: accepted shape-QA candidates stop near 6.4 km;
- 06:00 block, 28 profiles: accepted shape-QA candidates stop near 5.5 km.

The whole 53 min mean again produces many formal high-altitude candidates because the early block dominates the high-altitude signal. At 15 km about 97% of the signed whole-mean signal sum comes from the first 20 min block. Therefore a modern backbone needs explicit **temporal support/stability diagnostics**: contributing profiles/blocks, time span, and evidence that high-altitude support is coherent over the interval represented by the product.

### 2. Robust information from a broad molecular region

The legacy 5--25 km median was an implicit attempt to stabilize a noisy reference and reduce sensitivity to one bin or a local layer. That robustness idea is useful, but the implementation must not be copied.

The modern replacement should be:

- catalogue many narrow physical Rayleigh windows;
- reject contaminated/low-quality windows first;
- require sufficient separation before treating windows as distinct;
- form a robust calibration/reference consensus or an explicit multi-reference ensemble from accepted windows;
- keep every KFS member tied to its own exact physical boundary.

This preserves the legacy robustness objective without mixing one non-local measured RCS median with molecular backscatter from a different exact altitude.

### 3. Two-sided KFS as a research diagnostic

The archived KFS implementation contains both backward and forward integration. The current MILGRAU already has independently tested forward/two-sided research kernels, so the useful legacy lesson is not to restore forward retrieval as the productive default, but to retain it as a diagnostic/sensitivity tool.

In the saved legacy optical profile, backscatter above the 15 km reference is predominantly negative: about 66% of bins from 15--20 km, 81% from 20--25 km, and 94% from 25--30 km. Numerical finiteness therefore dramatically overstates physical usefulness in this example. Forward/two-sided output must never be counted as high-column support merely because the array is finite.

### 4. Scattering-ratio / molecular-fit visualization as QA

The legacy workflow explicitly compared measured signal with a scaled molecular signal and plotted scattering ratio. The current candidate catalogue already uses measured/molecular ratio slope and variance; useful extensions from the old visualization idea are robust residual/bias diagnostics and high-column plots that show accepted/rejected molecular-fit regions together with uncertainty. Scattering ratio above a backward boundary remains a diagnostic, not retrieved aerosol.

### 5. Raman retrieval as an independent validation direction

The archive contains generic Raman extinction/backscatter functions and a `LIRABEAR` experiment. The file itself warns that those functions were not tested, and the historical script does not provide enough instrument metadata here to validate the hard-coded emission/detection pairing. The old implementation should therefore **not** be ported as productive code.

The scientific direction remains valuable: current SPU Level 1 contains inelastic channels, so a future, separately validated nighttime Raman path could provide independent extinction/backscatter or lidar-ratio evidence for elastic KFS validation. That would be a validation/R&D project with explicit channel physics, uncertainty and synthetic tests, not a shortcut for the current high-column elastic redesign.

## What must not be copied from the legacy retrieval

### Non-local boundary inconsistency

The legacy KFS accepts `beta_aerosol_reference = 0`, but its signal reference is the smoothed median over roughly 5--25 km while `beta_molecular_reference` is the molecular value at exactly 15 km. In the archived case the broad RCS reference is about `333049` while the exact 15 km RCS is about `400069` (ratio `0.8325`).

As a result, the unsmoothed legacy inversion at the nominal reference does **not** satisfy the configured zero-aerosol boundary: reconstructed aerosol backscatter at 15 km is about `2.33e-7 m-1 sr-1`, roughly 20% of molecular backscatter there. Post-inversion smoothing changes the boundary again. The current exact-bin `beta_total_ref` contract is scientifically preferable and must be preserved.

### Whole-column free linear fit as calibration

The legacy molecular fit performs a free `np.polyfit` over 5--25 km, a region that in this case contains strong structure near 11--12 km and noisy far range. For the archived 355.PC mean, the broad fit has only about `R^2 = 0.48`; an origin-constrained factor differs from the free-fit slope by about 10.6%. The modern code correctly treats the free intercept as a diagnostic and uses an origin-constrained molecular calibration factor. Keep that policy.

### Finite output as support

The legacy KFS returns a value through most/all of the 30 km grid and then applies smoothing. The archived real output demonstrates why current MILGRAU must keep `isfinite(product) != supported(product)`. Unsupported/noisy forward bins can be finite, negative and visually easy to overlook.

### Post-retrieval smoothing as scientific support

Savitzky--Golay smoothing may remain useful for visualization or explicitly defined derived products, but it must not create scientific support, repair an invalid boundary, bridge gaps or hide uncertainty. Productive support is determined before display smoothing.

## Molecular lidar-ratio consistency discovered during the audit

The legacy helper library contains a wavelength-dependent molecular lidar-ratio calculation, even though the historical LEBEAR path hard-coded `8*pi/3`. This exposed a current consistency question worth resolving before high-column work is declared complete.

Current MILGRAU computes molecular extinction and 180-degree molecular backscatter with the depolarization-aware Bucholtz/Rayleigh phase function, but productive KFS defaults to the constant `8*pi/3 = 8.37758 sr`. From the same phase-function model, `alpha_mol / beta_mol` is approximately:

- 355 nm: `8.50366 sr`;
- 532 nm: `8.49663 sr`.

The difference is about 1.5% and 1.4%, respectively. This is not a reason to silently replace the current constant: the physically correct KFS molecular lidar ratio also depends on whether the detected elastic molecular component should be treated as total Rayleigh or Cabannes for the actual receiver/filter response. The required action is an explicit **molecular-model / detected-component consistency audit**, followed by a versioned method change if needed.

## Concrete implications for current P5

1. Keep the long-mean backbone objective, but add temporal support/stability and do not assume the entire measurement belongs in one backbone.
2. Treat the legacy broad 5--25 km reference as motivation for a robust consensus/ensemble of separated accepted Rayleigh windows, not as a boundary formula.
3. Keep productive retrieval backward-only until real evidence supports another contract; forward/two-sided remains a research diagnostic.
4. Add the legacy `20241219nt` case to the observational stress-test matrix because it contains a transient high-altitude feature and strongly time-dependent far-range support.
5. Audit molecular lidar-ratio consistency (`alpha_mol / beta_mol`, wavelength dependence, total-vs-Cabannes receiver semantics) before final high-column validation.
6. Preserve Raman retrieval as a future independent-validation direction, but do not port the untested legacy implementation.

Detailed machine-readable derived metrics are stored in `docs/regression_baselines/20241219nt_legacy_level2_audit.json`.
