# Offline high-column evidence export (P5.4)

This command joins the existing schema-3/method-4 candidate catalogue to
`HighColumnEvidence`. It does not run KFS, change candidate selection, rank
high-altitude candidates or create a composite score. It exports every slot,
including rejected and unevaluated candidates, so missingness and the original
productive choice remain auditable.

## Frozen observational case

The repository contains aggregate regression summaries, not the original
`20251107sapm_level2_optical.nc`. Those summaries cannot reconstruct individual
candidate records. Run against the original product when it is available:

```bash
python -m milgrau.cli.high_column_evidence \
  /path/to/20251107sapm_level2_optical.nc \
  /path/to/new_evidence_directory \
  --expected-sha256 32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212 \
  --block-weights 23 39 39 40 26 \
  --weight-basis profile_count_per_configured_20min_floor_block \
  --effective-vertical-resolution-m 7.5 \
  --resolution-basis native_7.5m_sampling_no_additional_aggregation_not_instrument_resolution_validation \
  --altitude-edges-m 0 5000 10000 15000 20000 25000 30000
```

Weights and block ordering come from
`regression_baselines/20251107sapm_temporal_support_preliminary.json`.
The resolution declaration describes the native processing grid, not validated
instrument optical resolution. There is no automatic inference of effective
resolution from coordinate spacing. Review these inputs for every other case.

The output directory must be new. The command creates `candidates.csv`,
`candidates.json` and `summary.json`; it records the input checksum, available
product provenance, exporter source identity, block times, explicit weights,
resolution declaration and diagnostic definitions. Keep all three files together.
JSON uses `null`, CSV an empty cell, for unavailable numerical diagnostics.

## Meaning and limitations

| Dimension | Definition |
| --- | --- |
| Shape QA / bin-wise SNR | Original catalogue values; no recomputed selection |
| Window SNR | Existing origin-calibration uncertainty helper; independent and fully correlated limits separately |
| Temporal persistence | Weighted acceptance of the **same window** across all blocks; unknown if any block did not evaluate it |
| Temporal evaluated fraction | Fraction of declared weights with that candidate evaluated |
| Dominant contribution | Maximum `weight * abs(RCS)` fraction at the exact center on finite signal/error support |
| Center support fraction | Supported block weight fraction at the exact center |
| Subwindow disagreement | `abs(C_lower - C_upper) / C_full` for disjoint contiguous positional halves; odd bin goes to upper half |
| Contamination fraction | Unknown: no validated detector supplied |
| Empirical covariance SNR | Unknown: no covariance model supplied |

The existing altitude-persistence helper asks whether **any** accepted window
exists above a target. That is a different quantity from exact-window persistence
here; they must not be interchanged. Dominance and persistence repeat across
rows because they describe the multi-block state, not independent block evidence.

The calibration helper uses finite positive measured/molecular signal and
strictly positive error, with at least two valid samples. Exported counts and
half-window factors expose that support. A factor recomputed on uncertainty
support can differ from the catalogue's signal-only calibration. No missing
error is replaced with zero. Unevaluated slots retain unknown fitted diagnostics.

Agreement between halves cannot establish molecular purity: broad contamination
can bias both halves together. An SNR gain is conditional on dependence, and
positive-signal conditioning can matter in weak-signal tails. Neither SNR nor
disagreement authorizes a new boundary. No contamination, covariance or accepted
retrieval-support claim is inferred from these fields.

Summary bands are `[lower, upper)` and are descriptive, not acceptance gates.
Summaries split accepted/rejected/unevaluated slots and report finite/missing
counts and quartiles. Overlapping candidates are not independent observations;
these distributions are not confidence intervals or ground truth.

## Exit condition

1. Run on the checksum-matched frozen product and retain candidate tables.
2. Reconcile the 25,340 slots, 16,224 accepted slots and original selected
   references with the frozen catalogue summary.
3. Inspect altitude distributions, missingness and distinct failure modes.
4. Strengthen contamination/noise/placement synthetics before choosing boundaries
   for a non-productive lower-column preservation experiment.

Until steps 1–3 are completed, observational P5.4 remains open. Synthetic
export tests establish implementation behavior, not validation of the SPU event.
