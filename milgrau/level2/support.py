"""Altitude-resolved scientific support semantics for Level 2 retrievals.

The low-level contract in this module is intentionally independent of NetCDF.
It distinguishes algorithmic/inversion support from a future stricter support
claim that will additionally require an evidence-backed instrument mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class BackwardRetrievalSupport:
    """Contiguous supported domain for one accepted backward retrieval.

    ``flag`` is true only on the contiguous interval that reaches the accepted
    upper inversion/reference boundary.  A missing/invalid bin inside that
    interval breaks the backward integration path; valid-looking values below
    such a gap are therefore not promoted to scientifically supported output.
    """

    flag: np.ndarray
    bottom_altitude_m: float
    top_altitude_m: float


@dataclass(frozen=True, slots=True)
class Level2InversionSupport:
    """Block and aggregate algorithmic support for processed wavelengths.

    This contract deliberately says *inversion* support.  No lower instrument
    mask is applied while overlap/near-field validity remains uncharacterized.
    ``effective_block_count`` reports how many successful block retrievals
    support each aggregate altitude bin, so a high top supported by one block
    is not confused with support shared by all blocks.
    """

    flag: np.ndarray
    flag_block: np.ndarray
    effective_block_count: np.ndarray
    bottom_altitude_m: np.ndarray
    top_altitude_m: np.ndarray
    bottom_altitude_m_block: np.ndarray
    top_altitude_m_block: np.ndarray


def backward_retrieval_support(
    altitude_m: np.ndarray,
    value: np.ndarray,
    uncertainty: np.ndarray,
    *,
    upper_index: int,
    instrument_valid: np.ndarray | None = None,
) -> BackwardRetrievalSupport:
    """Return scientifically supported bins for a backward retrieval.

    Support requires, at every retained bin:

    - altitude on a finite strictly increasing one-dimensional grid;
    - finite retrieved value;
    - finite, non-negative reported uncertainty;
    - validated instrument support when an ``instrument_valid`` mask is given;
    - membership in the contiguous valid path ending at ``upper_index``.

    Bins above ``upper_index`` are outside the productive backward inversion
    even when numerically finite.  An invalid bin below ``upper_index`` cuts
    support there and also invalidates every lower bin for that branch because
    the backward integral cannot jump across an unsupported gap.

    ``instrument_valid=None`` means that this function has not been asked to
    impose an additional instrument boundary; it must not be interpreted as a
    claim that overlap or any other instrument limitation is characterized.
    """
    altitude = np.asarray(altitude_m, dtype=np.float64)
    retrieved = np.asarray(value, dtype=np.float64)
    sigma = np.asarray(uncertainty, dtype=np.float64)

    if altitude.ndim != 1 or retrieved.ndim != 1 or sigma.ndim != 1:
        raise ValueError("altitude, value, and uncertainty must be one-dimensional.")
    if not (altitude.shape == retrieved.shape == sigma.shape):
        raise ValueError("altitude, value, and uncertainty must have identical shapes.")
    if altitude.size == 0:
        raise ValueError("support cannot be evaluated on an empty altitude grid.")
    if not np.all(np.isfinite(altitude)) or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("altitude must be finite and strictly increasing.")
    if isinstance(upper_index, bool) or not isinstance(upper_index, (int, np.integer)):
        raise ValueError("upper_index must be an integer index.")
    upper = int(upper_index)
    if upper < 0 or upper >= altitude.size:
        raise ValueError("upper_index is outside the altitude grid.")

    if instrument_valid is None:
        instrument = np.ones(altitude.shape, dtype=bool)
    else:
        instrument = np.asarray(instrument_valid)
        if instrument.ndim != 1 or instrument.shape != altitude.shape:
            raise ValueError("instrument_valid must be one-dimensional and match altitude.")
        if instrument.dtype.kind != "b":
            raise ValueError("instrument_valid must be a boolean mask.")
        instrument = instrument.astype(bool, copy=False)

    candidate = np.isfinite(retrieved) & np.isfinite(sigma) & (sigma >= 0.0) & instrument
    flag = np.zeros(altitude.shape, dtype=bool)

    if not candidate[upper]:
        return BackwardRetrievalSupport(
            flag=flag,
            bottom_altitude_m=float("nan"),
            top_altitude_m=float("nan"),
        )

    bottom = upper
    while bottom > 0 and candidate[bottom - 1]:
        bottom -= 1

    flag[bottom : upper + 1] = True
    return BackwardRetrievalSupport(
        flag=flag,
        bottom_altitude_m=float(altitude[bottom]),
        top_altitude_m=float(altitude[upper]),
    )


def _exact_reference_index(altitude_m: np.ndarray, reference_altitude_m: float) -> int:
    """Resolve a stored reference altitude back to its exact lidar-grid bin."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    reference = float(reference_altitude_m)
    if not np.isfinite(reference):
        raise ValueError("reference altitude must be finite.")
    spacing = float(np.min(np.diff(altitude))) if altitude.size > 1 else 1.0
    atol = max(abs(spacing) * 1.0e-9, 1.0e-9)
    matches = np.flatnonzero(np.isclose(altitude, reference, rtol=0.0, atol=atol))
    if matches.size != 1:
        raise ValueError(
            "Rayleigh reference altitude must identify exactly one altitude-grid bin."
        )
    return int(matches[0])


def _common_optical_support_inputs(
    backscatter: np.ndarray,
    backscatter_error: np.ndarray,
    extinction: np.ndarray,
    extinction_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return value/error arrays masked to common optical-product support."""
    beta = np.asarray(backscatter, dtype=np.float64)
    beta_error = np.asarray(backscatter_error, dtype=np.float64)
    alpha = np.asarray(extinction, dtype=np.float64)
    alpha_error = np.asarray(extinction_error, dtype=np.float64)
    if not (beta.shape == beta_error.shape == alpha.shape == alpha_error.shape):
        raise ValueError("Optical value/error arrays must have identical shapes.")
    common = (
        np.isfinite(beta)
        & np.isfinite(beta_error)
        & (beta_error >= 0.0)
        & np.isfinite(alpha)
        & np.isfinite(alpha_error)
        & (alpha_error >= 0.0)
    )
    return np.where(common, beta, np.nan), np.where(common, beta_error, np.nan)


def assemble_level2_inversion_support(
    results: Sequence[Any],
    altitude_m: np.ndarray,
) -> Level2InversionSupport:
    """Assemble block and aggregate backward inversion support.

    ``results`` must be sorted in the same wavelength order used by the Level 2
    dataset.  Each successful block is evaluated at its own exact accepted
    Rayleigh boundary.  Aggregate support is the contiguous union represented
    by the aggregate optical value/error fields, while
    ``effective_block_count`` preserves how many block solutions support each
    altitude.

    No instrument-validity mask is supplied here.  Therefore these fields are
    an algorithmic/inversion-domain diagnostic and are not a validated
    near-field overlap/support claim.
    """
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if altitude.ndim != 1 or altitude.size == 0:
        raise ValueError("altitude_m must be a non-empty one-dimensional grid.")
    if not np.all(np.isfinite(altitude)) or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("altitude_m must be finite and strictly increasing.")
    if not results:
        raise ValueError("At least one wavelength result is required.")

    n_wavelength = len(results)
    n_block = int(np.asarray(results[0].optical.retrieval_success_flag).size)
    if n_block <= 0:
        raise ValueError("At least one retrieval block is required.")

    flag_block = np.zeros(
        (n_block, n_wavelength, altitude.size), dtype=np.int8
    )
    bottom_block = np.full((n_block, n_wavelength), np.nan, dtype=np.float64)
    top_block = np.full((n_block, n_wavelength), np.nan, dtype=np.float64)

    for wavelength_index, result in enumerate(results):
        success = np.asarray(result.optical.retrieval_success_flag, dtype=np.int8)
        reference_altitude = np.asarray(
            result.rayleigh.reference_altitude_m_block, dtype=np.float64
        )
        beta = np.asarray(result.optical.aerosol_backscatter_block, dtype=np.float64)
        beta_error = np.asarray(
            result.optical.aerosol_backscatter_error_block, dtype=np.float64
        )
        alpha = np.asarray(result.optical.aerosol_extinction_block, dtype=np.float64)
        alpha_error = np.asarray(
            result.optical.aerosol_extinction_error_block, dtype=np.float64
        )
        expected_shape = (n_block, altitude.size)
        if success.shape != (n_block,) or reference_altitude.shape != (n_block,):
            raise ValueError("Block success/reference arrays do not share the common block axis.")
        if not (
            beta.shape
            == beta_error.shape
            == alpha.shape
            == alpha_error.shape
            == expected_shape
        ):
            raise ValueError("Block optical arrays do not match block/altitude dimensions.")

        for block_index in range(n_block):
            if int(success[block_index]) != 1:
                continue
            reference_index = _exact_reference_index(
                altitude, reference_altitude[block_index]
            )
            value, uncertainty = _common_optical_support_inputs(
                beta[block_index],
                beta_error[block_index],
                alpha[block_index],
                alpha_error[block_index],
            )
            support = backward_retrieval_support(
                altitude,
                value,
                uncertainty,
                upper_index=reference_index,
            )
            flag_block[block_index, wavelength_index] = support.flag.astype(np.int8)
            bottom_block[block_index, wavelength_index] = support.bottom_altitude_m
            top_block[block_index, wavelength_index] = support.top_altitude_m

    effective_count = flag_block.sum(axis=0, dtype=np.int16)
    aggregate_flag = np.zeros((n_wavelength, altitude.size), dtype=np.int8)
    aggregate_bottom = np.full(n_wavelength, np.nan, dtype=np.float64)
    aggregate_top = np.full(n_wavelength, np.nan, dtype=np.float64)

    for wavelength_index, result in enumerate(results):
        beta, beta_error = _common_optical_support_inputs(
            result.optical.aerosol_backscatter,
            result.optical.aerosol_backscatter_error,
            result.optical.aerosol_extinction,
            result.optical.aerosol_extinction_error,
        )
        has_block_support = effective_count[wavelength_index] > 0
        common = np.isfinite(beta) & np.isfinite(beta_error) & has_block_support
        if not np.any(common):
            continue
        upper_index = int(np.flatnonzero(common)[-1])
        masked_beta = np.where(has_block_support, beta, np.nan)
        masked_error = np.where(has_block_support, beta_error, np.nan)
        support = backward_retrieval_support(
            altitude,
            masked_beta,
            masked_error,
            upper_index=upper_index,
        )
        aggregate_flag[wavelength_index] = support.flag.astype(np.int8)
        aggregate_bottom[wavelength_index] = support.bottom_altitude_m
        aggregate_top[wavelength_index] = support.top_altitude_m

    return Level2InversionSupport(
        flag=aggregate_flag,
        flag_block=flag_block,
        effective_block_count=effective_count,
        bottom_altitude_m=aggregate_bottom,
        top_altitude_m=aggregate_top,
        bottom_altitude_m_block=bottom_block,
        top_altitude_m_block=top_block,
    )
