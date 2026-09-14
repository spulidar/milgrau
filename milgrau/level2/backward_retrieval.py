"""Backward-only productive optical retrieval policy.

The legacy retrieval kernel can calculate backward, forward, or two-sided
Fernald branches, but its aggregate-success contract historically required both
branches. The productive elastic aerosol product now follows the high-reference
backward Klett--Fernald convention. This adapter preserves all per-block
diagnostics from the numerical kernel and re-aggregates optical products using
only Rayleigh-valid blocks with a valid backward branch.

The forward branch remains available in the numerical kernel for research and
diagnostics; it is not required for the productive aerosol product.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Callable, Mapping

import numpy as np

from milgrau.level2.block_average import error_of_mean, nanmean_or_nan
from milgrau.level2.contracts import (
    KfsDiagnostics,
    MolecularProfiles,
    OpticalProducts,
    RayleighDiagnostics,
)


class _SuppressLegacyTwoSidedWarning:
    """Delegate logging while suppressing the obsolete two-sided failure line."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        if "has no valid retrieval block. Mean optical products set to NaN." in str(message):
            return
        self._logger.warning(message, *args, **kwargs)


def _valid_block_mean(values: np.ndarray, valid_block: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if valid.any():
        return nanmean_or_nan(values[valid, :], axis=0)
    return np.full(values.shape[-1], np.nan, dtype=np.float64)


def _valid_block_error(values: np.ndarray, valid_block: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if valid.any():
        return error_of_mean(values[valid, :])
    return np.full(values.shape[-1], np.nan, dtype=np.float64)


def _reaggregate_backward_optical_products(
    optical: OpticalProducts,
    rayleigh: RayleighDiagnostics,
    kfs: KfsDiagnostics,
) -> tuple[OpticalProducts, np.ndarray]:
    """Aggregate only blocks accepted by Rayleigh QA and backward KFS."""
    valid_block = (
        (np.asarray(rayleigh.reference_success_flag_block) == 1)
        & (np.asarray(kfs.backward_valid_flag_block) == 1)
    )
    updated = replace(
        optical,
        scattering_ratio_mean=_valid_block_mean(
            optical.scattering_ratio_block, valid_block
        ),
        aerosol_backscatter=_valid_block_mean(
            optical.aerosol_backscatter_block, valid_block
        ),
        aerosol_backscatter_error=_valid_block_error(
            optical.aerosol_backscatter_error_block, valid_block
        ),
        aerosol_extinction=_valid_block_mean(
            optical.aerosol_extinction_block, valid_block
        ),
        aerosol_extinction_error=_valid_block_error(
            optical.aerosol_extinction_error_block, valid_block
        ),
        retrieval_success_flag=valid_block.astype(np.int8),
    )
    return updated, valid_block


def make_backward_retrieve_optical_blocks(
    base_retrieve: Callable[..., tuple[
        MolecularProfiles,
        OpticalProducts,
        RayleighDiagnostics,
        KfsDiagnostics,
    ]],
) -> Callable[..., tuple[
    MolecularProfiles,
    OpticalProducts,
    RayleighDiagnostics,
    KfsDiagnostics,
]]:
    """Wrap the legacy retrieval with the productive backward-only contract."""

    def retrieve_optical_blocks_backward(
        inputs: Any,
        glued: Any,
        molecular: Any,
        altitude_m: np.ndarray,
        config: Mapping[str, Any],
        logger: logging.Logger,
    ) -> tuple[
        MolecularProfiles,
        OpticalProducts,
        RayleighDiagnostics,
        KfsDiagnostics,
    ]:
        molecular_profiles, optical, rayleigh, kfs = base_retrieve(
            inputs,
            glued,
            molecular,
            altitude_m,
            config,
            _SuppressLegacyTwoSidedWarning(logger),
        )

        if str(molecular.kfs_mode) != "backward":
            raise ValueError(
                "Productive optical aggregation requires backward KFS mode."
            )

        optical, valid_block = _reaggregate_backward_optical_products(
            optical, rayleigh, kfs
        )
        n_blocks = int(np.asarray(optical.retrieval_success_flag).size)
        if valid_block.any():
            logger.info(
                "  -> %d nm Rayleigh reference %.0f m [%.0f, %.0f m] | "
                "valid %.1f%% | slope %.3f | variance %.3f | "
                "backward KFS %d/%d blocks",
                int(inputs.wavelength_nm),
                float(rayleigh.reference_altitude_m),
                float(rayleigh.reference_start_altitude_m),
                float(rayleigh.reference_stop_altitude_m),
                100.0 * float(rayleigh.reference_valid_fraction),
                float(rayleigh.reference_relative_slope),
                float(rayleigh.reference_relative_variance),
                int(valid_block.sum()),
                n_blocks,
            )
        else:
            logger.warning(
                "  -> %d nm has no valid backward retrieval block. "
                "Mean optical products set to NaN.",
                int(inputs.wavelength_nm),
            )
        return molecular_profiles, optical, rayleigh, kfs

    return retrieve_optical_blocks_backward
