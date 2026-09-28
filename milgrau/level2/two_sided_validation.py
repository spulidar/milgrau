"""Truth-aware diagnostics for experimental two-sided elastic retrievals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from milgrau.level2.kfs import fernald_inversion


@dataclass(frozen=True, slots=True)
class BranchTruthMetrics:
    """Support and error metrics on one requested oriented branch domain."""

    start_altitude_m: float
    stop_altitude_m: float
    requested_bins: int
    supported_bins: int
    support_fraction: float
    relative_l2_error: float
    integrated_backscatter_relative_error: float


@dataclass(frozen=True, slots=True)
class TwoSidedTruthResult:
    """Two-sided solution plus branch-specific truth metrics."""

    aerosol_backscatter: np.ndarray
    reference_index: int
    reference_altitude_m: float
    backward: BranchTruthMetrics
    forward: BranchTruthMetrics
    backward_endpoint_altitude_m: float
    forward_endpoint_altitude_m: float
    backward_termination_reason: str
    forward_termination_reason: str


def _branch_metrics(
    altitude_m: np.ndarray,
    truth: np.ndarray,
    retrieved: np.ndarray,
    *,
    start_altitude_m: float,
    stop_altitude_m: float,
) -> BranchTruthMetrics:
    requested = (
        (altitude_m >= float(start_altitude_m))
        & (altitude_m <= float(stop_altitude_m))
        & np.isfinite(truth)
    )
    requested_bins = int(np.count_nonzero(requested))
    if requested_bins < 2:
        raise ValueError("truth metric domain must contain at least two bins.")
    supported = requested & np.isfinite(retrieved)
    supported_bins = int(np.count_nonzero(supported))
    support_fraction = float(supported_bins / requested_bins)
    if supported_bins:
        denominator = float(np.linalg.norm(truth[supported]))
        relative_l2 = (
            float(np.linalg.norm(retrieved[supported] - truth[supported]) / denominator)
            if denominator > 0.0
            else float("nan")
        )
    else:
        relative_l2 = float("nan")

    integrated_error = float("nan")
    if supported_bins == requested_bins:
        z = altitude_m[requested]
        truth_column = float(np.trapezoid(truth[requested], z))
        if truth_column > 0.0:
            retrieved_column = float(np.trapezoid(retrieved[requested], z))
            integrated_error = (retrieved_column - truth_column) / truth_column
    return BranchTruthMetrics(
        start_altitude_m=float(start_altitude_m),
        stop_altitude_m=float(stop_altitude_m),
        requested_bins=requested_bins,
        supported_bins=supported_bins,
        support_fraction=support_fraction,
        relative_l2_error=relative_l2,
        integrated_backscatter_relative_error=float(integrated_error),
    )


def evaluate_two_sided_truth(
    *,
    range_corrected_signal: np.ndarray,
    altitude_m: np.ndarray,
    molecular_backscatter: np.ndarray,
    aerosol_backscatter_truth: np.ndarray,
    aerosol_lidar_ratio_sr: float | np.ndarray,
    beta_total_reference: float,
    reference_index: int,
    backward_domain_m: tuple[float, float],
    forward_domain_m: tuple[float, float],
    min_lidar_ratio_sr: float = 10.0,
    allow_negative_aerosol: bool = False,
) -> TwoSidedTruthResult:
    """Evaluate deterministic two-sided bias without converting it to a gate."""
    signal = np.asarray(range_corrected_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    molecular = np.asarray(molecular_backscatter, dtype=np.float64)
    truth = np.asarray(aerosol_backscatter_truth, dtype=np.float64)
    if not (signal.shape == altitude.shape == molecular.shape == truth.shape):
        raise ValueError("two-sided truth inputs must have identical shapes.")
    if any(array.ndim != 1 for array in (signal, altitude, molecular, truth)):
        raise ValueError("two-sided truth inputs must be one-dimensional.")
    ref_idx = int(reference_index)
    retrieved, diagnostics = fernald_inversion(
        signal,
        altitude,
        molecular,
        aerosol_lidar_ratio_sr,
        float(beta_total_reference),
        ref_idx,
        altitude_units="m",
        min_lidar_ratio=float(min_lidar_ratio_sr),
        allow_negative_aerosol=bool(allow_negative_aerosol),
        mode="two_sided",
        return_diagnostics=True,
    )
    backward = _branch_metrics(
        altitude,
        truth,
        retrieved,
        start_altitude_m=float(backward_domain_m[0]),
        stop_altitude_m=float(backward_domain_m[1]),
    )
    forward = _branch_metrics(
        altitude,
        truth,
        retrieved,
        start_altitude_m=float(forward_domain_m[0]),
        stop_altitude_m=float(forward_domain_m[1]),
    )
    return TwoSidedTruthResult(
        aerosol_backscatter=np.asarray(retrieved, dtype=np.float64),
        reference_index=ref_idx,
        reference_altitude_m=float(altitude[ref_idx]),
        backward=backward,
        forward=forward,
        backward_endpoint_altitude_m=float(
            diagnostics["backward_endpoint_altitude_m"]
        ),
        forward_endpoint_altitude_m=float(diagnostics["forward_endpoint_altitude_m"]),
        backward_termination_reason=str(diagnostics["backward_termination_reason"]),
        forward_termination_reason=str(diagnostics["forward_termination_reason"]),
    )
