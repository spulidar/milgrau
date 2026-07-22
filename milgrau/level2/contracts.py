"""Typed runtime contract for one-wavelength Level 2 retrieval results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MolecularProfiles:
    """Molecular inputs and calibrated profiles on the altitude grid."""

    source: str
    backscatter: np.ndarray
    extinction: np.ndarray
    transmission: np.ndarray
    simulated_signal: np.ndarray
    simulated_range_corrected_signal: np.ndarray
    scaled_range_corrected_signal: np.ndarray
    scaled_range_corrected_signal_block: np.ndarray


@dataclass(frozen=True, slots=True)
class GluedSignals:
    """Time, block and mean signals produced by analog/PC gluing."""

    source: str
    analog_channel: str | None
    photon_channel: str | None
    corrected_signal: np.ndarray
    corrected_signal_error: np.ndarray
    corrected_signal_block: np.ndarray
    corrected_signal_error_block: np.ndarray
    corrected_signal_mean: np.ndarray
    corrected_signal_error_mean: np.ndarray
    range_corrected_signal: np.ndarray
    range_corrected_signal_error: np.ndarray
    range_corrected_signal_block: np.ndarray
    range_corrected_signal_error_block: np.ndarray
    range_corrected_signal_mean: np.ndarray
    range_corrected_signal_error_mean: np.ndarray
    merge_source_flag: np.ndarray
    merge_source_flag_block: np.ndarray


@dataclass(frozen=True, slots=True)
class OpticalProducts:
    """Block and aggregate aerosol optical products."""

    scattering_ratio_mean: np.ndarray
    scattering_ratio_block: np.ndarray
    aerosol_backscatter: np.ndarray
    aerosol_backscatter_error: np.ndarray
    aerosol_extinction: np.ndarray
    aerosol_extinction_error: np.ndarray
    aerosol_backscatter_block: np.ndarray
    aerosol_backscatter_error_block: np.ndarray
    aerosol_extinction_block: np.ndarray
    aerosol_extinction_error_block: np.ndarray
    valid_retrieval_block_flag: np.ndarray


@dataclass(frozen=True, slots=True)
class RayleighDiagnostics:
    """Aggregate and per-block molecular-reference diagnostics."""

    reference_altitude_m: float
    reference_start_altitude_m: float
    reference_stop_altitude_m: float
    reference_valid_bins: int
    reference_success_flag: int
    reference_relative_slope: float
    reference_relative_variance: float
    reference_valid_fraction: float
    calibration_factor: float
    calibration_intercept: float
    reference_altitude_m_block: np.ndarray
    reference_start_altitude_m_block: np.ndarray
    reference_stop_altitude_m_block: np.ndarray
    reference_valid_bins_block: np.ndarray
    reference_success_flag_block: np.ndarray
    reference_relative_slope_block: np.ndarray
    reference_relative_variance_block: np.ndarray
    reference_valid_fraction_block: np.ndarray
    calibration_factor_block: np.ndarray
    calibration_intercept_block: np.ndarray


@dataclass(frozen=True, slots=True)
class KfsDiagnostics:
    """Lidar-ratio assumptions plus KFS branch and side validity."""

    lidar_ratio_assumed_sr: float
    lidar_ratio_std_sr: float
    backward_valid_flag: int
    forward_valid_flag: int
    backward_valid_flag_block: np.ndarray
    forward_valid_flag_block: np.ndarray
    branch: np.ndarray
    branch_block: np.ndarray


@dataclass(frozen=True, slots=True)
class GluingDiagnostics:
    """Time-expanded and per-block gluing QA diagnostics."""

    success_flag: np.ndarray
    fallback_flag: np.ndarray
    split_altitude_m: np.ndarray
    start_altitude_m: np.ndarray
    stop_altitude_m: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    correlation: np.ndarray
    relative_rmse: np.ndarray
    relative_bias: np.ndarray
    success_flag_block: np.ndarray
    fallback_flag_block: np.ndarray
    split_altitude_m_block: np.ndarray
    start_altitude_m_block: np.ndarray
    stop_altitude_m_block: np.ndarray
    slope_block: np.ndarray
    intercept_block: np.ndarray
    correlation_block: np.ndarray
    relative_rmse_block: np.ndarray
    relative_bias_block: np.ndarray


@dataclass(frozen=True, slots=True)
class WavelengthRetrievalResult:
    """Complete, validated result for one requested wavelength."""

    wavelength_nm: int
    block_time: np.ndarray
    molecular: MolecularProfiles
    glued: GluedSignals
    optical: OpticalProducts
    rayleigh: RayleighDiagnostics
    kfs: KfsDiagnostics
    gluing: GluingDiagnostics

    def validate(self, *, n_time: int, n_altitude: int) -> None:
        """Validate scalar types plus exact array shapes and dtypes."""
        if isinstance(self.wavelength_nm, bool) or not isinstance(self.wavelength_nm, (int, np.integer)):
            raise TypeError("wavelength_nm must be an integer.")
        if int(self.wavelength_nm) <= 0:
            raise ValueError("wavelength_nm must be positive.")
        if not isinstance(self.block_time, np.ndarray):
            raise TypeError(f"block_time must be numpy.ndarray; got {type(self.block_time).__name__}.")
        _require_array("block_time", self.block_time, (self.block_time.shape[0],), "datetime64[ns]")
        n_block = self.block_time.shape[0]
        if n_block <= 0:
            raise ValueError("block_time must contain at least one block.")
        if n_time <= 0 or n_altitude <= 0:
            raise ValueError("n_time and n_altitude must be positive.")

        _require_nonempty_string("molecular.source", self.molecular.source)
        _require_nonempty_string("glued.source", self.glued.source)
        _require_optional_string("glued.analog_channel", self.glued.analog_channel)
        _require_optional_string("glued.photon_channel", self.glued.photon_channel)
        if self.glued.analog_channel is None and self.glued.photon_channel is None:
            raise ValueError("At least one glued source channel must be present.")

        for name in (
            "backscatter",
            "extinction",
            "transmission",
            "simulated_signal",
            "simulated_range_corrected_signal",
            "scaled_range_corrected_signal",
        ):
            _require_array(f"molecular.{name}", getattr(self.molecular, name), (n_altitude,), np.float64)
        _require_array(
            "molecular.scaled_range_corrected_signal_block",
            self.molecular.scaled_range_corrected_signal_block,
            (n_block, n_altitude),
            np.float64,
        )

        for name in (
            "corrected_signal",
            "corrected_signal_error",
            "range_corrected_signal",
            "range_corrected_signal_error",
        ):
            _require_array(f"glued.{name}", getattr(self.glued, name), (n_time, n_altitude), np.float64)
        for name in (
            "corrected_signal_block",
            "corrected_signal_error_block",
            "range_corrected_signal_block",
            "range_corrected_signal_error_block",
        ):
            _require_array(f"glued.{name}", getattr(self.glued, name), (n_block, n_altitude), np.float64)
        for name in (
            "corrected_signal_mean",
            "corrected_signal_error_mean",
            "range_corrected_signal_mean",
            "range_corrected_signal_error_mean",
        ):
            _require_array(f"glued.{name}", getattr(self.glued, name), (n_altitude,), np.float64)
        _require_array("glued.merge_source_flag", self.glued.merge_source_flag, (n_time, n_altitude), np.int8)
        _require_array(
            "glued.merge_source_flag_block",
            self.glued.merge_source_flag_block,
            (n_block, n_altitude),
            np.int8,
        )

        for name in (
            "scattering_ratio_mean",
            "aerosol_backscatter",
            "aerosol_backscatter_error",
            "aerosol_extinction",
            "aerosol_extinction_error",
        ):
            _require_array(f"optical.{name}", getattr(self.optical, name), (n_altitude,), np.float64)
        for name in (
            "scattering_ratio_block",
            "aerosol_backscatter_block",
            "aerosol_backscatter_error_block",
            "aerosol_extinction_block",
            "aerosol_extinction_error_block",
        ):
            _require_array(f"optical.{name}", getattr(self.optical, name), (n_block, n_altitude), np.float64)
        _require_array(
            "optical.valid_retrieval_block_flag",
            self.optical.valid_retrieval_block_flag,
            (n_block,),
            np.int8,
        )

        for name in (
            "reference_altitude_m",
            "reference_start_altitude_m",
            "reference_stop_altitude_m",
            "reference_relative_slope",
            "reference_relative_variance",
            "reference_valid_fraction",
            "calibration_factor",
            "calibration_intercept",
        ):
            _require_float(f"rayleigh.{name}", getattr(self.rayleigh, name))
        _require_integer("rayleigh.reference_valid_bins", self.rayleigh.reference_valid_bins)
        _require_integer("rayleigh.reference_success_flag", self.rayleigh.reference_success_flag)
        for name in (
            "reference_altitude_m_block",
            "reference_start_altitude_m_block",
            "reference_stop_altitude_m_block",
            "reference_relative_slope_block",
            "reference_relative_variance_block",
            "reference_valid_fraction_block",
            "calibration_factor_block",
            "calibration_intercept_block",
        ):
            _require_array(f"rayleigh.{name}", getattr(self.rayleigh, name), (n_block,), np.float64)
        _require_array(
            "rayleigh.reference_valid_bins_block",
            self.rayleigh.reference_valid_bins_block,
            (n_block,),
            np.int32,
        )
        _require_array(
            "rayleigh.reference_success_flag_block",
            self.rayleigh.reference_success_flag_block,
            (n_block,),
            np.int8,
        )

        _require_float("kfs.lidar_ratio_assumed_sr", self.kfs.lidar_ratio_assumed_sr)
        _require_float("kfs.lidar_ratio_std_sr", self.kfs.lidar_ratio_std_sr)
        _require_integer("kfs.backward_valid_flag", self.kfs.backward_valid_flag)
        _require_integer("kfs.forward_valid_flag", self.kfs.forward_valid_flag)
        _require_array(
            "kfs.backward_valid_flag_block",
            self.kfs.backward_valid_flag_block,
            (n_block,),
            np.int8,
        )
        _require_array(
            "kfs.forward_valid_flag_block",
            self.kfs.forward_valid_flag_block,
            (n_block,),
            np.int8,
        )
        _require_array("kfs.branch", self.kfs.branch, (n_altitude,), np.int8)
        _require_array("kfs.branch_block", self.kfs.branch_block, (n_block, n_altitude), np.int8)

        for name in ("success_flag", "fallback_flag"):
            _require_array(f"gluing.{name}", getattr(self.gluing, name), (n_time,), np.int8)
        for name in (
            "split_altitude_m",
            "start_altitude_m",
            "stop_altitude_m",
            "slope",
            "intercept",
            "correlation",
            "relative_rmse",
            "relative_bias",
        ):
            _require_array(f"gluing.{name}", getattr(self.gluing, name), (n_time,), np.float64)
        for name in ("success_flag_block", "fallback_flag_block"):
            _require_array(f"gluing.{name}", getattr(self.gluing, name), (n_block,), np.int8)
        for name in (
            "split_altitude_m_block",
            "start_altitude_m_block",
            "stop_altitude_m_block",
            "slope_block",
            "intercept_block",
            "correlation_block",
            "relative_rmse_block",
            "relative_bias_block",
        ):
            _require_array(f"gluing.{name}", getattr(self.gluing, name), (n_block,), np.float64)


def validate_retrieval_results(
    results: list[WavelengthRetrievalResult],
    *,
    n_time: int,
    n_altitude: int,
) -> None:
    """Validate a non-empty, mutually conformable wavelength-result collection."""
    if not isinstance(results, list):
        raise TypeError("results must be a list of WavelengthRetrievalResult objects.")
    if not results:
        raise ValueError("At least one wavelength retrieval result is required.")
    wavelengths: set[int] = set()
    reference_block_time: np.ndarray | None = None
    for index, result in enumerate(results):
        if not isinstance(result, WavelengthRetrievalResult):
            raise TypeError(f"results[{index}] must be WavelengthRetrievalResult; got {type(result).__name__}.")
        result.validate(n_time=n_time, n_altitude=n_altitude)
        wavelength = int(result.wavelength_nm)
        if wavelength in wavelengths:
            raise ValueError(f"Duplicate wavelength retrieval result: {wavelength} nm.")
        wavelengths.add(wavelength)
        if reference_block_time is None:
            reference_block_time = result.block_time
        elif not np.array_equal(result.block_time, reference_block_time):
            raise ValueError("All wavelength retrieval results must share identical block_time coordinates.")


def _require_array(name: str, value: object, shape: tuple[int, ...], dtype: object) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be numpy.ndarray; got {type(value).__name__}.")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {value.dtype}.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {value.shape}.")


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string.")


def _require_optional_string(name: str, value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise TypeError(f"{name} must be None or a non-empty string.")


def _require_float(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        raise TypeError(f"{name} must be a floating-point scalar; got {type(value).__name__}.")


def _require_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer scalar; got {type(value).__name__}.")
