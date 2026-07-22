"""Typed runtime contract for one-wavelength Level 2 retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SignalSource(IntEnum):
    """Per-block source selected as the Level 2 retrieval input."""

    INVALID = 0
    GLUED = 1
    PHOTON_COUNTING = 2
    ANALOG = 3


class RetrievalInputInvalidReason(IntEnum):
    """Stable summary code for a rejected retrieval input block."""

    VALID = 0
    NO_VALID_CHANNEL = 1
    NONFINITE_SIGNAL = 2
    INVALID_UNCERTAINTY = 3
    PHOTON_COUNTING_SATURATED = 4
    INSUFFICIENT_VERTICAL_COVERAGE = 5
    NONPOSITIVE_SIGNAL = 6
    LEVEL1_CORRECTION_FAILED_OR_UNCONFIRMED = 7
    SATURATION_DIAGNOSTIC_MISSING = 8
    SNR_UNAVAILABLE = 9
    SINGLE_CHANNEL_FALLBACK_DISABLED = 10


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
    """Time, block and mean signals selected from glued, PC-only, or AN-only input."""

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
    retrieval_success_flag: np.ndarray


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

    attempted_flag: np.ndarray
    success_flag: np.ndarray
    single_channel_fallback_flag: np.ndarray
    split_altitude_m: np.ndarray
    start_altitude_m: np.ndarray
    stop_altitude_m: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    correlation: np.ndarray
    relative_rmse: np.ndarray
    relative_bias: np.ndarray
    attempted_flag_block: np.ndarray
    success_flag_block: np.ndarray
    single_channel_fallback_flag_block: np.ndarray
    split_altitude_m_block: np.ndarray
    start_altitude_m_block: np.ndarray
    stop_altitude_m_block: np.ndarray
    slope_block: np.ndarray
    intercept_block: np.ndarray
    correlation_block: np.ndarray
    relative_rmse_block: np.ndarray
    relative_bias_block: np.ndarray


@dataclass(frozen=True, slots=True)
class SignalSelectionDiagnostics:
    """Selected source and retrieval-input QA, independent of gluing success."""

    source_flag: np.ndarray
    retrieval_input_valid_flag: np.ndarray
    retrieval_input_invalid_reason: np.ndarray
    retrieval_input_snr_median: np.ndarray
    source_flag_block: np.ndarray
    retrieval_input_valid_flag_block: np.ndarray
    retrieval_input_invalid_reason_block: np.ndarray
    retrieval_input_snr_median_block: np.ndarray


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
    signal_selection: SignalSelectionDiagnostics

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
        _require_array("optical.retrieval_success_flag", self.optical.retrieval_success_flag, (n_block,), np.int8)

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

        for name in ("attempted_flag", "success_flag", "single_channel_fallback_flag"):
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
        for name in ("attempted_flag_block", "success_flag_block", "single_channel_fallback_flag_block"):
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

        for name in (
            "source_flag",
            "retrieval_input_valid_flag",
            "retrieval_input_invalid_reason",
        ):
            _require_array(f"signal_selection.{name}", getattr(self.signal_selection, name), (n_time,), np.int8)
        _require_array(
            "signal_selection.retrieval_input_snr_median",
            self.signal_selection.retrieval_input_snr_median,
            (n_time,),
            np.float64,
        )
        for name in (
            "source_flag_block",
            "retrieval_input_valid_flag_block",
            "retrieval_input_invalid_reason_block",
        ):
            _require_array(f"signal_selection.{name}", getattr(self.signal_selection, name), (n_block,), np.int8)
        _require_array(
            "signal_selection.retrieval_input_snr_median_block",
            self.signal_selection.retrieval_input_snr_median_block,
            (n_block,),
            np.float64,
        )
        self._validate_scientific_state_invariants()

    def _validate_scientific_state_invariants(self) -> None:
        """Reject contradictory gluing, source, input-validity and retrieval states."""
        attempted = self.gluing.attempted_flag_block
        glued = self.gluing.success_flag_block
        fallback = self.gluing.single_channel_fallback_flag_block
        source = self.signal_selection.source_flag_block
        input_valid = self.signal_selection.retrieval_input_valid_flag_block
        reason = self.signal_selection.retrieval_input_invalid_reason_block
        retrieval_success = self.optical.retrieval_success_flag

        for label, flags in (
            ("gluing.attempted_flag_block", attempted),
            ("gluing.success_flag_block", glued),
            ("gluing.single_channel_fallback_flag_block", fallback),
            ("signal_selection.retrieval_input_valid_flag_block", input_valid),
            ("optical.retrieval_success_flag", retrieval_success),
        ):
            if not np.isin(flags, (0, 1)).all():
                raise ValueError(f"{label} must contain only 0 or 1.")
        if not np.isin(source, [int(item) for item in SignalSource]).all():
            raise ValueError("signal_selection.source_flag_block contains an unknown source code.")
        if not np.isin(reason, [int(item) for item in RetrievalInputInvalidReason]).all():
            raise ValueError("signal_selection.retrieval_input_invalid_reason_block contains an unknown reason code.")
        if np.any((attempted == 0) & (glued == 1)):
            raise ValueError("Successful gluing requires gluing_attempted_flag=1.")
        if np.any((glued == 1) & (source != SignalSource.GLUED)):
            raise ValueError("gluing_success_flag=1 requires signal source glued.")
        if np.any((source == SignalSource.GLUED) & ((attempted != 1) | (glued != 1) | (fallback != 0))):
            raise ValueError("Source glued requires attempted and successful gluing without fallback.")
        single = np.isin(source, (SignalSource.PHOTON_COUNTING, SignalSource.ANALOG))
        if np.any(single & ((glued != 0) | (fallback != 1) | (input_valid != 1))):
            raise ValueError(
                "A single-channel source requires failed/not-attempted gluing, fallback=1, and valid input."
            )
        invalid = source == SignalSource.INVALID
        if np.any(invalid & ((input_valid != 0) | (fallback != 0))):
            raise ValueError("An invalid source requires invalid retrieval input and no fallback selection.")
        if np.any((input_valid == 1) & ((source == SignalSource.INVALID) | (reason != 0))):
            raise ValueError("A valid retrieval input requires a selected source and reason code 0.")
        if np.any((input_valid == 0) & (reason == RetrievalInputInvalidReason.VALID)):
            raise ValueError("An invalid retrieval input requires a non-zero reason code.")
        if np.any((retrieval_success == 1) & (input_valid != 1)):
            raise ValueError("retrieval_success_flag=1 requires retrieval_input_valid_flag=1.")

        invalid_blocks = np.flatnonzero(input_valid == 0)
        for block_index in invalid_blocks:
            for name in (
                "scattering_ratio_block",
                "aerosol_backscatter_block",
                "aerosol_backscatter_error_block",
                "aerosol_extinction_block",
                "aerosol_extinction_error_block",
            ):
                if not np.isnan(getattr(self.optical, name)[block_index, :]).all():
                    raise ValueError(
                        f"optical.{name} must be all-NaN when retrieval_input_valid_flag=0 "
                        f"for block {block_index}."
                    )


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
