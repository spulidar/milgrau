"""Canonical metadata for the versioned Level 2 NetCDF schema."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Final

import numpy as np
import xarray as xr

from milgrau.level2.completeness import WavelengthFailureCode, WavelengthFailureStage
from milgrau.level2.contracts import RetrievalInputInvalidReason, SignalSource


def _attrs(
    long_name: str,
    *,
    units: str | None = None,
    description: str | None = None,
    missing_value_semantics: str | None = None,
    unit_status: str | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {"long_name": long_name}
    if units is not None:
        values["units"] = units
    if description is not None:
        values["description"] = description
    if missing_value_semantics is not None:
        values["missing_value_semantics"] = missing_value_semantics
    if unit_status is not None:
        values["unit_status"] = unit_status
    return values


def _flag_attrs(
    values: list[int] | tuple[int, ...] | np.ndarray,
    meanings: str,
    *,
    dtype: np.dtype[Any] | type[np.integer[Any]] = np.int8,
    description: str | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "flag_values": np.asarray(values, dtype=dtype),
        "flag_meanings": meanings,
    }
    if description is not None:
        attrs["description"] = description
    return attrs


def _enum_flag_attrs(
    enum_type: type[IntEnum],
    *,
    dtype: np.dtype[Any] | type[np.integer[Any]],
    description: str,
) -> dict[str, Any]:
    members = tuple(sorted(enum_type, key=int))
    return _flag_attrs(
        [int(member) for member in members],
        " ".join(member.name.lower() for member in members),
        dtype=dtype,
        description=description,
    )


LEVEL2_COORDINATE_METADATA: Final[dict[str, dict[str, Any]]] = {
    "time": _attrs(
        "Level 1 profile time represented by the time-expanded Level 2 diagnostics",
        description=(
            "Original Level 1 profile timestamps. Block-resolved Level 2 state is repeated "
            "onto this coordinate only for traceability; productive inversion is performed per block."
        ),
    ),
    "block_time": _attrs(
        "Start time of the Level 2 averaging and retrieval block",
        description="Timestamp obtained by flooring Level 1 profile time to the configured block interval.",
    ),
    "wavelength": _attrs("Successfully processed elastic lidar wavelength", units="nm"),
    "altitude": {
        **_attrs("Altitude above station", units="m"),
        "positive": "up",
        "reference": "above_station",
    },
}


LEVEL2_DATA_VARIABLE_METADATA: Final[dict[str, dict[str, Any]]] = {
    "molecular_backscatter": _attrs(
        "Molecular volume backscatter coefficient",
        units="m-1 sr-1",
        description="Rayleigh angular volume scattering coefficient at 180 degrees.",
    ),
    "molecular_extinction": _attrs(
        "Molecular extinction coefficient",
        units="m-1",
        description="Total Rayleigh volume scattering coefficient used as molecular extinction.",
    ),
    "molecular_transmission": _attrs(
        "Two-way molecular transmission",
        units="1",
        description="Two-way molecular transmission from station level to altitude and back.",
    ),
    "simulated_molecular_signal": _attrs(
        "Unscaled simulated molecular elastic signal",
        units="m-3 sr-1",
        description=(
            "Molecular signal shape beta_mol*T_mol/r^2 before instrument/Rayleigh calibration; "
            "no lidar system constant is applied."
        ),
    ),
    "simulated_molecular_range_corrected_signal": _attrs(
        "Unscaled simulated molecular range-corrected signal",
        units="m-1 sr-1",
        description="Simulated molecular signal multiplied by range squared before Rayleigh calibration.",
    ),
    "scaled_molecular_range_corrected_signal": _attrs(
        "Rayleigh-calibrated aggregate molecular range-corrected signal",
        description=(
            "Molecular RCS scaled into the selected instrument-signal RCS space using the median "
            "factor from successful block Rayleigh references."
        ),
        missing_value_semantics="NaN when no block Rayleigh reference passed QA.",
        unit_status="source_dependent_relative_range_squared",
    ),
    "scaled_molecular_range_corrected_signal_block": _attrs(
        "Rayleigh-calibrated block molecular range-corrected signal",
        description="Block molecular RCS scaled into the selected instrument-signal RCS space.",
        missing_value_semantics="NaN when the block has no evaluable Rayleigh calibration.",
        unit_status="source_dependent_relative_range_squared",
    ),
}


for name, long_name in {
    "glued_corrected_signal": "Selected corrected lidar signal",
    "glued_corrected_signal_error": "One-sigma uncertainty of selected corrected lidar signal",
    "glued_corrected_signal_block": "Block-mean selected corrected lidar signal",
    "glued_corrected_signal_error_block": "One-sigma uncertainty of block-mean selected corrected lidar signal",
    "glued_corrected_signal_mean": "Aggregate selected corrected lidar signal",
    "glued_corrected_signal_error_mean": "One-sigma uncertainty of aggregate selected corrected lidar signal",
}.items():
    LEVEL2_DATA_VARIABLE_METADATA[name] = _attrs(
        long_name,
        description=(
            "Analog/PC glued or single-channel selected Level 1 corrected signal before range correction. "
            "The numeric scale is source dependent and is not an absolute SI radiometric calibration."
        ),
        missing_value_semantics="NaN indicates unavailable or invalid selected-signal support; values are not interpolated.",
        unit_status="source_dependent_channel_native_corrected",
    )

for name, long_name in {
    "glued_range_corrected_signal": "Selected range-corrected lidar signal",
    "glued_range_corrected_signal_error": "One-sigma uncertainty of selected range-corrected lidar signal",
    "glued_range_corrected_signal_block": "Block-mean selected range-corrected lidar signal",
    "glued_range_corrected_signal_error_block": "One-sigma uncertainty of block-mean selected range-corrected lidar signal",
    "glued_range_corrected_signal_mean": "Aggregate selected range-corrected lidar signal",
    "glued_range_corrected_signal_error_mean": "One-sigma uncertainty of aggregate selected range-corrected lidar signal",
}.items():
    LEVEL2_DATA_VARIABLE_METADATA[name] = _attrs(
        long_name,
        description=(
            "Selected corrected signal multiplied by altitude squared after source selection/gluing. "
            "The amplitude remains source dependent and is not an absolute SI radiometric calibration."
        ),
        missing_value_semantics="NaN indicates unavailable or invalid selected-signal support; values are not interpolated.",
        unit_status="source_dependent_relative_range_squared",
    )

_merge_flag = _flag_attrs(
    [0, 1, 2, 3],
    "photon_counting blend analog invalid",
    description="Per-altitude source used by corrected-signal gluing: PC, blend, analog, or invalid.",
)
for name in ("gluing_merge_source_flag", "gluing_merge_source_flag_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_merge_flag)


LEVEL2_DATA_VARIABLE_METADATA.update(
    {
        "scattering_ratio_mean": _attrs(
            "Aggregate measured-to-molecular scattering ratio",
            units="1",
            description=(
                "Mean of block scattering-ratio diagnostics over blocks accepted by Rayleigh QA and backward KFS. "
                "A finite scattering ratio above the backward KFS boundary is not supported aerosol retrieval."
            ),
            missing_value_semantics="NaN means the ratio is unavailable; finite values do not define retrieval support.",
        ),
        "scattering_ratio_block": _attrs(
            "Block measured-to-molecular scattering ratio",
            units="1",
            description=(
                "Block selected RCS divided by block Rayleigh-scaled molecular RCS. This diagnostic may be finite "
                "outside productive backward aerosol-retrieval support."
            ),
            missing_value_semantics="NaN means the ratio is unavailable; finite values do not define retrieval support.",
        ),
    }
)

for name, long_name, units in (
    ("aerosol_backscatter_mean", "Aggregate aerosol backscatter coefficient", "m-1 sr-1"),
    ("aerosol_backscatter_mean_error", "One-sigma uncertainty of aggregate aerosol backscatter coefficient", "m-1 sr-1"),
    ("aerosol_extinction_mean", "Aggregate aerosol extinction coefficient", "m-1"),
    ("aerosol_extinction_mean_error", "One-sigma uncertainty of aggregate aerosol extinction coefficient", "m-1"),
    ("aerosol_backscatter_block", "Block aerosol backscatter coefficient", "m-1 sr-1"),
    ("aerosol_backscatter_error_block", "One-sigma uncertainty of block aerosol backscatter coefficient", "m-1 sr-1"),
    ("aerosol_extinction_block", "Block aerosol extinction coefficient", "m-1"),
    ("aerosol_extinction_error_block", "One-sigma uncertainty of block aerosol extinction coefficient", "m-1"),
):
    aggregate = "_mean" in name
    LEVEL2_DATA_VARIABLE_METADATA[name] = _attrs(
        long_name,
        units=units,
        description=(
            "Productive backward Klett-Fernald-Sasano optical product using the configured aerosol lidar ratio. "
            + ("Aggregate uses only successful retrieval blocks. " if aggregate else "")
            + "Elastic extinction is conditional on the assumed aerosol lidar ratio."
        ),
        missing_value_semantics=(
            "NaN means no accepted productive backward retrieval support at this altitude; unsupported bins are never filled or bridged."
        ),
    )

LEVEL2_DATA_VARIABLE_METADATA["retrieval_success_flag"] = _flag_attrs(
    [0, 1],
    "not_successful successful",
    description=(
        "Block-level productive retrieval acceptance. Zero includes blocks not attempted or rejected at input, "
        "Rayleigh QA, or backward KFS; use the dedicated diagnostic flags/reason code to identify stage."
    ),
)
LEVEL2_DATA_VARIABLE_METADATA["retrieval_success_fraction"] = _attrs(
    "Fraction of Level 2 blocks with successful productive optical retrieval",
    units="1",
    description="Successful-block fraction per processed wavelength; it does not encode altitude-resolved support.",
)


for suffix, qualifier in (("", "Median over successful block Rayleigh references"), ("_block", "Block Rayleigh reference")):
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_reference_altitude_m{suffix}"] = _attrs(
        f"{qualifier} center altitude", units="m"
    )
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_reference_start_altitude_m{suffix}"] = _attrs(
        f"{qualifier} window start altitude", units="m"
    )
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_reference_stop_altitude_m{suffix}"] = _attrs(
        f"{qualifier} window stop altitude", units="m"
    )
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_reference_valid_bins{suffix}"] = _attrs(
        f"{qualifier} valid-bin count",
        units="1",
        description="Number of finite positive measured/molecular samples in the evaluated reference window.",
    )
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_reference_relative_slope{suffix}"] = _attrs(
        f"{qualifier} relative slope diagnostic",
        units="1",
        description="Absolute fitted ratio change across the valid reference-window altitude span, normalized by mean ratio.",
    )
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_reference_relative_variance{suffix}"] = _attrs(
        f"{qualifier} relative variance diagnostic",
        units="1",
        description="Variance of measured/molecular ratio normalized by squared mean ratio.",
    )
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_reference_valid_fraction{suffix}"] = _attrs(
        f"{qualifier} valid fraction",
        units="1",
        description="Fraction of reference-window bins with finite positive measured/molecular ratio.",
    )
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_calibration_factor{suffix}"] = _attrs(
        f"{qualifier} multiplicative calibration factor",
        description=(
            "Origin-constrained factor mapping simulated molecular RCS into the selected instrument RCS space. "
            "Its units depend on the selected signal source and are intentionally not labeled as SI."
        ),
        unit_status="source_dependent_calibration_factor",
    )
    LEVEL2_DATA_VARIABLE_METADATA[f"rayleigh_calibration_intercept{suffix}"] = _attrs(
        f"{qualifier} free-fit intercept diagnostic",
        description=(
            "Intercept from the free linear Rayleigh diagnostic fit; it is not used as the productive calibration boundary."
        ),
        unit_status="source_dependent_relative_range_squared",
    )

_rayleigh_success_flag = _flag_attrs(
    [0, 1],
    "not_passed passed",
    description="Rayleigh reference QA result; zero includes blocks for which reference QA was not attempted.",
)
LEVEL2_DATA_VARIABLE_METADATA["rayleigh_reference_success_flag"] = {
    **_rayleigh_success_flag,
    "long_name": "Whether at least one block Rayleigh reference passed QA",
}
LEVEL2_DATA_VARIABLE_METADATA["rayleigh_reference_success_flag_block"] = {
    **_rayleigh_success_flag,
    "long_name": "Block Rayleigh reference QA acceptance flag",
}


LEVEL2_DATA_VARIABLE_METADATA["lidar_ratio_assumed_sr"] = _attrs(
    "Assumed aerosol lidar ratio", units="sr", description="Aerosol lidar ratio used by the elastic KFS inversion."
)
LEVEL2_DATA_VARIABLE_METADATA["lidar_ratio_std_sr"] = _attrs(
    "One-sigma aerosol lidar-ratio uncertainty", units="sr", description="Configured lidar-ratio spread sampled by Monte Carlo."
)

_kfs_valid_flag = _flag_attrs(
    [0, 1],
    "not_valid valid",
    description=(
        "Branch validity diagnostic. Zero means either that the branch was not requested by the configured integration mode "
        "or that the requested branch was invalid; KFS_Mode/integration_mode identifies which branch was productive."
    ),
)
for name in (
    "kfs_backward_valid_flag",
    "kfs_forward_valid_flag",
    "kfs_backward_valid_flag_block",
    "kfs_forward_valid_flag_block",
):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_kfs_valid_flag)

_kfs_branch_flag = _flag_attrs(
    [0, 1, 2, 3],
    "invalid backward_below_reference exact_reference_bin forward_above_reference",
    description=(
        "Diagnostic branch relative to the Rayleigh boundary. Productive Level 2 uses the backward branch; "
        "a nonzero branch label alone is not the future altitude-resolved retrieval-support contract."
    ),
)
for name in ("kfs_branch", "kfs_branch_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_kfs_branch_flag)


_gluing_attempted = _flag_attrs([0, 1], "not_attempted attempted")
_gluing_success = _flag_attrs(
    [0, 1], "failed_or_not_attempted approved", description="Approval of the analog/PC gluing candidate."
)
_single_channel = _flag_attrs(
    [0, 1],
    "not_selected selected",
    description="A QA-valid PC-only or analog-only source was selected without mixing channels.",
)
for name in ("gluing_attempted_flag", "gluing_attempted_flag_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_gluing_attempted)
for name in ("gluing_success_flag", "gluing_success_flag_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_gluing_success)
for name in ("single_channel_fallback_flag", "single_channel_fallback_flag_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_single_channel)

for base_name, long_name in (
    ("gluing_split_altitude_m", "Analog/photon-counting gluing split altitude"),
    ("gluing_start_altitude_m", "Analog/photon-counting gluing-window start altitude"),
    ("gluing_stop_altitude_m", "Analog/photon-counting gluing-window stop altitude"),
):
    for suffix in ("", "_block"):
        LEVEL2_DATA_VARIABLE_METADATA[f"{base_name}{suffix}"] = _attrs(long_name, units="m")

for base_name, long_name, description in (
    (
        "gluing_slope",
        "Analog-to-virtual-photon-counting gluing slope",
        "Multiplicative coefficient in virtual_PC = slope*analog + intercept; unit scale is channel/source dependent.",
    ),
    (
        "gluing_intercept",
        "Analog-to-virtual-photon-counting gluing intercept",
        "Additive coefficient in virtual_PC = slope*analog + intercept; numeric scale follows virtual photon-counting signal space.",
    ),
):
    for suffix in ("", "_block"):
        LEVEL2_DATA_VARIABLE_METADATA[f"{base_name}{suffix}"] = _attrs(
            long_name,
            description=description,
            missing_value_semantics="NaN when no approved/evaluable gluing fit exists.",
            unit_status="source_dependent_gluing_coefficient",
        )

for base_name, long_name in (
    ("gluing_correlation", "Analog/photon-counting gluing correlation"),
    ("gluing_relative_rmse", "Relative RMSE of analog/photon-counting gluing fit"),
    ("gluing_relative_bias", "Relative bias of analog/photon-counting gluing fit"),
):
    for suffix in ("", "_block"):
        LEVEL2_DATA_VARIABLE_METADATA[f"{base_name}{suffix}"] = _attrs(
            long_name,
            units="1",
            missing_value_semantics="NaN when no evaluable gluing fit exists.",
        )


_signal_source = _enum_flag_attrs(
    SignalSource,
    dtype=np.int8,
    description="Retrieval input source selected independently for each averaging block.",
)
_input_invalid_reason = _enum_flag_attrs(
    RetrievalInputInvalidReason,
    dtype=np.int8,
    description="Stable reason code for a rejected retrieval input; zero means the selected input is valid.",
)
_input_valid = _flag_attrs(
    [0, 1],
    "invalid valid",
    description="Whether the selected signal passed the minimum source-specific QA required before Rayleigh/KFS retrieval.",
)
for name in ("signal_source_flag", "signal_source_flag_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_signal_source)
for name in ("retrieval_input_valid_flag", "retrieval_input_valid_flag_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_input_valid)
for name in ("retrieval_input_invalid_reason", "retrieval_input_invalid_reason_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = dict(_input_invalid_reason)
for name in ("retrieval_input_snr_median", "retrieval_input_snr_median_block"):
    LEVEL2_DATA_VARIABLE_METADATA[name] = _attrs(
        "Median retrieval-input signal-to-noise diagnostic",
        units="1",
        description=(
            "Median absolute selected signal divided by one-sigma uncertainty over viable Rayleigh-search support; "
            "diagnostic only, with no productive hard SNR threshold."
        ),
        missing_value_semantics="NaN when SNR cannot be evaluated from the selected input.",
    )


for name, long_name in (
    ("requested_wavelengths", "Requested elastic wavelengths"),
    ("processed_wavelengths", "Successfully processed elastic wavelengths"),
    ("failed_wavelengths", "Failed requested elastic wavelengths"),
):
    LEVEL2_DATA_VARIABLE_METADATA[name] = _attrs(long_name, units="nm")

LEVEL2_DATA_VARIABLE_METADATA["failed_wavelength_stage"] = {
    "long_name": "Stable processing stage for each failed wavelength",
    **_enum_flag_attrs(
        WavelengthFailureStage,
        dtype=np.int8,
        description="Machine-readable failure stage; use with failed_wavelength_code and the human-readable message.",
    ),
}
LEVEL2_DATA_VARIABLE_METADATA["failed_wavelength_code"] = {
    "long_name": "Stable failure code for each failed wavelength",
    **_enum_flag_attrs(
        WavelengthFailureCode,
        dtype=np.int16,
        description="Machine-readable failure class independent of Python exception names.",
    ),
}
LEVEL2_DATA_VARIABLE_METADATA["failed_wavelength_message"] = _attrs(
    "Human-readable diagnosis for each failed wavelength",
    description="Short diagnosis for operators; stable program logic must use numeric failure stage/code.",
)
LEVEL2_DATA_VARIABLE_METADATA["failed_wavelength_cause"] = _attrs(
    "Compact exception cause summary for each failed wavelength",
    description="Exception class/summary only; full tracebacks remain operational logs rather than product schema.",
)


LEVEL2_METADATA_VARIABLE_NAMES: Final[tuple[str, ...]] = tuple(LEVEL2_DATA_VARIABLE_METADATA)


def apply_level2_variable_metadata(ds: xr.Dataset) -> None:
    """Apply the complete canonical coordinate/data-variable metadata contract in place."""
    missing_coords = [name for name in LEVEL2_COORDINATE_METADATA if name not in ds.coords]
    missing_variables = [name for name in LEVEL2_DATA_VARIABLE_METADATA if name not in ds.data_vars]
    extra_variables = [name for name in ds.data_vars if name not in LEVEL2_DATA_VARIABLE_METADATA]
    if missing_coords or missing_variables or extra_variables:
        raise ValueError(
            "Level 2 metadata inventory mismatch: "
            f"missing_coords={missing_coords}, missing_variables={missing_variables}, extra_variables={extra_variables}"
        )
    for name, attrs in LEVEL2_COORDINATE_METADATA.items():
        ds[name].attrs.update(attrs)
    for name, attrs in LEVEL2_DATA_VARIABLE_METADATA.items():
        ds[name].attrs.update(attrs)
