"""Tests for LEBEAR round-1 architecture and traceability behavior."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.paths import level2_output_path
from milgrau.level2 import lebear
from milgrau.level2 import qa as level2_qa
from milgrau.level2 import retrieval as retrieval_module
from milgrau.level2.contracts import WavelengthRetrievalResult
from milgrau.level2.contracts import RetrievalInputInvalidReason, SignalSource
from milgrau.level2.dataset import build_level2_dataset
from milgrau.level2.retrieval import (
    BlockGluingResult,
    RetrievalStageError,
    WavelengthBlockInputs,
    assemble_wavelength_result,
    build_molecular_model,
    glue_signal_blocks,
    prepare_wavelength_blocks,
    process_wavelength,
    retrieve_optical_blocks,
)
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary


class _ListLogger:
    """Small logger stub used to capture pipeline messages in tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(f"INFO: {message}")

    def warning(self, message: str) -> None:
        self.messages.append(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.messages.append(f"ERROR: {message}")


def _dataset_contract_digest(dataset: xr.Dataset) -> str:
    """Hash values, dimensions, dtypes and attributes for the frozen L2 fixture."""
    digest = hashlib.sha256()
    for group_name, variables in (("coords", dataset.coords), ("data_vars", dataset.data_vars)):
        digest.update(group_name.encode())
        for name in sorted(variables):
            array = dataset[name]
            values = np.ascontiguousarray(array.values)
            metadata = {
                "name": name,
                "dims": list(array.dims),
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "attrs": dict(array.attrs),
            }
            digest.update(json.dumps(metadata, sort_keys=True, default=str, separators=(",", ":")).encode())
            digest.update(values.tobytes())
    digest.update(json.dumps(dict(dataset.attrs), sort_keys=True, default=str, separators=(",", ":")).encode())
    return digest.hexdigest()


def _write_level1(path: Path, channels: list[str], n_altitude: int = 240) -> Path:
    """Write a synthetic Level 1 file with smooth finite RCS profiles."""
    time = pd.date_range("2024-01-01T00:00:00", periods=3, freq="5min")
    altitude = np.arange(n_altitude, dtype=np.float64) * 7.5
    channel = np.array(channels, dtype=object)
    shape = (time.size, channel.size, altitude.size)

    base = np.exp(-altitude / 900.0) + 0.1
    corrected = np.empty(shape, dtype=np.float32)
    corrected_error = np.empty(shape, dtype=np.float32)
    rcs = np.empty(shape, dtype=np.float32)
    rcs_error = np.empty(shape, dtype=np.float32)
    for time_idx in range(time.size):
        for channel_idx, channel_name in enumerate(channels):
            scale = 1.0 + 0.05 * time_idx + 0.02 * channel_idx
            if channel_name.endswith(".PC"):
                scale *= 1.05
            corrected[time_idx, channel_idx, :] = scale * base
            corrected_error[time_idx, channel_idx, :] = 0.02 * np.abs(corrected[time_idx, channel_idx, :])
            rcs[time_idx, channel_idx, :] = corrected[time_idx, channel_idx, :] * altitude.astype(np.float32) ** 2
            rcs_error[time_idx, channel_idx, :] = corrected_error[time_idx, channel_idx, :] * altitude.astype(np.float32) ** 2

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_error),
            "range_corrected_signal": (("time", "channel", "altitude"), rcs),
            "range_corrected_signal_error": (("time", "channel", "altitude"), rcs_error),
            "pc_saturation_mask": (("time", "channel", "altitude"), np.zeros(shape, dtype=np.int8)),
            "channel_correction_success": (("channel",), np.ones(channel.size, dtype=np.int8)),
            "deadtime_correction_applied": (
                ("channel",),
                np.asarray([int(name.endswith(".PC")) for name in channels], dtype=np.int8),
            ),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={"Processing_level": "Level 1 synthetic test product", "Altitude_units": "m"},
    )
    ds.to_netcdf(path)
    return path


def _config(tmp_path: Path, *, allow_single_channel_fallback: bool = True, kfs_mode: str = "two_sided") -> dict:
    """Return a compact LEBEAR config for synthetic tests."""
    return {
        "processing": {"incremental": True},
        "directories": {"processed_data": str(tmp_path)},
        "site": {"station_altitude_m": 760.0},
        "inversion": {
            "wavelengths_to_process": [532],
            "kfs_mode": kfs_mode,
            "temporal_average_minutes": 15,
            "monte_carlo_iterations": 5,
            "random_seed": 123,
            "molecular_fit": {"ref_alt_min_m": 500.0, "ref_alt_max_m": 1500.0, "ref_window_bins": 20},
            "gluing": {
                "window_length_bins": 20,
                "correlation_threshold": 0.5,
                "search_min_idx": 20,
                "search_max_idx": 120,
                "allow_single_channel_fallback": allow_single_channel_fallback,
                "single_channel_priority": "photon_counting",
            },
            "lidar_ratios_sr": {"532": {"01": 60.0}},
            "lidar_ratio_std_sr": {"532": 5.0},
        },
        "visualization": {"level2_qa": {"enabled": False}},
    }


def test_typed_retrieval_freezes_fernald_v2_level2_dataset(tmp_path: Path) -> None:
    """The typed contract freezes the deliberately changed Fernald-v2 product."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        result = process_wavelength(ds_l1, 532, altitude_m, config, logger)  # type: ignore[arg-type]
        dataset = build_level2_dataset(ds_l1, [result], altitude_m, level1, config)

    assert isinstance(result, WavelengthRetrievalResult)
    result.validate(n_time=3, n_altitude=240)
    # SCI-002 deliberately changes state variables/metadata. Numeric assertions
    # elsewhere preserve the already-approved glued Fernald-v2 optical path.
    assert _dataset_contract_digest(dataset) == "4aa7f2102e5a2d4060d32311539780196a4d0b2e794770b96782ba715457d002"


def test_retrieval_stages_have_explicit_conformable_boundaries(tmp_path: Path) -> None:
    """Selection, gluing, molecular, optical and assembly stages compose explicitly."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        inputs = prepare_wavelength_blocks(ds_l1, 532, altitude_m, config)
        glued = glue_signal_blocks(inputs, altitude_m, logger)  # type: ignore[arg-type]
        molecular_model = build_molecular_model(ds_l1, 532, altitude_m, config)
        molecular, optical, rayleigh, kfs = retrieve_optical_blocks(
            inputs,
            glued,
            molecular_model,
            altitude_m,
            config,
            logger,  # type: ignore[arg-type]
        )
        result = assemble_wavelength_result(inputs, glued, molecular, optical, rayleigh, kfs)

    assert isinstance(inputs, WavelengthBlockInputs)
    assert isinstance(glued, BlockGluingResult)
    assert inputs.analog_block is not None and inputs.analog_block.shape == (1, 240)
    assert inputs.photon_block is not None and inputs.photon_block.shape == (1, 240)
    assert glued.corrected_signal.shape == (1, 240)
    assert np.all(glued.attempted_flag == 1)
    assert np.all(glued.success_flag == 1)
    assert np.all(glued.signal_source_flag == SignalSource.GLUED)
    assert np.all(glued.retrieval_input_valid_flag == 1)
    assert result.optical.retrieval_success_flag.dtype == np.int8
    assert np.all(result.optical.retrieval_success_flag == 1)
    assert np.isfinite(result.optical.aerosol_backscatter_block).any()
    result.validate(n_time=3, n_altitude=240)


def test_gluing_stage_exposes_single_channel_fallback(tmp_path: Path) -> None:
    """The block-gluing stage reports its photon-only fallback without later stages."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.PC"])
    config = _config(tmp_path)
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        inputs = prepare_wavelength_blocks(ds_l1, 532, altitude_m, config)
        glued = glue_signal_blocks(inputs, altitude_m, logger)  # type: ignore[arg-type]

    assert glued.source == "block_mean_corrected_signal_single_channel_532.PC"
    assert np.all(glued.attempted_flag == 0)
    assert np.all(glued.single_channel_fallback_flag == 1)
    assert np.all(glued.success_flag == 0)
    assert np.all(glued.signal_source_flag == SignalSource.PHOTON_COUNTING)
    assert np.all(glued.retrieval_input_valid_flag == 1)
    assert np.all(glued.merge_source_flag == 0)
    assert inputs.photon_error_block is not None
    np.testing.assert_allclose(glued.corrected_signal_error, inputs.photon_error_block)


@pytest.mark.parametrize(
    ("channels", "expected_source", "expected_merge_source"),
    [
        (["532.PC"], SignalSource.PHOTON_COUNTING, 0),
        (["532.AN"], SignalSource.ANALOG, 2),
    ],
)
def test_sci002_valid_single_channel_executes_retrieval(
    tmp_path: Path,
    channels: list[str],
    expected_source: SignalSource,
    expected_merge_source: int,
) -> None:
    """PC-only and AN-only blocks remain independent of gluing and reach KFS."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", channels)
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        result = process_wavelength(ds_l1, 532, altitude_m, _config(tmp_path), logger)  # type: ignore[arg-type]

    assert np.isfinite(result.glued.corrected_signal_block).all()
    assert np.all(result.glued.merge_source_flag_block == expected_merge_source)
    assert np.all(result.gluing.attempted_flag_block == 0)
    assert np.all(result.gluing.single_channel_fallback_flag_block == 1)
    assert np.all(result.gluing.success_flag_block == 0)
    assert np.all(result.signal_selection.source_flag_block == expected_source)
    assert np.all(result.signal_selection.retrieval_input_valid_flag_block == 1)
    assert np.all(result.optical.retrieval_success_flag == 1)
    assert np.isfinite(result.optical.aerosol_backscatter_block).any()
    assert np.all(result.rayleigh.reference_success_flag_block == 1)


@pytest.mark.parametrize(
    ("channel", "expected_source"),
    [
        ("532.PC", SignalSource.PHOTON_COUNTING),
        ("532.AN", SignalSource.ANALOG),
    ],
)
def test_sci002_lebear_file_contains_finite_single_channel_optics(
    tmp_path: Path,
    channel: str,
    expected_source: SignalSource,
) -> None:
    """The complete writer persists finite PC-only and AN-only optical products."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", [channel])
    summary = lebear.process_single_level1_file(level1, _config(tmp_path), _ListLogger())  # type: ignore[arg-type]

    assert summary.results[0].status is ExecutionStatus.SUCCESS
    with xr.open_dataset(level2_output_path(level1)) as ds:
        assert int(ds["gluing_attempted_flag_block"].item()) == 0
        assert int(ds["gluing_success_flag_block"].item()) == 0
        assert int(ds["single_channel_fallback_flag_block"].item()) == 1
        assert int(ds["signal_source_flag_block"].item()) == expected_source
        assert int(ds["retrieval_input_valid_flag_block"].item()) == 1
        assert int(ds["retrieval_success_flag"].item()) == 1
        assert np.isfinite(ds["aerosol_backscatter_block"].values).any()


def test_sci002_failed_gluing_selects_valid_pc_and_executes_retrieval(tmp_path: Path) -> None:
    """Rejected gluing may coexist with a valid, deterministic PC retrieval."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    config["inversion"]["gluing"]["correlation_threshold"] = 1.1
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        result = process_wavelength(ds_l1, 532, altitude_m, config, logger)  # type: ignore[arg-type]

    assert np.isfinite(result.glued.corrected_signal_block).all()
    assert np.all(result.glued.merge_source_flag_block == 0)
    assert np.all(result.gluing.attempted_flag_block == 1)
    assert np.all(result.gluing.single_channel_fallback_flag_block == 1)
    assert np.all(result.gluing.success_flag_block == 0)
    assert np.all(result.signal_selection.source_flag_block == SignalSource.PHOTON_COUNTING)
    assert np.all(result.signal_selection.retrieval_input_valid_flag_block == 1)
    assert np.all(result.optical.retrieval_success_flag == 1)
    assert np.isfinite(result.optical.aerosol_backscatter_block).any()


def test_sci002_failed_gluing_rejects_saturated_pc_and_selects_valid_analog(tmp_path: Path) -> None:
    """PC saturation invalidates only PC, allowing independently valid AN fallback."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    config["inversion"]["gluing"]["correlation_threshold"] = 1.1
    logger = _ListLogger()
    with xr.open_dataset(level1) as opened:
        ds_l1 = opened.load()
    ds_l1["pc_saturation_mask"].loc[dict(channel="532.PC")] = 1
    altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)

    result = process_wavelength(ds_l1, 532, altitude_m, config, logger)  # type: ignore[arg-type]

    assert np.all(result.gluing.attempted_flag_block == 1)
    assert np.all(result.gluing.success_flag_block == 0)
    assert np.all(result.signal_selection.source_flag_block == SignalSource.ANALOG)
    assert np.all(result.signal_selection.retrieval_input_valid_flag_block == 1)
    assert np.all(result.optical.retrieval_success_flag == 1)
    assert np.isfinite(result.optical.aerosol_backscatter_block).any()


@pytest.mark.parametrize("priority", ["photon_counting", "analog"])
def test_sci002_both_valid_failed_gluing_obeys_explicit_priority(
    tmp_path: Path,
    priority: str,
) -> None:
    """Two individually valid channels are never mixed after rejected gluing."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    config["inversion"]["gluing"]["correlation_threshold"] = 1.1
    config["inversion"]["gluing"]["single_channel_priority"] = priority
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        result = process_wavelength(ds_l1, 532, altitude_m, config, logger)  # type: ignore[arg-type]

    expected = SignalSource.PHOTON_COUNTING if priority == "photon_counting" else SignalSource.ANALOG
    expected_merge = 0 if expected == SignalSource.PHOTON_COUNTING else 2
    assert np.all(result.signal_selection.source_flag_block == expected)
    assert np.all(result.glued.merge_source_flag_block == expected_merge)
    assert np.all(result.gluing.single_channel_fallback_flag_block == 1)


@pytest.mark.parametrize(
    ("invalid_case", "expected_reason"),
    [
        ("nonfinite", RetrievalInputInvalidReason.NONFINITE_SIGNAL),
        ("saturated", RetrievalInputInvalidReason.PHOTON_COUNTING_SATURATED),
        ("coverage", RetrievalInputInvalidReason.INSUFFICIENT_VERTICAL_COVERAGE),
        ("uncertainty", RetrievalInputInvalidReason.INVALID_UNCERTAINTY),
        ("correction", RetrievalInputInvalidReason.LEVEL1_CORRECTION_FAILED_OR_UNCONFIRMED),
        ("correction_unconfirmed", RetrievalInputInvalidReason.LEVEL1_CORRECTION_FAILED_OR_UNCONFIRMED),
        ("missing_saturation_diagnostic", RetrievalInputInvalidReason.SATURATION_DIAGNOSTIC_MISSING),
    ],
)
def test_sci002_invalid_pc_only_is_not_retrieved(
    tmp_path: Path,
    invalid_case: str,
    expected_reason: RetrievalInputInvalidReason,
) -> None:
    """A rejected single channel persists its reason and produces only NaN optical data."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.PC"])
    config = _config(tmp_path)
    logger = _ListLogger()
    with xr.open_dataset(level1) as opened:
        ds_l1 = opened.load()
    if invalid_case == "nonfinite":
        ds_l1["corrected_signal"].loc[dict(channel="532.PC")].values[:, 10] = np.nan
    elif invalid_case == "saturated":
        ds_l1["pc_saturation_mask"].loc[dict(channel="532.PC")].values[:, 10] = 1
    elif invalid_case == "coverage":
        config["inversion"]["molecular_fit"]["ref_alt_max_m"] = 2500.0
    elif invalid_case == "uncertainty":
        ds_l1["corrected_signal_error"].loc[dict(channel="532.PC")].values[:, 10] = np.nan
    elif invalid_case == "correction":
        ds_l1["channel_correction_success"].loc[dict(channel="532.PC")] = 0
    elif invalid_case == "correction_unconfirmed":
        ds_l1 = ds_l1.drop_vars("channel_correction_success")
    elif invalid_case == "missing_saturation_diagnostic":
        ds_l1 = ds_l1.drop_vars("pc_saturation_mask")
    altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)

    result = process_wavelength(ds_l1, 532, altitude_m, config, logger)  # type: ignore[arg-type]

    assert np.all(result.signal_selection.source_flag_block == SignalSource.INVALID)
    assert np.all(result.signal_selection.retrieval_input_valid_flag_block == 0)
    assert np.all(result.signal_selection.retrieval_input_invalid_reason_block == expected_reason)
    assert np.all(result.optical.retrieval_success_flag == 0)
    assert np.isnan(result.optical.aerosol_backscatter_block).all()


@pytest.mark.parametrize(
    ("variable", "expected_reason"),
    [
        ("corrected_signal", RetrievalInputInvalidReason.NONFINITE_SIGNAL),
        ("corrected_signal_error", RetrievalInputInvalidReason.INVALID_UNCERTAINTY),
    ],
)
def test_sci002_invalid_analog_only_is_not_retrieved(
    tmp_path: Path,
    variable: str,
    expected_reason: RetrievalInputInvalidReason,
) -> None:
    """AN-only applies the same finite signal/uncertainty minimum QA as PC."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN"])
    with xr.open_dataset(level1) as opened:
        ds_l1 = opened.load()
    ds_l1[variable].loc[dict(channel="532.AN")].values[:, 10] = np.nan
    altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)

    result = process_wavelength(
        ds_l1,
        532,
        altitude_m,
        _config(tmp_path),
        _ListLogger(),  # type: ignore[arg-type]
    )

    assert np.all(result.signal_selection.source_flag_block == SignalSource.INVALID)
    assert np.all(result.signal_selection.retrieval_input_valid_flag_block == 0)
    assert np.all(result.signal_selection.retrieval_input_invalid_reason_block == expected_reason)
    assert np.all(result.optical.retrieval_success_flag == 0)
    assert np.isnan(result.optical.aerosol_backscatter_block).all()


def test_sci002_missing_wavelength_channels_fails_in_selection_stage(tmp_path: Path) -> None:
    """No source channel fails clearly before any false optical result exists."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["355.PC"])
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        with pytest.raises(RetrievalStageError, match=r"\[selection_and_blocking\].*No channel found"):
            process_wavelength(ds_l1, 532, altitude_m, _config(tmp_path), logger)  # type: ignore[arg-type]


def test_process_wavelength_identifies_failing_stage(monkeypatch) -> None:
    """The sequencer should preserve the stable stage name and original cause."""
    def fail_selection(*_args, **_kwargs):
        raise ValueError("synthetic selection failure")

    monkeypatch.setattr(retrieval_module, "prepare_wavelength_blocks", fail_selection)

    with pytest.raises(RetrievalStageError, match=r"\[selection_and_blocking\]") as captured:
        process_wavelength(xr.Dataset(), 532, np.array([0.0]), {}, _ListLogger())  # type: ignore[arg-type]

    assert captured.value.stage == "selection_and_blocking"
    assert isinstance(captured.value.__cause__, ValueError)


def test_dataset_builder_rejects_wrong_retrieval_dtype_before_writing(tmp_path: Path) -> None:
    """A typed result with a wrong flag dtype should fail before xarray assembly."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        result = process_wavelength(ds_l1, 532, altitude_m, config, logger)  # type: ignore[arg-type]
        invalid_kfs = replace(result.kfs, branch=result.kfs.branch.astype(np.int16))
        invalid_result = replace(result, kfs=invalid_kfs)
        with pytest.raises(TypeError, match=r"kfs\.branch must have dtype int8"):
            build_level2_dataset(ds_l1, [invalid_result], altitude_m, level1, config)


def test_dataset_builder_rejects_wrong_retrieval_shape_before_writing(tmp_path: Path) -> None:
    """A typed result with an incomplete altitude vector should fail explicitly."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    logger = _ListLogger()
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        result = process_wavelength(ds_l1, 532, altitude_m, config, logger)  # type: ignore[arg-type]
        invalid_molecular = replace(result.molecular, backscatter=result.molecular.backscatter[:-1])
        invalid_result = replace(result, molecular=invalid_molecular)
        with pytest.raises(ValueError, match=r"molecular\.backscatter must have shape \(240,\)"):
            build_level2_dataset(ds_l1, [invalid_result], altitude_m, level1, config)


def test_sci002_contract_rejects_contradictory_gluing_and_source_state(tmp_path: Path) -> None:
    """An approved gluing state cannot serialize a single-channel source."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        result = process_wavelength(ds_l1, 532, altitude_m, config, _ListLogger())  # type: ignore[arg-type]
        invalid_selection = replace(
            result.signal_selection,
            source_flag=np.full_like(result.signal_selection.source_flag, SignalSource.PHOTON_COUNTING),
            source_flag_block=np.full_like(
                result.signal_selection.source_flag_block,
                SignalSource.PHOTON_COUNTING,
            ),
        )
        invalid_result = replace(result, signal_selection=invalid_selection)

        with pytest.raises(ValueError, match="gluing_success_flag=1 requires signal source glued"):
            build_level2_dataset(ds_l1, [invalid_result], altitude_m, level1, config)


def test_sci002_contract_rejects_retrieval_success_with_invalid_input(tmp_path: Path) -> None:
    """Optical success cannot coexist with a rejected retrieval input."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.PC"])
    config = _config(tmp_path)
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        result = process_wavelength(ds_l1, 532, altitude_m, config, _ListLogger())  # type: ignore[arg-type]
        invalid_gluing = replace(
            result.gluing,
            single_channel_fallback_flag=np.zeros_like(
                result.gluing.single_channel_fallback_flag
            ),
            single_channel_fallback_flag_block=np.zeros_like(
                result.gluing.single_channel_fallback_flag_block
            ),
        )
        invalid_selection = replace(
            result.signal_selection,
            source_flag=np.full_like(result.signal_selection.source_flag, SignalSource.INVALID),
            source_flag_block=np.full_like(result.signal_selection.source_flag_block, SignalSource.INVALID),
            retrieval_input_valid_flag=np.zeros_like(
                result.signal_selection.retrieval_input_valid_flag
            ),
            retrieval_input_valid_flag_block=np.zeros_like(
                result.signal_selection.retrieval_input_valid_flag_block
            ),
            retrieval_input_invalid_reason=np.full_like(
                result.signal_selection.retrieval_input_invalid_reason,
                RetrievalInputInvalidReason.NONFINITE_SIGNAL,
            ),
            retrieval_input_invalid_reason_block=np.full_like(
                result.signal_selection.retrieval_input_invalid_reason_block,
                RetrievalInputInvalidReason.NONFINITE_SIGNAL,
            ),
        )
        invalid_result = replace(
            result,
            gluing=invalid_gluing,
            signal_selection=invalid_selection,
        )

        with pytest.raises(ValueError, match="retrieval_success_flag=1 requires"):
            build_level2_dataset(ds_l1, [invalid_result], altitude_m, level1, config)


def test_retrieval_contract_rejects_missing_fields_and_free_mapping(tmp_path: Path) -> None:
    """Required dataclass fields and the dataset collection type are enforced."""
    with pytest.raises(TypeError, match="required positional arguments"):
        WavelengthRetrievalResult()  # type: ignore[call-arg]

    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    with xr.open_dataset(level1) as ds_l1:
        ds_l1.load()
        altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
        with pytest.raises(TypeError, match=r"results\[0\] must be WavelengthRetrievalResult"):
            build_level2_dataset(ds_l1, [{"wavelength": 532}], altitude_m, level1, _config(tmp_path))  # type: ignore[list-item]


def test_lebear_saves_real_kfs_mode_and_branch_flags(tmp_path: Path) -> None:
    """Level 2 output should preserve the configured KFS mode and branch traceability."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    logger = _ListLogger()

    summary = lebear.process_single_level1_file(level1, _config(tmp_path, kfs_mode="two_sided"), logger)  # type: ignore[arg-type]

    assert summary.results[0].status is ExecutionStatus.SUCCESS
    assert summary.results[1].status is ExecutionStatus.SKIPPED
    with xr.open_dataset(level2_output_path(level1)) as ds:
        assert ds.attrs["KFS_Mode"] == "two_sided"
        assert "mathematically validated" in ds.attrs["KFS_Mode_Description"]
        assert ds.attrs["fernald_implementation_version"] == "2"
        assert ds.attrs["scientific_change"] == "corrected_backward_molecular_factor_sign"
        assert "kfs_branch" in ds
        branch_values = set(np.asarray(ds["kfs_branch"].values).ravel().astype(int).tolist())
        assert 1 in branch_values
        assert 2 in branch_values
        assert 3 in branch_values
        assert int(ds["kfs_backward_valid_flag"].item()) == 1
        assert int(ds["kfs_forward_valid_flag"].item()) == 1
        assert "single_channel_fallback_flag" in ds


def test_gluing_failure_with_disabled_single_channel_fallback_persists_invalid_state(tmp_path: Path) -> None:
    """Disabled fallback writes explicit invalid input without false optical data."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    logger = _ListLogger()
    config = _config(tmp_path, allow_single_channel_fallback=False)
    config["inversion"]["gluing"]["correlation_threshold"] = 1.1

    summary = lebear.process_single_level1_file(level1, config, logger)  # type: ignore[arg-type]
    assert summary.results[0].status is ExecutionStatus.SUCCESS
    with xr.open_dataset(level2_output_path(level1)) as ds:
        assert int(ds["gluing_attempted_flag_block"].item()) == 1
        assert int(ds["gluing_success_flag_block"].item()) == 0
        assert int(ds["signal_source_flag_block"].item()) == SignalSource.INVALID
        assert int(ds["retrieval_input_valid_flag_block"].item()) == 0
        assert int(ds["retrieval_input_invalid_reason_block"].item()) == RetrievalInputInvalidReason.SINGLE_CHANNEL_FALLBACK_DISABLED
        assert int(ds["retrieval_success_flag"].item()) == 0
        assert np.isnan(ds["aerosol_backscatter_block"].values).all()


def test_gluing_failure_uses_photon_fallback_when_enabled(tmp_path: Path) -> None:
    """When photon fallback is enabled, failed gluing should be flagged and processed."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    logger = _ListLogger()
    config = _config(tmp_path, allow_single_channel_fallback=True)
    config["inversion"]["gluing"]["correlation_threshold"] = 1.1

    summary = lebear.process_single_level1_file(level1, config, logger)  # type: ignore[arg-type]

    assert summary.results[0].status is ExecutionStatus.SUCCESS
    with xr.open_dataset(level2_output_path(level1)) as ds:
        assert int(ds["gluing_success_flag"].sum()) == 0
        assert int(ds["single_channel_fallback_flag"].sum()) == ds.sizes["time"]
        assert int(ds["retrieval_input_valid_flag"].sum()) == ds.sizes["time"]
        assert int(ds["retrieval_success_flag"].sum()) == ds.sizes["block_time"]
        assert np.isfinite(ds["aerosol_backscatter_block"].values).any()


def test_process_level2_skips_output_with_current_provenance(tmp_path: Path, monkeypatch) -> None:
    """LEBEAR process_level_2 should skip only a current Level 2 product."""
    level1 = _write_level1(tmp_path / "20240101sant_level1_rcs.nc", ["532.AN", "532.PC"])
    output = level2_output_path(level1)
    output.write_text("existing", encoding="utf-8")
    logger = _ListLogger()
    calls = {"count": 0}

    def fake_process_single_level1_file(nc_file: str | Path, config: dict, logger: logging.Logger) -> ExecutionSummary:
        calls["count"] += 1
        return ExecutionSummary.from_results(
            [ExecutionResult.success("level2.complete", "should not run", input_path=nc_file)]
        )

    monkeypatch.setattr(lebear, "process_single_level1_file", fake_process_single_level1_file)
    monkeypatch.setattr(lebear, "level2_output_is_current", lambda *_args, **_kwargs: True)

    summary = lebear.process_level_2(_config(tmp_path), logger)  # type: ignore[arg-type]

    assert calls["count"] == 0
    assert any("SKIPPED" in message for message in logger.messages)
    assert summary.results[0].status is ExecutionStatus.SKIPPED


def test_process_single_level1_file_supports_utc_time_window_outputs_variant(tmp_path: Path) -> None:
    """LEBEAR should allow UTC subsetting and save the output with a variant tag."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    logger = _ListLogger()
    output = level2_output_path(level1, variant_tag="0000-0010")

    summary = lebear.process_single_level1_file(
        level1,
        _config(tmp_path),
        logger,  # type: ignore[arg-type]
        start_utc="00:00",
        stop_utc="00:10",
        output_tag="0000-0010",
    )

    assert summary.results[0].status is ExecutionStatus.SUCCESS
    with xr.open_dataset(output) as ds:
        assert ds.sizes["time"] == 2
        assert ds.attrs["LEBEAR_Time_Window_Tag"] == "0000-0010"
        assert "LEBEAR_Time_Window_UTC" in ds.attrs


def test_valid_level2_product_remains_success_when_qa_plotting_fails(tmp_path: Path, monkeypatch) -> None:
    """A plotting exception should be a separate recoverable result after product success."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    config["visualization"]["level2_qa"]["enabled"] = True
    logger = _ListLogger()

    def failing_plotter(**_kwargs):
        raise RuntimeError("synthetic plotting failure")

    monkeypatch.setattr(level2_qa, "_load_plotter", lambda: failing_plotter)

    summary = lebear.process_single_level1_file(level1, config, logger)  # type: ignore[arg-type]

    assert [result.status for result in summary.results] == [
        ExecutionStatus.SUCCESS,
        ExecutionStatus.RECOVERABLE_FAILURE,
    ]
    assert summary.results[1].stage == "level2.qa"
    assert isinstance(summary.results[1].cause, RuntimeError)
    assert level2_output_path(level1).exists()


def test_level2_qa_disabled_is_explicit_and_does_not_load_plotter(tmp_path: Path, monkeypatch) -> None:
    """Disabling QA should produce a skip without importing the plotting implementation."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    config = _config(tmp_path)
    logger = _ListLogger()

    def unexpected_plotter_load():
        pytest.fail("QA plotter must not load when QA is disabled")

    monkeypatch.setattr(level2_qa, "_load_plotter", unexpected_plotter_load)

    summary = lebear.process_single_level1_file(level1, config, logger)  # type: ignore[arg-type]

    assert [result.status for result in summary.results] == [ExecutionStatus.SUCCESS, ExecutionStatus.SKIPPED]
    assert summary.results[1].stage == "level2.qa"
    assert not (level2_output_path(level1).parent / "level2_qa").exists()


def test_atomic_level2_write_preserves_existing_product_and_removes_temporary_file(tmp_path: Path, monkeypatch) -> None:
    """A failed write should not replace a prior product or leave a partial temporary file."""
    output_path = tmp_path / "product_level2_optical.nc"
    output_path.write_text("stable product", encoding="utf-8")
    dataset = xr.Dataset({"value": (("x",), np.array([1.0]))})

    def fail_write(_self, path, **_kwargs):
        Path(path).write_text("partial product", encoding="utf-8")
        raise OSError("synthetic write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_write)

    with pytest.raises(OSError, match="synthetic write failure"):
        lebear._write_level2_atomically(dataset, output_path, {})

    assert output_path.read_text(encoding="utf-8") == "stable product"
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))
