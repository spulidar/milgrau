"""Tests for current MILGRAU batch orchestration semantics."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from milgrau.level0 import libids
from milgrau.level1 import lipancora
from milgrau.level2 import lebear
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary, ExitCode
from milgrau.viz import liracos


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def test_level1_batch_continues_after_one_file_error(tmp_path: Path, monkeypatch) -> None:
    files = [tmp_path / "20240101saam.nc", tmp_path / "20240101sapm.nc"]
    config = {"directories": {"processed_data": str(tmp_path)}}
    calls: list[Path] = []

    monkeypatch.setattr(lipancora, "validate_level1_config", lambda _config: None)
    monkeypatch.setattr(lipancora, "_discover_level0_files", lambda _config: files)
    monkeypatch.setattr(lipancora, "_files_requiring_level1", lambda discovered, _config, _logger: (discovered, []))

    def fake_process(args) -> ExecutionResult:
        path = Path(args[0])
        calls.append(path)
        if path == files[0]:
            return ExecutionResult.failure("level1.ingestion", "first failed", input_path=path)
        return ExecutionResult.success("level1.complete", "second succeeded", input_path=path, metadata={"channel_count": 1})

    monkeypatch.setattr(lipancora, "process_single_file", fake_process)

    summary = lipancora.process_level_1(config, _logger("test.orchestration.l1"))

    assert calls == files
    assert [result.status for result in summary.results] == [ExecutionStatus.ERROR, ExecutionStatus.OK]
    assert summary.exit_code is ExitCode.ERROR


def test_level2_batch_continues_after_one_file_error(tmp_path: Path, monkeypatch) -> None:
    files = [
        tmp_path / "20240101saam_level1_rcs.nc",
        tmp_path / "20240101sapm_level1_rcs.nc",
    ]
    config = {"processing": {"incremental": False}}
    calls: list[Path] = []

    monkeypatch.setattr(lebear, "discover_level1_files", lambda _config: files)

    def fake_process(path, _config, _logger, **_kwargs) -> ExecutionSummary:
        path = Path(path)
        calls.append(path)
        if path == files[0]:
            result = ExecutionResult.failure("level2.retrieval", "first failed", input_path=path)
        else:
            result = ExecutionResult.success("level2.complete", "second succeeded", input_path=path)
        return ExecutionSummary.from_results([result])

    monkeypatch.setattr(lebear, "process_single_level1_file", fake_process)

    summary = lebear.process_level_2(config, _logger("test.orchestration.l2"))

    assert calls == files
    assert [result.status for result in summary.results] == [ExecutionStatus.ERROR, ExecutionStatus.OK]
    assert summary.exit_code is ExitCode.ERROR


def _visualization_config(tmp_path: Path) -> dict:
    return {
        "directories": {"processed_data": str(tmp_path)},
        "processing": {"incremental": False},
        "visualization": {
            "output_format": "webp",
            "dpi": 120,
            "altitude_ranges_km": [5.0],
            "channels_to_plot": ["532.AN"],
            "quicklook": {
                "show_pbl": True,
                "show_tropopause": True,
                "mean_profile_smooth_bins": 5,
                "max_time_gap_minutes": 10.0,
                "missing_data_color": "lightgray",
                "colormap": "jet",
            },
        },
    }


def test_liracos_batch_aggregates_skip_and_error(tmp_path: Path, monkeypatch) -> None:
    files = [
        tmp_path / "20240101saam_level1_rcs.nc",
        tmp_path / "20240101sapm_level1_rcs.nc",
    ]
    for path in files:
        path.write_text("synthetic", encoding="utf-8")

    def fake_process(args) -> ExecutionResult:
        path = Path(args[0])
        if path == files[0]:
            return ExecutionResult.skipped("visualization.incremental", "already current", input_path=path)
        return ExecutionResult.failure("visualization.ingestion", "invalid product", input_path=path)

    monkeypatch.setattr(liracos, "process_single_nc", fake_process)

    summary = liracos.process_all_level1_files(
        _visualization_config(tmp_path),
        _logger("test.orchestration.viz"),
        root_dir=tmp_path,
    )

    assert [result.status for result in summary.results] == [ExecutionStatus.SKIPPED, ExecutionStatus.ERROR]
    assert summary.exit_code is ExitCode.ERROR


def test_liracos_invalid_filename_returns_structured_error(tmp_path: Path) -> None:
    input_path = tmp_path / "not-a-product.nc"
    result = liracos.process_single_nc(
        (input_path, _visualization_config(tmp_path), tmp_path, _logger("test.orchestration.viz.invalid"))
    )

    assert result.status is ExecutionStatus.ERROR
    assert result.stage == "visualization.initialize"
    assert result.metadata["save_id"] == "-"


def test_libids_aggregates_ok_skip_and_error_groups(tmp_path: Path, monkeypatch) -> None:
    group_ids = ["20240101am", "20240101pm", "20240101nt"]
    inventory = pd.DataFrame(
        {
            "meas_id": group_ids,
            "meas_type": ["measurements"] * 3,
            "filepath": [str(tmp_path / name) for name in group_ids],
        }
    )
    config = {
        "directories": {"raw_data": str(tmp_path / "raw"), "processed_data": str(tmp_path / "processed")},
        "processing": {"incremental": True},
    }
    calls: list[str] = []
    resolved = SimpleNamespace(
        acquisition_qa=SimpleNamespace(laser_shot_tolerance_fraction=0.002, licel_header_time_jitter_s=1.0)
    )

    monkeypatch.setattr(libids, "validate_level0_config", lambda _config: None)
    monkeypatch.setattr(libids, "resolve_level0_config", lambda _config: resolved)
    monkeypatch.setattr(libids, "build_measurement_inventory", lambda *_args, **_kwargs: inventory)
    monkeypatch.setattr(libids, "filter_laser_shots", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(libids, "incremental_enabled", lambda _config: True)
    monkeypatch.setattr(libids, "_level0_is_current", lambda meas_id, *_args: meas_id == "20240101am")

    def fake_process(meas_id, _group, _config, _logger) -> ExecutionResult:
        calls.append(meas_id)
        if meas_id == "20240101nt":
            raise RuntimeError("synthetic group failure")
        return ExecutionResult.success("level0.complete", meas_id)

    monkeypatch.setattr(libids, "process_measurement_group", fake_process)

    summary = libids.process_level_0(config, _logger("test.orchestration.l0"))

    assert set(calls) == {"20240101pm", "20240101nt"}
    assert summary.counts == {
        ExecutionStatus.OK: 1,
        ExecutionStatus.SKIPPED: 1,
        ExecutionStatus.ERROR: 1,
    }
    assert summary.exit_code is ExitCode.ERROR
