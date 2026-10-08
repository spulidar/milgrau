"""Tests for current MILGRAU batch orchestration semantics."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.level0 import libids
from milgrau.level1 import lipancora
from milgrau.level2 import lebear
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary, ExitCode


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def test_level1_batch_continues_after_one_file_error(tmp_path: Path, monkeypatch) -> None:
    files = [tmp_path / "spu_20240101-0000Z_20240101-0100Z_L0.nc", tmp_path / "spu_20240101-0200Z_20240101-0300Z_L0.nc"]
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
        tmp_path / "spu_20240101-0000Z_20240101-0100Z_L1.nc",
        tmp_path / "spu_20240101-0200Z_20240101-0300Z_L1.nc",
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


def test_libids_aggregates_ok_skip_and_error_groups(tmp_path: Path, monkeypatch) -> None:
    session_ids = ["spu_20231231-2200Z_20231231-2300Z", "spu_20240101-0000Z_20240101-0100Z", "spu_20240101-0200Z_20240101-0300Z"]
    inventory = pd.DataFrame(
        {
            "session_id": session_ids,
            "meas_type": ["measurements"] * 3,
            "filepath": [str(tmp_path / name) for name in session_ids],
        }
    )
    config = {
        "directories": {"raw_data": str(tmp_path / "raw"), "processed_data": str(tmp_path / "processed")},
        "processing": {"incremental": True},
        "_station_catalog": {"station": {"id": "spu"}},
    }
    calls: list[str] = []
    resolved = SimpleNamespace(
        acquisition_qa=SimpleNamespace(laser_shot_tolerance_fraction=0.002, licel_header_time_jitter_s=1.0)
    )

    monkeypatch.setattr(libids, "validate_level0_config", lambda _config: None)
    monkeypatch.setattr(libids, "resolve_level0_config", lambda _config: resolved)
    monkeypatch.setattr(libids, "build_session_inventory", lambda *_args, **_kwargs: inventory)
    monkeypatch.setattr(libids, "filter_laser_shots", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(libids, "incremental_enabled", lambda _config: True)
    monkeypatch.setattr(libids, "_level0_is_current", lambda session_id, *_args: session_id == "spu_20231231-2200Z_20231231-2300Z")

    def fake_process(session_id, _group, _config, _logger) -> ExecutionResult:
        calls.append(session_id)
        if session_id == "spu_20240101-0200Z_20240101-0300Z":
            raise RuntimeError("synthetic group failure")
        return ExecutionResult.success("level0.complete", session_id)

    monkeypatch.setattr(libids, "process_session_group", fake_process)

    summary = libids.process_level_0(config, _logger("test.orchestration.l0"))

    assert set(calls) == {"spu_20240101-0000Z_20240101-0100Z", "spu_20240101-0200Z_20240101-0300Z"}
    assert summary.counts == {
        ExecutionStatus.OK: 1,
        ExecutionStatus.SKIPPED: 1,
        ExecutionStatus.ERROR: 1,
    }
    assert summary.exit_code is ExitCode.ERROR



def test_libids_resolves_segment_scc_context_from_decoded_utc_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "spu_20250101-0000Z_20250101-0100Z_L0.nc"
    ds = xr.Dataset(
        data_vars={
            "channel_string": (("channels",), np.array(["532.AN"], dtype=object)),
            "Segment_Label": (("segments",), np.array(["seg00"], dtype=object)),
            "Segment_Regime": (("segments",), np.array(["night"], dtype=object)),
            "Segment_Start_Time_UTC": (("segments",), np.array([1735689600], dtype=np.int64)),
        }
    )
    ds["Segment_Start_Time_UTC"].attrs.update(
        {
            "units": "seconds since 1970-01-01 00:00:00 UTC",
            "calendar": "standard",
        }
    )
    ds.to_netcdf(output)

    observed: dict[str, object] = {}

    def fake_context(_config, measurement_time, available_channels, *, mode=None):
        observed["measurement_time"] = measurement_time
        observed["channels"] = list(available_channels)
        observed["mode"] = mode
        return {
            "scc_available": True,
            "scc_export_ready": True,
            "scc_channels": ["532.AN"],
            "channel_ids": {"532.AN": 1},
            "lr_input": {},
        }

    monkeypatch.setattr(libids, "resolve_station_context", fake_context)
    contexts = libids._resolve_expected_scc_contexts(
        "spu_20250101-0000Z_20250101-0100Z",
        pd.DataFrame(),
        {"_station_catalog": {}},
        output,
    )

    assert len(contexts) == 1
    assert contexts[0][0] == "night"
    assert contexts[0][1]["source_segments"] == "seg00"
    assert contexts[0][1]["source_segment_count"] == 1
    assert contexts[0][1]["contains_time_gaps"] is False
    assert observed["mode"] == "night"
    assert observed["channels"] == ["532.AN"]
    stamp = pd.Timestamp(observed["measurement_time"])
    assert stamp == pd.Timestamp("2025-01-01T00:00:00Z")



def test_libids_groups_repeated_day_segments_into_one_expected_scc_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "spu_20250101-1200Z_20250102-1200Z_L0.nc"
    ds = xr.Dataset(
        data_vars={
            "channel_string": (
                ("channels",),
                np.array(["532.AN"], dtype=object),
            ),
            "Segment_Label": (
                ("segments",),
                np.array(["seg00", "seg01", "seg02"], dtype=object),
            ),
            "Segment_Regime": (
                ("segments",),
                np.array(["day", "night", "day"], dtype=object),
            ),
            "Segment_Start_Time_UTC": (
                ("segments",),
                np.array(
                    [1735732800, 1735754400, 1735786800],
                    dtype=np.int64,
                ),
            ),
        }
    )
    ds["Segment_Start_Time_UTC"].attrs.update(
        {
            "units": "seconds since 1970-01-01 00:00:00 UTC",
            "calendar": "standard",
        }
    )
    ds.to_netcdf(output)

    calls: list[str] = []

    def fake_context(_config, measurement_time, available_channels, *, mode=None):
        del measurement_time, available_channels
        calls.append(str(mode))
        return {
            "scc_available": True,
            "scc_export_ready": True,
            "scc_channels": ["532.AN"],
            "channel_ids": {"532.AN": 1},
            "lr_input": {},
        }

    monkeypatch.setattr(libids, "resolve_station_context", fake_context)
    contexts = libids._resolve_expected_scc_contexts(
        "spu_20250101-1200Z_20250102-1200Z",
        pd.DataFrame(),
        {"_station_catalog": {}},
        output,
    )

    assert [name for name, _context in contexts] == ["day", "night"]
    day = contexts[0][1]
    night = contexts[1][1]
    assert day["source_segments"] == "seg00,seg02"
    assert day["source_segment_count"] == 2
    assert day["contains_time_gaps"] is True
    assert night["source_segments"] == "seg01"
    assert night["source_segment_count"] == 1
    assert night["contains_time_gaps"] is False
    assert calls == ["day", "night"]
