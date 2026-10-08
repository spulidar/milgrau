"""Tests for structured LIBIDS session-processing results."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from milgrau.level0 import processing
from milgrau.operations import ExecutionStatus

SESSION_ID = "spu_20240101-1200Z_20240101-1205Z"


def _config(tmp_path: Path) -> dict:
    return {
        "directories": {"processed_data": str(tmp_path / "processed")},
        "_station_catalog": {"station": {"id": "spu"}},
    }


def _logger() -> logging.Logger:
    logger = logging.getLogger("test.level0.processing_results")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def test_session_without_measurements_is_explicit_skip(tmp_path: Path) -> None:
    group = pd.DataFrame({"meas_type": ["dark_current"], "filepath": [str(tmp_path / "dark")]})

    result = processing.process_session_group(SESSION_ID, group, _config(tmp_path), _logger())

    assert result.status is ExecutionStatus.SKIPPED
    assert result.stage == "level0.measurements"
    assert result.metadata["session_id"] == SESSION_ID


def test_session_preserves_parse_failure_stage_and_cause(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "measurement"
    input_path.write_text("invalid raw lidar", encoding="utf-8")
    group = pd.DataFrame({"meas_type": ["measurements"], "filepath": [str(input_path)]})
    monkeypatch.setattr(processing, "fetch_group_weather", lambda *_args: {})

    def fail_parse(*_args):
        raise OSError("invalid Licel header")

    monkeypatch.setattr(processing, "parse_licel_group", fail_parse)

    result = processing.process_session_group(SESSION_ID, group, _config(tmp_path), _logger())

    assert result.status is ExecutionStatus.ERROR
    assert result.stage == "level0.parse"
    assert isinstance(result.cause, OSError)
    assert "invalid Licel header" in result.traceback


def test_session_success_keeps_only_level0_file_effect(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "measurement"
    input_path.write_text("raw lidar", encoding="utf-8")
    group = pd.DataFrame(
        {"meas_type": ["measurements"], "filepath": [str(input_path)]}
    )
    monkeypatch.setattr(processing, "fetch_group_weather", lambda *_args: {})
    monkeypatch.setattr(
        processing,
        "parse_licel_group",
        lambda *_args: {"tensors": {"532.AN": [[1.0]]}, "channels": ["532.AN"]},
    )
    monkeypatch.setattr(
        processing,
        "_resolve_group_station_config",
        lambda _group, lidar, config, _logger: (dict(config), dict(lidar), {}),
    )

    def fake_build(**kwargs) -> None:
        assert kwargs["session_id"] == SESSION_ID
        Path(kwargs["netcdf_path"]).write_text("level0", encoding="utf-8")

    monkeypatch.setattr(processing, "build_level0_netcdf", fake_build)
    monkeypatch.setattr(processing, "write_netcdf_provenance", lambda *_args, **_kwargs: {})

    result = processing.process_session_group(SESSION_ID, group, _config(tmp_path), _logger())

    assert result.status is ExecutionStatus.OK
    assert result.stage == "level0.complete"
    assert result.metadata["session_id"] == SESSION_ID
    assert result.output_path is not None and result.output_path.exists()



def test_session_success_execution_metadata_remains_json_scalar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "measurement"
    input_path.write_text("raw lidar", encoding="utf-8")
    group = pd.DataFrame(
        {"meas_type": ["measurements"], "filepath": [str(input_path)]}
    )
    lidar = {"tensors": {"532.AN": [[1.0]]}, "channels": ["532.AN"]}
    station_context = {
        "profile_id": "spu-test-profile",
        "calibration_id": "spu-test-calibration",
        "solar_regimes_present": ["day", "night"],
        "solar_segments_present": ["seg00", "seg01"],
        "scc_available": False,
        "lr_input": {},
    }
    effective_config = {
        **_config(tmp_path),
        "_resolved_station": station_context,
    }

    monkeypatch.setattr(processing, "parse_licel_group", lambda *_args: lidar)
    monkeypatch.setattr(processing, "_annotate_solar_context", lambda frame, _config: frame)
    monkeypatch.setattr(
        processing,
        "_resolve_group_station_config",
        lambda _group, _lidar, _config, _logger: (
            effective_config,
            lidar,
            station_context,
        ),
    )
    monkeypatch.setattr(processing, "fetch_group_weather", lambda *_args: {})

    def fake_build(**kwargs) -> None:
        Path(kwargs["netcdf_path"]).write_text("level0", encoding="utf-8")

    monkeypatch.setattr(processing, "build_level0_netcdf", fake_build)
    monkeypatch.setattr(processing, "write_netcdf_provenance", lambda *_args, **_kwargs: {})
    scc_path = tmp_path / "processed" / "scc-a.nc"
    monkeypatch.setattr(
        processing,
        "_write_scc_exports",
        lambda **_kwargs: [scc_path],
    )

    result = processing.process_session_group(
        SESSION_ID,
        group,
        _config(tmp_path),
        _logger(),
    )

    assert result.status is ExecutionStatus.OK
    assert result.metadata["solar_regimes"] == "day,night"
    assert result.metadata["solar_segments"] == "seg00,seg01"
    assert result.metadata["scc_export_count"] == 1
    assert result.metadata["scc_export_paths"] == str(scc_path)
    assert all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in result.metadata.values()
    )
