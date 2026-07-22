"""Tests for structured LIBIDS measurement-group results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from milgrau.level0 import processing
from milgrau.operations import ExecutionStatus


def _config(tmp_path: Path) -> dict:
    return {"directories": {"processed_data": str(tmp_path / "processed")}}


def test_measurement_group_without_measurements_is_explicit_skip(tmp_path: Path) -> None:
    group = pd.DataFrame({"meas_type": ["dark_current"], "filepath": [str(tmp_path / "dark")]})

    result = processing.process_measurement_group("20240101am", group, _config(tmp_path), _NullLogger())

    assert result.status is ExecutionStatus.SKIPPED
    assert result.stage == "level0.measurements"
    assert result.metadata["save_id"] == "20240101saam"


def test_measurement_group_preserves_parse_failure_stage_and_cause(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "measurement"
    input_path.write_text("invalid raw lidar", encoding="utf-8")
    group = pd.DataFrame({"meas_type": ["measurements"], "filepath": [str(input_path)]})
    monkeypatch.setattr(processing, "fetch_group_weather", lambda *_args: {})

    def fail_parse(*_args):
        raise OSError("invalid Licel header")

    monkeypatch.setattr(processing, "parse_licel_group", fail_parse)

    result = processing.process_measurement_group("20240101am", group, _config(tmp_path), _NullLogger())

    assert result.status is ExecutionStatus.RECOVERABLE_FAILURE
    assert result.stage == "level0.parse"
    assert isinstance(result.cause, OSError)
    assert "invalid Licel header" in result.traceback


def test_measurement_group_success_keeps_level0_file_effect(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "measurement"
    input_path.write_text("raw lidar", encoding="utf-8")
    group = pd.DataFrame({"meas_type": ["measurements"], "filepath": [str(input_path)]})
    monkeypatch.setattr(processing, "fetch_group_weather", lambda *_args: {})
    monkeypatch.setattr(processing, "parse_licel_group", lambda *_args: {"tensors": {"532.AN": [[1.0]]}})

    def fake_build(**kwargs) -> None:
        Path(kwargs["netcdf_path"]).write_text("level0", encoding="utf-8")

    monkeypatch.setattr(processing, "build_level0_netcdf", fake_build)

    result = processing.process_measurement_group("20240101am", group, _config(tmp_path), _NullLogger())

    assert result.status is ExecutionStatus.SUCCESS
    assert result.stage == "level0.complete"
    assert result.output_path is not None and result.output_path.exists()
    assert result.output_path.with_suffix(result.output_path.suffix + ".provenance.json").exists()


class _NullLogger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass
