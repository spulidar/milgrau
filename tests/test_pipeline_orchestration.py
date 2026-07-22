"""Characterization tests for pipeline orchestration and current exit behavior."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.paths import level0_output_path
from milgrau.level0 import libids
from milgrau.level1 import lipancora
from milgrau.level2 import lebear
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary, ExitCode
from milgrau.viz import liracos


class _ListLogger:
    """Capture log levels and messages without configuring global logging."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def debug(self, message: str) -> None:
        self.records.append(("DEBUG", message))

    def info(self, message: str) -> None:
        self.records.append(("INFO", message))

    def warning(self, message: str) -> None:
        self.records.append(("WARNING", message))

    def error(self, message: str) -> None:
        self.records.append(("ERROR", message))


def _minimal_level1_dataset() -> xr.Dataset:
    """Return the smallest conforming Level 1 dataset used by orchestration tests."""
    time = pd.date_range("2024-01-01", periods=2, freq="5min")
    channel = np.array(["532.AN"], dtype=object)
    altitude = np.array([0.0, 7.5, 15.0], dtype=np.float64)
    shape = (time.size, channel.size, altitude.size)
    values = np.ones(shape, dtype=np.float32)
    altitude_squared = altitude.astype(np.float32) ** 2
    return xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), values),
            "corrected_signal_error": (("time", "channel", "altitude"), values * 0.1),
            "range_corrected_signal": (("time", "channel", "altitude"), values * altitude_squared),
            "range_corrected_signal_error": (
                ("time", "channel", "altitude"),
                (values * np.float32(0.1)) * altitude_squared,
            ),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={"source_attribute": "preserved"},
    )


def test_lipancora_single_file_success_writes_contract_product(tmp_path: Path, monkeypatch) -> None:
    """A successful Level 1 orchestration should write metadata and the canonical product."""
    input_path = tmp_path / "input" / "20240101saam.nc"
    input_path.parent.mkdir()
    input_path.write_text("synthetic", encoding="utf-8")
    config = {"directories": {"processed_data": str(tmp_path / "processed")}}
    logger = _ListLogger()
    raw = _minimal_level1_dataset()
    altitude = np.asarray(raw.altitude.values)

    monkeypatch.setattr(lipancora, "load_and_prepare_level0", lambda *_: (raw, altitude))
    monkeypatch.setattr(lipancora, "apply_all_physical_corrections", lambda ds, *_: ds)
    monkeypatch.setattr(lipancora, "estimate_pbl_timeseries", lambda ds, *_: ds)
    monkeypatch.setattr(lipancora, "integrate_thermodynamics", lambda ds, *_: ds)

    result = lipancora.process_single_file((input_path, config, logger))
    output_path = tmp_path / "processed" / "2024" / "01" / "20240101saam" / "20240101saam_level1_rcs.nc"

    assert result.status is ExecutionStatus.SUCCESS
    assert result.output_path == output_path
    assert output_path.exists()
    with xr.open_dataset(output_path) as product:
        assert product["range_corrected_signal"].dims == ("time", "channel", "altitude")
        assert product["range_corrected_signal"].dtype == np.dtype("float32")
        assert product.attrs["Pipeline"] == "MILGRAU/LIPANCORA"
        assert product.attrs["Input_Level0_File"] == input_path.name
        assert product.attrs["source_attribute"] == "preserved"


def test_lipancora_single_file_failure_returns_trace_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    """A Level 0 ingestion failure should retain its stage/cause and leave no product."""
    input_path = tmp_path / "20240101saam.nc"
    input_path.write_text("invalid level 0", encoding="utf-8")
    config = {"directories": {"processed_data": str(tmp_path / "processed")}}
    logger = _ListLogger()

    def fail_ingestion(*_args, **_kwargs):
        raise RuntimeError("synthetic ingestion failure")

    monkeypatch.setattr(lipancora, "load_and_prepare_level0", fail_ingestion)

    result = lipancora.process_single_file((input_path, config, logger))

    assert result.status is ExecutionStatus.RECOVERABLE_FAILURE
    assert result.stage == "level1.ingestion"
    assert isinstance(result.cause, RuntimeError)
    assert str(result.cause) == "synthetic ingestion failure"
    assert "RuntimeError: synthetic ingestion failure" in result.traceback
    assert not list((tmp_path / "processed").rglob("*_level1_rcs.nc"))


def test_lipancora_batch_continues_after_one_file_failure(tmp_path: Path, monkeypatch) -> None:
    """Level 1 batch orchestration should aggregate results and continue after failure."""
    files = [tmp_path / "20240101saam.nc", tmp_path / "20240101sapm.nc"]
    config = {"directories": {"processed_data": str(tmp_path)}, "processing": {"incremental": False}}
    logger = _ListLogger()
    calls: list[Path] = []

    monkeypatch.setattr(lipancora, "_discover_level0_files", lambda _config: files)

    def fake_process(args) -> ExecutionResult:
        path = Path(args[0])
        calls.append(path)
        if path == files[0]:
            return ExecutionResult.failure("level1.ingestion", "first", input_path=path)
        return ExecutionResult.success("level1.complete", "second", input_path=path)

    monkeypatch.setattr(lipancora, "process_single_file", fake_process)

    summary = lipancora.process_level_1(config, logger)

    assert calls == files
    assert [result.status for result in summary.results] == [
        ExecutionStatus.RECOVERABLE_FAILURE,
        ExecutionStatus.SUCCESS,
    ]
    assert summary.exit_code is ExitCode.PARTIAL_FAILURE
    assert any(level == "WARNING" and message.startswith("[FAILED]") for level, message in logger.records)
    assert any(level == "INFO" and message.startswith("[OK]") for level, message in logger.records)


def test_lebear_batch_continues_after_one_file_failure(tmp_path: Path, monkeypatch) -> None:
    """Level 2 batch orchestration should process later files after a recoverable failure."""
    files = [tmp_path / "first_level1_rcs.nc", tmp_path / "second_level1_rcs.nc"]
    config = {"processing": {"incremental": False}}
    logger = _ListLogger()
    calls: list[Path] = []

    monkeypatch.setattr(lebear, "discover_level1_files", lambda _config: files)

    def fake_process(path, _config, _logger) -> ExecutionSummary:
        path = Path(path)
        calls.append(path)
        if path == files[0]:
            result = ExecutionResult.failure("level2.retrieval", "first", input_path=path)
        else:
            result = ExecutionResult.success("level2.complete", "second", input_path=path)
        return ExecutionSummary.from_results([result])

    monkeypatch.setattr(lebear, "process_single_level1_file", fake_process)

    summary = lebear.process_level_2(config, logger)

    assert calls == files
    assert [result.status for result in summary.results] == [
        ExecutionStatus.RECOVERABLE_FAILURE,
        ExecutionStatus.SUCCESS,
    ]
    assert summary.exit_code is ExitCode.PARTIAL_FAILURE


def test_liracos_batch_classifies_skips_and_failures_in_logs(tmp_path: Path, monkeypatch) -> None:
    """LIRACOS should log skipped files as info and structured failures as warnings."""
    files = [tmp_path / "first_level1_rcs.nc", tmp_path / "second_level1_rcs.nc"]
    for path in files:
        path.write_text("synthetic", encoding="utf-8")
    config = {"directories": {"processed_data": str(tmp_path)}}
    logger = _ListLogger()

    def fake_process(args) -> ExecutionResult:
        path = Path(args[0])
        if path == files[0]:
            return ExecutionResult.skipped("visualization.validation", "first", input_path=path)
        return ExecutionResult.failure("visualization.ingestion", "second", input_path=path)

    monkeypatch.setattr(liracos, "process_single_nc", fake_process)

    summary = liracos.process_all_level1_files(config, logger, root_dir=tmp_path)

    assert [result.status for result in summary.results] == [
        ExecutionStatus.SKIPPED,
        ExecutionStatus.RECOVERABLE_FAILURE,
    ]
    assert summary.exit_code is ExitCode.PARTIAL_FAILURE
    assert any(level == "INFO" and message.startswith("[SKIPPED]") for level, message in logger.records)
    assert any(level == "WARNING" and message.startswith("[FAILED]") for level, message in logger.records)


def test_liracos_single_file_failure_returns_trace_and_no_plot(tmp_path: Path) -> None:
    """A missing Level 1 input should retain its ingestion cause without plot output."""
    input_path = tmp_path / "missing_level1_rcs.nc"
    config = {"visualization": {"channels_to_plot": ["532.AN"], "altitude_ranges_km": [1.0]}}
    logger = _ListLogger()

    result = liracos.process_single_nc((input_path, config, tmp_path, logger))

    assert result.status is ExecutionStatus.RECOVERABLE_FAILURE
    assert result.stage == "visualization.ingestion"
    assert isinstance(result.cause, FileNotFoundError)
    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("*.manifest.json"))


def test_libids_aggregates_success_skip_and_exception_groups(tmp_path: Path, monkeypatch) -> None:
    """LIBIDS should continue groups and report its current success/skip totals."""
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
    logger = _ListLogger()
    skipped_output = level0_output_path("20240101am", config)
    skipped_output.parent.mkdir(parents=True)
    skipped_output.write_text("existing", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(libids, "build_measurement_inventory", lambda *_: inventory)
    monkeypatch.setattr(libids, "filter_laser_shots", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(libids, "_level0_is_current", lambda meas_id, *_args: meas_id == "20240101am")

    def fake_process(meas_id, _group, _config, _logger) -> ExecutionResult:
        calls.append(meas_id)
        if meas_id == "20240101nt":
            raise RuntimeError("synthetic group failure")
        return ExecutionResult.success("level0.complete", meas_id)

    monkeypatch.setattr(libids, "process_measurement_group", fake_process)

    summary = libids.process_level_0(config, logger)

    assert set(calls) == {"20240101pm", "20240101nt"}
    assert summary.counts[ExecutionStatus.SUCCESS] == 1
    assert summary.counts[ExecutionStatus.SKIPPED] == 1
    assert summary.counts[ExecutionStatus.RECOVERABLE_FAILURE] == 1
    assert summary.exit_code is ExitCode.PARTIAL_FAILURE
    assert any(level == "WARNING" and "synthetic group failure" in message for level, message in logger.records)
    assert (
        "INFO",
        "=== LIBIDS finished: processed 1, skipped 1, total groups 3. ===",
    ) in logger.records


def test_lipancora_cli_applies_success_partial_total_and_unexpected_exit_codes(tmp_path: Path) -> None:
    """The installed-style CLI call should apply ADR-002 and log chained exceptions."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH", "")]))
    setup = """
import sys
from milgrau.cli import lipancora
class Logger:
    def info(self, message):
        print(message, file=sys.stderr)
    def warning(self, message):
        print(message, file=sys.stderr)
    def error(self, message):
        print(message, file=sys.stderr)
lipancora.load_config = lambda: {}
lipancora.setup_logger = lambda *args, **kwargs: Logger()
from milgrau.operations import ExecutionResult, ExecutionSummary
"""

    success = subprocess.run(
        [
            sys.executable,
            "-c",
            setup
            + "\nlipancora.process_level_1 = lambda *args: ExecutionSummary.from_results([ExecutionResult.success('level1', 'done')])"
            + "\nraise SystemExit(lipancora.main())",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    partial = subprocess.run(
        [
            sys.executable,
            "-c",
            setup
            + "\nlipancora.process_level_1 = lambda *args: ExecutionSummary.from_results([ExecutionResult.success('level1', 'done'), ExecutionResult.failure('level1', 'one failed')])"
            + "\nraise SystemExit(lipancora.main())",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    total_failure = subprocess.run(
        [
            sys.executable,
            "-c",
            setup
            + "\nlipancora.process_level_1 = lambda *args: ExecutionSummary.from_results([ExecutionResult.failure('level1', 'all failed')])"
            + "\nraise SystemExit(lipancora.main())",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    unexpected = subprocess.run(
        [
            sys.executable,
            "-c",
            setup
            + "\ndef fail(*args):\n"
            + "    try:\n        raise ValueError('root cause')\n"
            + "    except ValueError as exc:\n        raise RuntimeError('outer failure') from exc\n"
            + "lipancora.process_level_1 = fail\nraise SystemExit(lipancora.main())",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert success.returncode == 0
    assert partial.returncode == 1
    assert total_failure.returncode == 2
    assert unexpected.returncode == 2
    assert "ValueError: root cause" in unexpected.stderr
    assert "RuntimeError: outer failure" in unexpected.stderr
