"""Incremental provenance tests for Level 2 QA visual products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from milgrau.level2 import qa
from milgrau.operations import ExecutionStatus
from milgrau.provenance import provenance_manifest_path


class _NullLogger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass


def _write_netcdf(path: Path) -> Path:
    xr.Dataset({"value": (("x",), np.array([1.0], dtype=np.float32))}).to_netcdf(path)
    return path


def _config() -> dict:
    return {
        "processing": {"incremental": True},
        "visualization": {
            "output_format": "png",
            "dpi": 80,
            "level2_qa": {"enabled": True, "generate_gluing_qa": True},
        },
    }


def test_level2_qa_reuses_only_an_intact_provenance_set(tmp_path: Path, monkeypatch) -> None:
    level1 = _write_netcdf(tmp_path / "sample_level1_rcs.nc")
    level2 = _write_netcdf(tmp_path / "sample_level2_optical.nc")
    calls = {"count": 0}

    def fake_plotter(**kwargs) -> list[Path]:
        calls["count"] += 1
        path = Path(kwargs["output_folder"]) / "QA_sample.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"plot {calls['count']}".encode())
        return [path]

    monkeypatch.setattr(qa, "_load_plotter", lambda: fake_plotter)
    config = _config()

    first = qa.generate_level2_qa(level1, level2, config, _NullLogger(), root_dir=tmp_path)
    second = qa.generate_level2_qa(level1, level2, config, _NullLogger(), root_dir=tmp_path)
    plot_path = tmp_path / "level2_qa" / "QA_sample.png"

    assert first.status is ExecutionStatus.SUCCESS
    assert second.status is ExecutionStatus.SKIPPED
    assert calls["count"] == 1
    assert provenance_manifest_path(plot_path).exists()

    plot_path.write_bytes(b"")
    regenerated = qa.generate_level2_qa(level1, level2, config, _NullLogger(), root_dir=tmp_path)

    assert regenerated.status is ExecutionStatus.SUCCESS
    assert calls["count"] == 2


def test_level2_qa_relevant_config_change_invalidates(tmp_path: Path, monkeypatch) -> None:
    level1 = _write_netcdf(tmp_path / "sample_level1_rcs.nc")
    level2 = _write_netcdf(tmp_path / "sample_level2_optical.nc")
    calls = {"count": 0}

    def fake_plotter(**kwargs) -> list[Path]:
        calls["count"] += 1
        path = Path(kwargs["output_folder"]) / "QA_sample.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"plot")
        return [path]

    monkeypatch.setattr(qa, "_load_plotter", lambda: fake_plotter)
    config = _config()
    qa.generate_level2_qa(level1, level2, config, _NullLogger(), root_dir=tmp_path)
    config["visualization"]["dpi"] = 120

    result = qa.generate_level2_qa(level1, level2, config, _NullLogger(), root_dir=tmp_path)

    assert result.status is ExecutionStatus.SUCCESS
    assert calls["count"] == 2
