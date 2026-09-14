"""Regression tests for FAIR provenance in incremental NetCDF reuse."""

from __future__ import annotations

from pathlib import Path

import milgrau.incremental as incremental


def _files(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.dat"
    output = tmp_path / "product.nc"
    source.write_text("source", encoding="utf-8")
    output.write_text("product", encoding="utf-8")
    return source, output


def test_incremental_netcdf_reuse_rejects_incomplete_provenance(tmp_path: Path, monkeypatch) -> None:
    source, output = _files(tmp_path)
    monkeypatch.setattr(incremental, "netcdf_provenance_is_complete", lambda _path: False)

    assert not incremental.output_is_current(
        output,
        [source],
        include_code=False,
        integrity_check=lambda _path: True,
    )


def test_incremental_netcdf_reuse_accepts_complete_provenance_when_other_checks_pass(tmp_path: Path, monkeypatch) -> None:
    source, output = _files(tmp_path)
    monkeypatch.setattr(incremental, "netcdf_provenance_is_complete", lambda _path: True)

    assert incremental.output_is_current(
        output,
        [source],
        include_code=False,
        integrity_check=lambda _path: True,
    )
