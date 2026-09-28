from __future__ import annotations

import numpy as np
import xarray as xr

from milgrau.cli.inspect import _expand_inputs, inspect_product, main


def _write_level1_like(path) -> None:
    ds = xr.Dataset(
        {
            "corrected_signal": (
                ("time", "channel", "altitude"),
                np.ones((2, 1, 3), dtype=np.float32),
            ),
            "range_corrected_signal": (
                ("time", "channel", "altitude"),
                np.ones((2, 1, 3), dtype=np.float32),
            ),
            "Atmospheric_Temperature_K": (
                ("altitude",),
                np.array([290.0, 289.0, 288.0], dtype=np.float32),
            ),
        },
        coords={
            "time": np.array(
                ["2024-03-01T00:00:00", "2024-03-01T00:01:00"],
                dtype="datetime64[s]",
            ),
            "channel": np.array(["532.AN"]),
            "altitude": np.array([7.5, 15.0, 22.5]),
        },
        attrs={
            "Processing_level": "Level 1 synthetic product",
            "software_name": "MILGRAU",
        },
    )
    ds["altitude"].attrs["units"] = "m"
    ds["corrected_signal"].attrs["long_name"] = "Corrected lidar signal"
    ds.to_netcdf(path)


def test_inspect_product_prints_structural_summary(tmp_path, capsys) -> None:
    path = tmp_path / "demo.nc"
    _write_level1_like(path)

    inspect_product(path, max_vars=10)

    output = capsys.readouterr().out
    assert "MILGRAU PRODUCT INSPECTOR" in output
    assert "Detected level : L1" in output
    assert "DIMENSIONS" in output
    assert "COORDINATES" in output
    assert "DATA VARIABLES" in output
    assert "GLOBAL METADATA" in output
    assert "corrected_signal" in output
    assert "time=2" in output
    assert "channel=1" in output
    assert "altitude=3" in output


def test_main_reports_missing_file(capsys, tmp_path) -> None:
    missing = tmp_path / "missing_L0.nc"

    result = main([str(missing)])

    assert result == 1
    assert "NetCDF product not found" in capsys.readouterr().out


def _selection_config(tmp_path) -> dict:
    return {
        "directories": {"processed_data": str(tmp_path / "processed")},
        "_station_catalog": {"station": {"id": "spu"}},
    }


def test_expand_date_finds_all_processing_levels(tmp_path) -> None:
    config = _selection_config(tmp_path)
    day_dir = tmp_path / "processed" / "spu" / "2024" / "06" / "20240620"
    day_dir.mkdir(parents=True)

    expected_names = [
        "20240620_spu_00_L0.nc",
        "20240620_spu_00_L1.nc",
        "20240620_spu_00_L2.nc",
        "20240620_spu_18_L0.nc",
        "20240620_spu_18_L1.nc",
        "20240620_spu_18_L2.nc",
    ]
    for name in expected_names:
        (day_dir / name).write_bytes(b"placeholder")
    (day_dir / "unrelated.nc").write_bytes(b"placeholder")

    paths = _expand_inputs([["20240620"]], config)

    assert [path.name for path in paths] == expected_names


def test_expand_measurement_id_finds_related_levels(tmp_path) -> None:
    config = _selection_config(tmp_path)
    day_dir = tmp_path / "processed" / "spu" / "2024" / "06" / "20240620"
    day_dir.mkdir(parents=True)

    expected_names = [
        "20240620_spu_18_L0.nc",
        "20240620_spu_18_L1.nc",
        "20240620_spu_18_L2.nc",
    ]
    for name in expected_names:
        (day_dir / name).write_bytes(b"placeholder")
    (day_dir / "20240620_spu_00_L0.nc").write_bytes(b"placeholder")

    paths = _expand_inputs([["20240620_spu_18"]], config)

    assert [path.name for path in paths] == expected_names
