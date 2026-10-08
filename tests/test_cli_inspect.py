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
    assert "PRODUCT L1" in output
    assert "File        : demo.nc" in output
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
        "_station_catalog": {"station": {"id": "spu", "timezone": "America/Sao_Paulo"}},
    }


def test_expand_date_finds_all_processing_levels(tmp_path) -> None:
    config = _selection_config(tmp_path)
    first = "spu_20240620-0300Z_20240620-0900Z"
    second = "spu_20240620-2100Z_20240621-0500Z"
    expected_names: list[str] = []

    for session_id in (first, second):
        session_dir = tmp_path / "processed" / "spu" / "2024" / "06" / session_id
        session_dir.mkdir(parents=True)
        for level in ("L0", "L1", "L2"):
            name = f"{session_id}_{level}.nc"
            (session_dir / name).write_bytes(b"placeholder")
            expected_names.append(name)

    paths = _expand_inputs([["20240620"]], config)

    assert [path.name for path in paths] == sorted(expected_names)


def test_expand_session_id_finds_only_available_related_levels(tmp_path) -> None:
    config = _selection_config(tmp_path)
    session_id = "spu_20240620-2100Z_20240621-0500Z"
    session_dir = tmp_path / "processed" / "spu" / "2024" / "06" / session_id
    session_dir.mkdir(parents=True)

    expected_names = [
        f"{session_id}_L0.nc",
        f"{session_id}_L1.nc",
    ]
    for name in expected_names:
        (session_dir / name).write_bytes(b"placeholder")

    paths = _expand_inputs([[session_id]], config)

    assert [path.name for path in paths] == expected_names



def test_inspect_product_prints_human_session_summary(tmp_path, capsys) -> None:
    session_id = "spu_20250511-0012Z_20250511-0737Z"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    path = session_dir / f"{session_id}_L1.nc"
    figures = session_dir / "figures"
    figures.mkdir()
    (figures / f"{session_id}_L1_MeanRCS.webp").write_bytes(b"figure")
    (session_dir / f"{session_id}_L0.nc").write_bytes(b"placeholder")

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (
                ("time", "channel", "altitude"),
                np.ones((2, 1, 3), dtype=np.float32),
            ),
            "range_corrected_signal": (
                ("time", "channel", "altitude"),
                np.ones((2, 1, 3), dtype=np.float32),
            ),
            "Segment_Label": (("segments",), np.array(["seg00"], dtype=object)),
            "Segment_Regime": (("segments",), np.array(["night"], dtype=object)),
        },
        coords={
            "time": np.array(
                ["2025-05-11T00:12:00", "2025-05-11T07:37:00"],
                dtype="datetime64[m]",
            ),
            "channel": np.array(["532.AN"]),
            "altitude": np.array([7.5, 15.0, 22.5]),
        },
        attrs={
            "Session_ID": session_id,
            "timezone": "America/Sao_Paulo",
            "Processing_level": "Level 1 synthetic product",
        },
    )
    ds.to_netcdf(path)

    inspect_product(path, max_vars=10)

    output = capsys.readouterr().out
    assert "SESSION" in output
    assert "Human interval" not in output
    assert "Local time" in output
    assert session_id in output
    assert "10/05/2025 21:12" in output
    assert "11/05/2025 04:37" in output
    assert "7h25" in output
    assert "highest L1" in output
    assert "night" in output
    assert "seg00 night" in output
    assert "Figures        : 1 files" in output
    assert f"{session_id}_L1_MeanRCS.webp" not in output



def test_inspect_uses_explicit_station_timezone_even_when_product_has_no_timezone(
    tmp_path,
    capsys,
) -> None:
    session_id = "spu_20240620-2103Z_20240621-0848Z"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    path = session_dir / f"{session_id}_L2.nc"
    ds = xr.Dataset(
        data_vars={
            "Segment_Label": (("segments",), np.array(["seg00", "seg01", "seg02"], dtype=object)),
            "Segment_Regime": (("segments",), np.array(["day", "night", "day"], dtype=object)),
        },
        attrs={"Session_ID": session_id},
    )
    ds.to_netcdf(path)

    inspect_product(
        path,
        max_vars=10,
        timezone_name="America/Sao_Paulo",
    )

    output = capsys.readouterr().out
    assert "20/06/2024 18:03" in output
    assert "21/06/2024 05:48" in output
    assert "Time zone      : America/Sao_Paulo" in output
    assert "Solar sequence : day → night → day" in output


def test_inspect_datetime_coordinate_preview_is_readable(tmp_path, capsys) -> None:
    path = tmp_path / "time_demo.nc"
    ds = xr.Dataset(
        coords={
            "time": np.array(
                [
                    "2024-06-20T21:03:48",
                    "2024-06-20T22:03:48",
                    "2024-06-20T23:03:48",
                    "2024-06-21T00:03:48",
                    "2024-06-21T01:03:48",
                    "2024-06-21T02:03:48",
                    "2024-06-21T03:03:48",
                ],
                dtype="datetime64[s]",
            )
        }
    )
    ds.to_netcdf(path)

    inspect_product(path, max_vars=10)

    output = capsys.readouterr().out
    assert "2024-06-20T21:03:48" in output
    assert "171" not in output.split("preview:", 1)[1].split("\n", 1)[0]
