"""Phase 12.3: disposable SCC staging must not control scientific L0 identity."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from milgrau.level0 import libids, processing
from milgrau.operations import ExecutionStatus
from milgrau.level0.scc_staging import (
    atomic_scc_package,
    pending_scc_package,
    scc_staging_root,
    validate_pending_scc_package,
    write_scc_manifest,
)


def _config(tmp_path: Path) -> dict:
    return {
        "scc": {"enabled": True},
        "_scc_catalog": {},
        "_station_catalog": {"station": {"id": "spu"}},
        "directories": {"processed_data": str(tmp_path / "processed")},
    }


def _stage(config: dict, identity: str = "20251001spu0000") -> Path:
    with atomic_scc_package(config, identity, session_id="spu_20251001-0000Z_20251001-0400Z") as work:
        (work / f"{identity}.nc").write_bytes(b"staged netcdf content")
        write_scc_manifest(
            work,
            measurement_id=identity,
            session_id="spu_20251001-0000Z_20251001-0400Z",
            regime="day",
            month_utc="202510",
            config_id=1047,
            product_ids=[3683],
            source_segments=["seg00"],
            profile_start_utc="2025-10-01T00:00:00Z",
            profile_stop_utc="2025-10-01T04:00:00Z",
            source_canonical_l0=Path("processed/canonical_L0.nc"),
            config=config,
            ancillary_name=None,
        )
    return pending_scc_package(config, identity)


def test_complete_pending_package_and_manifest_checksum(tmp_path: Path) -> None:
    config = _config(tmp_path)
    target = _stage(config)
    manifest = validate_pending_scc_package(target)
    assert target == scc_staging_root(config, "spu_20251001-0000Z_20251001-0400Z") / "20251001spu0000"
    assert manifest["scc_product_ids"] == [3683]
    assert manifest["scientific_session_id"] == "spu_20251001-0000Z_20251001-0400Z"
    assert manifest["files"]["20251001spu0000.nc"]["size_bytes"] > 0
    assert not list(target.parent.glob(".*"))
    (target / "20251001spu0000.nc").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_pending_scc_package(target)


def test_scc_manifest_rejects_long_sounding_filename_before_stage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    identity = "20241008spu1650"
    with pytest.raises(ValueError, match="rs_20241008spu16.nc"):
        with atomic_scc_package(config, identity, session_id="spu_20241008-1650Z_20241008-1800Z") as work:
            (work / f"{identity}.nc").write_bytes(b"raw bytes")
            write_scc_manifest(
                work, measurement_id=identity,
                session_id="spu_20241008-1650Z_20241008-1800Z",
                regime="day", month_utc="202410", config_id=1047,
                product_ids=[3683], source_segments=["seg00"],
                profile_start_utc="2024-10-08T16:50:00Z",
                profile_stop_utc="2024-10-08T18:00:00Z",
                source_canonical_l0=Path("processed/session_L0.nc"),
                config=config, ancillary_name=None,
                sounding_name=f"rs_{identity}.nc",
            )
    assert not pending_scc_package(config, identity, session_id="spu_20241008-1650Z_20241008-1800Z").exists()


def test_staging_failures_leave_no_visible_or_partial_packages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="simulated"):
        with atomic_scc_package(config, "20251001spu0000", session_id="spu_20251001-0000Z_20251001-0400Z") as work:
            (work / "20251001spu0000.nc").write_bytes(b"partial")
            raise RuntimeError("simulated failure")
    assert not pending_scc_package(config, "20251001spu0000", session_id="spu_20251001-0000Z_20251001-0400Z").exists()
    assert not list(scc_staging_root(config, "spu_20251001-0000Z_20251001-0400Z").glob(".*"))


def test_pending_identity_collision_must_not_replace_original_bytes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    target = _stage(config)
    digest = validate_pending_scc_package(target)["files"]["20251001spu0000.nc"]["sha256"]
    with pytest.raises(FileExistsError, match="already staged"):
        with atomic_scc_package(config, "20251001spu0000", session_id="spu_20251001-0000Z_20251001-0400Z"):
            pass
    assert validate_pending_scc_package(target)["files"]["20251001spu0000.nc"]["sha256"] == digest
    with pytest.raises(ValueError, match="Unsafe SCC Measurement_ID"):
        pending_scc_package(config, "../20251001spu0000")


def test_scientific_incrementality_ignores_deleted_or_missing_scc_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls = []
    monkeypatch.setattr(
        libids, "output_is_current",
        lambda output, paths, **kwargs: calls.append(str(output)) or True,
    )
    source = tmp_path / "licel"
    source.touch()
    group = pd.DataFrame({"filepath": [str(source)]})
    output = tmp_path / "processed" / "canonical_L0.nc"
    assert libids._level0_is_current("spu_20251001-0000Z_20251001-0400Z", group, config, output)
    _stage(config)
    assert libids._level0_is_current("spu_20251001-0000Z_20251001-0400Z", group, config, output)
    assert len(calls) == 2  # only canonical L0 checked, never SCC packages


def test_month_splitting_and_repeated_segments_keep_one_scientific_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    session = "spu_20250930-2300Z_20251001-0300Z"
    starts = pd.to_datetime([
        "2025-09-30T23:00:00Z",
        "2025-09-30T23:20:00Z",
        "2025-10-01T00:10:00Z",
        "2025-10-01T02:10:00Z",
    ])
    group = pd.DataFrame({
        "meas_type": ["measurements"] * 4,
        "start_time_utc": starts,
        "stop_time": starts + pd.Timedelta(minutes=5),
        "solar_regime": ["day", "night", "day", "day"],
        "segment_id": ["seg00", "seg01", "seg02", "seg02"],
        "_profile_index": list(range(4)),
        "filepath": [f"measurement{i}" for i in range(4)],
    })
    lidar = {
        "channels": ["532.AN"],
        "tensors": {"532.AN": np.ones((4, 7))},
        "laser_shots": np.ones((4, 1)),
        "channel_metadata": {"532.AN": {"bin_width_m": 7.5}},
    }

    def fake_context(_config, measurement_time, available_channels, *, mode=None):
        del measurement_time, available_channels
        return {
            "station_id": "spu", "scc_available": True,
            "scc_export_ready": True, "scc_channels": ["532.AN"],
            "channel_ids": {"532.AN": 4070}, "lr_input": {},
            "scc_configuration_id": 1047 if mode == "day" else 1046,
            "scc_product_ids": [3684],
        }

    monkeypatch.setattr(processing, "resolve_station_context", fake_context)
    monkeypatch.setattr(processing, "resolve_level0_config", lambda _: SimpleNamespace(
        solar_regime=SimpleNamespace(day_night_threshold_deg=-3.0),
    ))
    monkeypatch.setattr(processing, "_weather_for_measurement_rows", lambda *_: {})
    monkeypatch.setattr(processing, "select_lidar_channels", lambda source, _: source)
    from milgrau.level0 import scc_lidar_ratio

    monkeypatch.setattr(scc_lidar_ratio, "write_scc_lidar_ratio_ancillary", lambda *_: None)
    def fake_build(**kwargs):
        Path(kwargs["netcdf_path"]).write_bytes(b"not actual netcdf; export unit test")
        assert kwargs["scc_export"] is True
        assert kwargs["session_id"] == session
        assert len(kwargs["lidar_data"]["tensors"]["532.AN"]) == (
            len(kwargs["group_df"].query("meas_type == 'measurements'"))
        )
    monkeypatch.setattr(processing, "build_level0_netcdf", fake_build)
    monkeypatch.setattr(processing, "write_netcdf_provenance", lambda *_: None)
    outputs = processing._write_scc_exports(
        session, lidar, group, {}, config, logging.getLogger("test.scc.month"),
    )
    assert len(outputs) == 3
    manifests = [validate_pending_scc_package(path) for path in outputs]
    assert {m["climatology_month_utc"] for m in manifests} == {"202509", "202510"}
    assert sum(m["solar_regime"] == "day" for m in manifests) == 2
    assert all(m["scientific_session_id"] == session for m in manifests)
    assert {tuple(m["source_profile_indices"]) for m in manifests} == {
        (0,), (1,), (2, 3),
    }
    assert all(m["source_measurement_files"] for m in manifests)
    assert sorted(m["measurement_id"] for m in manifests) == [
        "20250930spu2300", "20250930spu2320", "20251001spu0010",
    ]


def test_legacy_scientific_scc_derivative_is_left_untouched(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scientific = tmp_path / "processed" / "old-session"
    scientific.mkdir(parents=True)
    original = scientific / "old-session_day_L0_scc.nc"
    original.write_bytes(b"historic SCC derivative - intentionally retained")
    _stage(config)
    assert original.read_bytes() == b"historic SCC derivative - intentionally retained"


def test_scc_only_reexport_never_writes_canonical_level0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    session = "spu_20251001-0000Z_20251001-0400Z"
    from milgrau.io.paths import level0_output_path
    original = level0_output_path(session, config)
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"untouched scientific L0")
    monkeypatch.setattr(libids, "_primary_output_satisfies_contract", lambda _: True)
    monkeypatch.setattr(processing, "parse_licel_group", lambda *_: {
        "channels": ["532.AN"], "tensors": {"532.AN": np.ones((1, 8))},
    })
    monkeypatch.setattr(
        processing, "_annotate_solar_context", lambda group, _: group,
    )
    context = {"profile_id": "test", "calibration_id": "test-calibration",
               "solar_regimes_present": ["day"], "solar_segments_present": ["seg00"]}
    monkeypatch.setattr(
        processing, "_resolve_group_station_config",
        lambda group, data, cfg, logger: ({**cfg, "_resolved_station": context}, data, context),
    )
    monkeypatch.setattr(processing, "fetch_group_weather", lambda *_: {})
    monkeypatch.setattr(
        processing, "build_level0_netcdf",
        lambda **_: pytest.fail("SCC-only re-export must not write canonical L0"),
    )
    monkeypatch.setattr(
        processing, "_write_scc_exports", lambda **_: [_stage(config)],
    )
    group = pd.DataFrame({
        "meas_type": ["measurements"],
        "filepath": ["raw-measurement"],
        "start_time_utc": [pd.Timestamp("2025-10-01T00:00:00Z")],
    })
    result = processing.process_session_group(
        session, group, config, logging.getLogger("test.scc.only"), scc_only=True,
    )
    assert result.status is ExecutionStatus.OK
    assert result.metadata["scc_export_count"] == 1
    assert original.read_bytes() == b"untouched scientific L0"


def test_profile_crossing_month_is_refused_without_touching_scientific_l0(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    group = pd.DataFrame({
        "meas_type": ["measurements"],
        "start_time_utc": [pd.Timestamp("2025-09-30T23:59:00Z")],
        "stop_time": [pd.Timestamp("2025-10-01T00:04:00Z")],
        "solar_regime": ["night"],
    })
    with pytest.raises(ValueError, match="crosses a UTC month boundary"):
        processing._write_scc_exports(
            "spu_20250930-2359Z_20251001-0004Z", {"channels": []},
            group, {}, config, logging.getLogger("test.scc.boundary"),
        )
    assert not (tmp_path / ".cache/scc").exists()
