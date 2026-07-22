"""Tests for deterministic product provenance and safe incremental reuse."""

from __future__ import annotations

from pathlib import Path

import milgrau.provenance as provenance_module

from milgrau.provenance import (
    build_product_provenance,
    output_is_current,
    provenance_manifest_path,
    relevant_configuration,
    write_provenance_manifest,
)


def _config() -> dict:
    return {
        "directories": {"raw_data": "raw", "processed_data": "processed"},
        "processing": {
            "incremental": True,
            "console_level": "INFO",
            "laser_shot_tolerance_fraction": 0.002,
            "dark_current_max_association_hours": 12.0,
        },
        "site": {"timezone": "America/Sao_Paulo", "station_altitude_m": 760.0},
        "physics": {
            "vertical_resolution_m": 7.5,
            "speed_of_light_m_s": 299792458.0,
            "pbl_min_search_m": 500.0,
            "channels": {
                "532.AN": {"deadtime_us": 0.0, "bin_shift_bins": -12, "background_offset": 0.0}
            },
        },
        "hardware": {"name_to_id": {"532.AN": 722}},
        "radiosonde": {"station_id": "83779", "cache_dir": "cache-a"},
        "inversion": {
            "enabled": True,
            "wavelengths_to_process": [532],
            "monte_carlo_iterations": 10,
            "cloud_screening": {"enabled": True},
        },
        "visualization": {
            "output_format": "png",
            "dpi": 100,
            "altitude_ranges_km": [5.0],
            "channels_to_plot": ["532.AN"],
            "quicklook": {
                "max_time_gap_minutes": 10,
                "colormap": "viridis",
                "show_pbl": True,
            },
            "level2_qa": {"enabled": True, "generate_gluing_qa": True, "max_altitude_km": 30.0},
        },
    }


def test_fingerprint_is_deterministic_and_input_order_independent(tmp_path: Path) -> None:
    first_input = tmp_path / "a.raw"
    second_input = tmp_path / "b.raw"
    first_input.write_bytes(b"first")
    second_input.write_bytes(b"second")

    first = build_product_provenance("level0", [first_input, second_input], _config())
    second = build_product_provenance("level0", [second_input, first_input], _config())

    assert first.fingerprint == second.fingerprint
    assert first.payload == second.payload


def test_relevant_input_and_configuration_changes_invalidate(tmp_path: Path) -> None:
    input_path = tmp_path / "level0.nc"
    input_path.write_bytes(b"original")
    config = _config()
    original = build_product_provenance("level1", [input_path], config)

    config["physics"]["pbl_min_search_m"] = 600.0
    changed_config = build_product_provenance("level1", [input_path], config)
    input_path.write_bytes(b"changed")
    changed_input = build_product_provenance("level1", [input_path], config)

    assert changed_config.fingerprint != original.fingerprint
    assert changed_input.fingerprint != changed_config.fingerprint


def test_operational_and_dormant_configuration_is_excluded(tmp_path: Path) -> None:
    input_path = tmp_path / "level1.nc"
    input_path.write_bytes(b"level 1")
    config = _config()
    original = build_product_provenance("level2", [input_path], config)

    config["directories"]["processed_data"] = "another-location"
    config["processing"]["console_level"] = "DEBUG"
    config["inversion"]["enabled"] = False
    config["inversion"]["cloud_screening"]["enabled"] = False
    unchanged = build_product_provenance("level2", [input_path], config)

    assert unchanged.fingerprint == original.fingerprint


def test_individual_quicklook_excludes_other_channel_selection(tmp_path: Path) -> None:
    input_path = tmp_path / "level1.nc"
    input_path.write_bytes(b"level 1")
    config = _config()
    original = build_product_provenance(
        "liracos.quicklook",
        [input_path],
        config,
        variant={"channel": "532.AN", "max_altitude_km": 5.0},
    )

    config["visualization"]["channels_to_plot"].append("355.AN")
    config["visualization"]["altitude_ranges_km"].append(15.0)
    unchanged = build_product_provenance(
        "liracos.quicklook",
        [input_path],
        config,
        variant={"channel": "532.AN", "max_altitude_km": 5.0},
    )

    assert unchanged.fingerprint == original.fingerprint


def test_old_output_without_manifest_is_stale(tmp_path: Path) -> None:
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"legacy output")
    expected = build_product_provenance("level1", [input_path], _config())

    assert not output_is_current(output_path, expected)


def test_current_output_requires_matching_output_digest(tmp_path: Path) -> None:
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"complete output")
    expected = build_product_provenance("level1", [input_path], _config())
    write_provenance_manifest(output_path, expected)

    assert output_is_current(output_path, expected)
    output_path.write_bytes(b"tampered output")
    assert not output_is_current(output_path, expected)


def test_empty_output_is_incomplete_even_with_an_old_sidecar(tmp_path: Path) -> None:
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"once complete")
    expected = build_product_provenance("level1", [input_path], _config())
    write_provenance_manifest(output_path, expected)
    output_path.write_bytes(b"")

    assert provenance_manifest_path(output_path).exists()
    assert not output_is_current(output_path, expected)


def test_integrity_contract_can_reject_a_digest_matching_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"not really netcdf")
    expected = build_product_provenance("level1", [input_path], _config())
    write_provenance_manifest(output_path, expected)

    assert not output_is_current(output_path, expected, integrity_check=lambda _path: False)


def test_documented_product_subsets_are_available() -> None:
    config = _config()

    assert "console_level" not in relevant_configuration("level0", config)["processing"]
    assert "cache_dir" not in relevant_configuration("level1", config)["radiosonde"]
    assert "cloud_screening" not in relevant_configuration("level2", config)["inversion"]
    assert relevant_configuration("level2", config)["scientific_algorithms"] == {
        "elastic_backscatter_inversion_method": "Klett-Fernald-Sasano",
        "integration_mode": "two_sided",
        "uncertainty_method": "Monte Carlo",
        "fernald_implementation_version": "2",
        "scientific_change": "corrected_backward_molecular_factor_sign",
    }
    assert "show_pbl" not in relevant_configuration("liracos.quicklook", config)["visualization"]["quicklook"]
    assert "max_altitude_km" not in relevant_configuration("lebear.qa", config)["visualization"]["level2_qa"]


def test_level2_algorithm_version_is_part_of_the_fingerprint(tmp_path: Path, monkeypatch) -> None:
    """Changing only the explicit Fernald identity must invalidate old Level 2 products."""
    input_path = tmp_path / "level1.nc"
    input_path.write_bytes(b"level 1")
    original = build_product_provenance("level2", [input_path], _config())
    changed_metadata = dict(original.payload["configuration"]["scientific_algorithms"])
    changed_metadata["fernald_implementation_version"] = "3"
    monkeypatch.setattr(
        provenance_module,
        "elastic_inversion_algorithm_metadata",
        lambda: changed_metadata,
    )
    changed = build_product_provenance("level2", [input_path], _config())

    assert original.fingerprint != changed.fingerprint
