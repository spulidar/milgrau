"""Tests for portable thermodynamic source identities in derived products."""

from __future__ import annotations

from milgrau.provenance import thermodynamic_source_provenance


def test_era5_identity_uses_configured_dataset_and_doi_not_cache_name() -> None:
    attrs = {
        "thermodynamic_profile_source_type": "era5",
        "thermodynamic_profile_doi": "10.24381/cds.bd0915c6",
        "thermodynamic_profile_source": "Copernicus Climate Change Service ERA5 pressure-level reanalysis",
        "cache_file": "machine-local-cache.nc",
    }
    config = {
        "level1": {
            "atmosphere": {
                "era5": {"dataset": "reanalysis-era5-pressure-levels"}
            }
        }
    }

    result = thermodynamic_source_provenance(attrs, config)

    assert result["thermodynamic_profile_provider"] == (
        "copernicus_climate_change_service"
    )
    assert result["thermodynamic_profile_product"] == (
        "reanalysis-era5-pressure-levels"
    )
    assert result["thermodynamic_profile_version_or_release"] == (
        "doi:10.24381/cds.bd0915c6"
    )
    assert result["thermodynamic_profile_source_id"] == (
        "copernicus_climate_change_service/"
        "reanalysis-era5-pressure-levels/doi:10.24381/cds.bd0915c6"
    )
    assert "cache" not in result["thermodynamic_profile_source_id"]


def test_radiosonde_identity_is_service_family_not_station_or_cache() -> None:
    attrs = {
        "thermodynamic_profile_source_type": "radiosonde",
        "thermodynamic_profile_station_id": "83779",
        "csv_file": "radiosonde_83779_20240101_12Z.csv",
    }

    result = thermodynamic_source_provenance(attrs, {})

    assert result["thermodynamic_profile_source_id"] == (
        "university_of_wyoming/upper_air_sounding/service_unversioned"
    )
    assert "83779" not in result["thermodynamic_profile_source_id"]


def test_ussa76_identity_records_standard_edition() -> None:
    attrs = {"thermodynamic_profile_source_type": "ussa76"}

    result = thermodynamic_source_provenance(attrs, {})

    assert result == {
        "thermodynamic_profile_provider": "us_standard_atmosphere",
        "thermodynamic_profile_product": "standard_atmosphere",
        "thermodynamic_profile_version_or_release": "1976",
        "thermodynamic_profile_source_id": (
            "us_standard_atmosphere/standard_atmosphere/1976"
        ),
    }
