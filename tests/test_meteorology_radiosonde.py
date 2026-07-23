"""SCI-004A offline Wyoming/Siphon payload normalization."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from milgrau.meteorology.contracts import HumidityFlag, ProfileQuality
from milgrau.meteorology.radiosonde import normalize_wyoming_radiosonde


def test_radiosonde_fixture_normalizes_units_duplicates_gaps_and_coverage(
    radiosonde_normalization,
) -> None:
    result = radiosonde_normalization
    profile = result.profile

    assert result.duplicate_levels_removed == 1
    assert result.incomplete_levels_removed == 1
    assert result.preserved_gap_count == 5
    assert result.maximum_gap_m == pytest.approx(3000.0)
    assert profile.geometric_altitude_m.tolist() == [760.0, 1000.0, 1600.0, 3500.0, 6500.0, 9000.0, 12000.0, 15000.0]
    assert profile.pressure_pa[0] == pytest.approx(93000.0)
    assert profile.temperature_k[0] == pytest.approx(291.65)
    assert profile.vertical_coverage_m == (760.0, 15000.0)
    assert profile.station_or_dataset_id == "83779"


def test_radiosonde_derives_q_from_dewpoint_and_marks_missing_humidity(
    radiosonde_normalization,
) -> None:
    profile = radiosonde_normalization.profile

    assert np.all(np.isfinite(profile.specific_humidity_kg_kg[:5]))
    assert np.all(
        profile.humidity_flag[:5] == int(HumidityFlag.DERIVED_FROM_DEWPOINT)
    )
    assert np.all(np.isnan(profile.specific_humidity_kg_kg[5:]))
    assert np.all(profile.humidity_flag[5:] == int(HumidityFlag.MISSING))
    assert profile.profile_quality is ProfileQuality.INCOMPLETE
    assert not profile.quantitative_retrieval_allowed


def test_radiosonde_dry_assumption_is_explicit_not_missing(radiosonde_fixture_payload) -> None:
    payload, raw = radiosonde_fixture_payload
    profile = normalize_wyoming_radiosonde(
        pd.DataFrame(payload["records"]),
        nominal_time=datetime.fromisoformat(payload["nominal_time_utc"]),
        observation_time=datetime.fromisoformat(payload["observation_time_utc"]),
        station_id=payload["station_id"],
        latitude_deg_north=payload["latitude_deg_north"],
        longitude_deg_east=payload["longitude_deg_east"],
        raw_snapshot=raw,
        assume_dry_when_humidity_missing=True,
    ).profile

    assumed = profile.humidity_flag == int(HumidityFlag.DRY_AIR_ASSUMED)
    assert assumed.sum() == 3
    assert np.all(profile.specific_humidity_kg_kg[assumed] == 0.0)
    assert profile.profile_quality is ProfileQuality.QUANTITATIVE


def test_hydrostatic_check_is_diagnostic_and_does_not_adjust_observations(
    radiosonde_normalization,
) -> None:
    result = radiosonde_normalization

    assert result.hydrostatic.compared_layer_count == 4
    assert result.hydrostatic.maximum_absolute_log_pressure_residual > 0.0
    assert result.profile.pressure_pa[:5].tolist() == pytest.approx(
        [93000.0, 90000.0, 84000.0, 65000.0, 47000.0]
    )


def test_radiosonde_snapshot_hash_is_deterministic(radiosonde_fixture_payload) -> None:
    payload, raw = radiosonde_fixture_payload
    kwargs = dict(
        nominal_time=datetime.fromisoformat(payload["nominal_time_utc"]),
        observation_time=datetime.fromisoformat(payload["observation_time_utc"]),
        station_id=payload["station_id"],
        latitude_deg_north=payload["latitude_deg_north"],
        longitude_deg_east=payload["longitude_deg_east"],
        raw_snapshot=raw,
    )
    first = normalize_wyoming_radiosonde(pd.DataFrame(payload["records"]), **kwargs)
    second = normalize_wyoming_radiosonde(pd.DataFrame(payload["records"]), **kwargs)

    assert first.profile.raw_snapshot_sha256 == second.profile.raw_snapshot_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("pressure", -1.0, "pressure"), ("temperature", -300.0, "absolute zero")],
)
def test_radiosonde_rejects_nonphysical_pressure_and_temperature(
    field: str, value: float, message: str
) -> None:
    payload = pd.DataFrame(
        {
            "height": [1000.0, 2000.0],
            "pressure": [900.0, 800.0],
            "temperature": [20.0, 10.0],
            "dewpoint": [15.0, 5.0],
        }
    )
    payload.loc[1, field] = value
    with pytest.raises(ValueError, match=message):
        normalize_wyoming_radiosonde(
            payload,
            nominal_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
            observation_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
            station_id="83779",
            latitude_deg_north=-23.5167,
            longitude_deg_east=-46.6333,
        )


def test_radiosonde_requires_contract_columns() -> None:
    with pytest.raises(KeyError, match="pressure"):
        normalize_wyoming_radiosonde(
            [{"height": 1000.0, "temperature": 20.0}],
            nominal_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
            observation_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
            station_id="83779",
            latitude_deg_north=-23.5167,
            longitude_deg_east=-46.6333,
        )
