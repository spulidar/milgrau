"""Station-owned overlap configuration validation and resolution."""

from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Any, Mapping

import numpy as np

from milgrau.physics.overlap import (
    OVERLAP_MODEL_ID,
    coaxial_full_overlap_range_m,
    coaxial_geometric_overlap,
    receiver_field_stop_diameter_m,
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _positive(value: Any, label: str) -> float:
    result = _number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return result


def _non_negative(value: Any, label: str) -> float:
    result = _number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return result


def _validate_receiver(receiver: Mapping[str, Any], label: str) -> None:
    required = {
        "telescope_diameter_m",
        "focal_length_m",
        "field_of_view_full_angle_mrad",
        "field_of_view_status",
    }
    optional = {"field_stop_diameter_m", "field_stop_status"}
    missing = sorted(required - set(receiver))
    unknown = sorted(set(receiver) - required - optional)
    if missing or unknown:
        raise ValueError(f"{label} keys invalid; missing={missing}, unknown={unknown}.")
    _positive(receiver["telescope_diameter_m"], f"{label}.telescope_diameter_m")
    _positive(receiver["focal_length_m"], f"{label}.focal_length_m")
    _positive(receiver["field_of_view_full_angle_mrad"], f"{label}.field_of_view_full_angle_mrad")
    _text(receiver["field_of_view_status"], f"{label}.field_of_view_status")
    has_stop = "field_stop_diameter_m" in receiver
    has_stop_status = "field_stop_status" in receiver
    if has_stop != has_stop_status:
        raise ValueError(
            f"{label}.field_stop_diameter_m and {label}.field_stop_status must be provided together."
        )
    if has_stop:
        _positive(receiver["field_stop_diameter_m"], f"{label}.field_stop_diameter_m")
        _text(receiver["field_stop_status"], f"{label}.field_stop_status")


def _validate_transmitter(transmitter: Mapping[str, Any], label: str) -> None:
    required = {
        "beam_diameter_m",
        "beam_diameter_status",
        "divergence_full_angle_mrad",
        "divergence_status",
        "axis_separation_m",
        "axis_tilt_mrad",
        "alignment_status",
    }
    if set(transmitter) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}.")
    _positive(transmitter["beam_diameter_m"], f"{label}.beam_diameter_m")
    _text(transmitter["beam_diameter_status"], f"{label}.beam_diameter_status")
    _non_negative(transmitter["divergence_full_angle_mrad"], f"{label}.divergence_full_angle_mrad")
    _text(transmitter["divergence_status"], f"{label}.divergence_status")
    separation = _non_negative(transmitter["axis_separation_m"], f"{label}.axis_separation_m")
    tilt = _number(transmitter["axis_tilt_mrad"], f"{label}.axis_tilt_mrad")
    _text(transmitter["alignment_status"], f"{label}.alignment_status")
    if separation != 0.0 or tilt != 0.0:
        raise ValueError(
            f"{label} uses non-zero axis separation/tilt, but {OVERLAP_MODEL_ID} supports only coaxial parallel axes."
        )


def validate_overlap_catalog(catalog: Mapping[str, Any]) -> None:
    """Validate optional station overlap metadata and profile transmitter overrides."""
    station = _mapping(catalog.get("station"), "station")
    overlap = station.get("overlap")
    if overlap is None:
        return
    overlap = _mapping(overlap, "station.overlap")
    required = {
        "model",
        "status",
        "correction_policy",
        "receiver",
        "transmitter_fallback",
        "provenance",
    }
    optional = {"reported_full_overlap"}
    missing = sorted(required - set(overlap))
    unknown = sorted(set(overlap) - required - optional)
    if missing or unknown:
        raise ValueError(f"station.overlap keys invalid; missing={missing}, unknown={unknown}.")

    model = _text(overlap["model"], "station.overlap.model")
    if model != OVERLAP_MODEL_ID:
        raise ValueError(f"station.overlap.model must be {OVERLAP_MODEL_ID!r}.")
    _text(overlap["status"], "station.overlap.status")
    correction_policy = _text(overlap["correction_policy"], "station.overlap.correction_policy")
    if correction_policy != "diagnostic_only_no_correction":
        raise ValueError("station.overlap.correction_policy must be 'diagnostic_only_no_correction' for the unvalidated model.")

    _validate_receiver(_mapping(overlap["receiver"], "station.overlap.receiver"), "station.overlap.receiver")
    _validate_transmitter(
        _mapping(overlap["transmitter_fallback"], "station.overlap.transmitter_fallback"),
        "station.overlap.transmitter_fallback",
    )
    provenance = _mapping(overlap["provenance"], "station.overlap.provenance")
    if set(provenance) != {"source"}:
        raise ValueError("station.overlap.provenance must contain exactly ['source'].")
    _text(provenance["source"], "station.overlap.provenance.source")

    if "reported_full_overlap" in overlap:
        reported = _mapping(overlap["reported_full_overlap"], "station.overlap.reported_full_overlap")
        required_reported = {"altitude_m_agl", "status", "source"}
        if set(reported) != required_reported:
            raise ValueError(
                f"station.overlap.reported_full_overlap must contain exactly {sorted(required_reported)}."
            )
        _positive(reported["altitude_m_agl"], "station.overlap.reported_full_overlap.altitude_m_agl")
        _text(reported["status"], "station.overlap.reported_full_overlap.status")
        _text(reported["source"], "station.overlap.reported_full_overlap.source")

    for index, raw_profile in enumerate(catalog.get("profiles", [])):
        if not isinstance(raw_profile, Mapping):
            continue
        laser = raw_profile.get("laser")
        if not isinstance(laser, Mapping) or "overlap_geometry" not in laser:
            continue
        profile_id = str(raw_profile.get("id", index))
        _validate_transmitter(
            _mapping(laser["overlap_geometry"], f"profiles.{profile_id}.laser.overlap_geometry"),
            f"profiles.{profile_id}.laser.overlap_geometry",
        )


def resolve_overlap_context(catalog: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one profile's diagnostic overlap geometry with station fallback.

    Station-specific values remain in ``station.yaml``.  The resolver only
    combines the shared receiver geometry with either a profile transmitter
    override or the explicitly declared station fallback, then evaluates the
    generic model diagnostics.  No signal correction is applied here.
    """
    station = catalog.get("station")
    if not isinstance(station, Mapping) or not isinstance(station.get("overlap"), Mapping):
        return {}
    overlap = station["overlap"]
    receiver = deepcopy(dict(overlap["receiver"]))
    fallback = deepcopy(dict(overlap["transmitter_fallback"]))
    laser = profile.get("laser")
    override = laser.get("overlap_geometry") if isinstance(laser, Mapping) else None
    if isinstance(override, Mapping):
        transmitter = deepcopy(dict(override))
        parameter_source = "profile"
    else:
        transmitter = fallback
        parameter_source = "station_fallback"

    full_overlap = coaxial_full_overlap_range_m(
        telescope_diameter_m=receiver["telescope_diameter_m"],
        laser_beam_diameter_m=transmitter["beam_diameter_m"],
        receiver_fov_full_angle_mrad=receiver["field_of_view_full_angle_mrad"],
        laser_divergence_full_angle_mrad=transmitter["divergence_full_angle_mrad"],
    )
    field_stop = receiver_field_stop_diameter_m(
        focal_length_m=receiver["focal_length_m"],
        fov_full_angle_mrad=receiver["field_of_view_full_angle_mrad"],
    )

    reported = overlap.get("reported_full_overlap")
    reported_copy = deepcopy(dict(reported)) if isinstance(reported, Mapping) else None
    fraction_at_reported: float | None = None
    relation = "not_assessed"
    if reported_copy is not None:
        reported_altitude = float(reported_copy["altitude_m_agl"])
        fraction_at_reported = float(
            coaxial_geometric_overlap(
                reported_altitude,
                telescope_diameter_m=receiver["telescope_diameter_m"],
                laser_beam_diameter_m=transmitter["beam_diameter_m"],
                receiver_fov_full_angle_mrad=receiver["field_of_view_full_angle_mrad"],
                laser_divergence_full_angle_mrad=transmitter["divergence_full_angle_mrad"],
            )
        )
        if full_overlap is None:
            relation = "model_has_no_finite_full_overlap"
        elif reported_altitude < full_overlap:
            relation = "reported_range_below_model_full_overlap"
        else:
            relation = "reported_range_at_or_above_model_full_overlap"

    return {
        "model": str(overlap["model"]),
        "status": str(overlap["status"]),
        "correction_policy": str(overlap["correction_policy"]),
        "correction_applied": False,
        "receiver": receiver,
        "transmitter": transmitter,
        "parameter_source": parameter_source,
        "provenance": deepcopy(dict(overlap["provenance"])),
        "reported_full_overlap": reported_copy,
        "modeled_field_stop_diameter_m": field_stop,
        "model_full_overlap_altitude_m": full_overlap,
        "model_overlap_fraction_at_reported_altitude": fraction_at_reported,
        "reported_vs_model_relation": relation,
    }
