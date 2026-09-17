"""Contract tests for traceable SPU MerionC Raman metadata."""

from __future__ import annotations

from pathlib import Path

import yaml


def _merionc_profile() -> dict:
    catalog = yaml.safe_load(Path("station.yaml").read_text(encoding="utf-8"))
    return next(
        profile
        for profile in catalog["profiles"]
        if profile["id"] == "spu-merionc-2024"
    )


def test_merionc_raman_identity_is_explicit_without_invented_passbands() -> None:
    profile = _merionc_profile()
    raman = profile["raman_detection"]

    assert raman["status"] == "partial_current_metadata"
    channels = raman["channels"]
    assert set(channels) == {"387", "408", "530"}

    assert channels["387"]["companion_elastic_nm"] == 355
    assert channels["387"]["species"] == "N2"
    assert channels["387"]["scattering_type"] == "vibrational_raman"

    assert channels["408"]["companion_elastic_nm"] == 355
    assert channels["408"]["species"] == "H2O"
    assert channels["408"]["scattering_type"] == "vibrational_raman"

    assert channels["530"]["companion_elastic_nm"] == 532
    assert channels["530"]["species"] == "N2"
    assert channels["530"]["scattering_type"] == "rotational_raman"

    for wavelength in ("387", "408", "530"):
        spectral = channels[wavelength]["current_spectral_response"]
        assert spectral == {"status": "current_instrument_evidence_required"}
        assert "passband_fwhm_nm" not in spectral
        assert "effective_detection_wavelength_nm" not in spectral


def test_merionc_raman_metadata_matches_current_scc_exposure() -> None:
    profile = _merionc_profile()
    channels = profile["raman_detection"]["channels"]
    night = profile["scc"]["night"]["channels"]
    day = profile["scc"]["day"]["channels"]

    assert channels["387"]["scc_night_channel_ids"] == {"AN": 4076, "PC": 4077}
    assert channels["530"]["scc_night_channel_ids"] == {"AN": 4075, "PC": 4074}
    assert night["387.AN"] == 4076
    assert night["387.PC"] == 4077
    assert night["530.AN"] == 4075
    assert night["530.PC"] == 4074

    assert channels["408"]["scc_mapping_status"] == (
        "published_current_identity_not_exposed_in_scc_1046"
    )
    assert "408.AN" not in night
    assert "408.PC" not in night
    assert not any(name.startswith(("387.", "408.", "530.")) for name in day)
