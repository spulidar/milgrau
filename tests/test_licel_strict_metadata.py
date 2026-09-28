"""Strict acquisition-metadata tests for Licel channel headers."""

from __future__ import annotations

import pytest

from milgrau.io.licel import _parse_channel_metadata


def _header(*, is_pc: int = 0, bin_width: str = "7.5", adc_bits: str = "12", discriminator: str = "0.5") -> str:
    return " ".join(
        [
            "1", str(is_pc), "1", "1000", "1", "800", bin_width, "532.0",
            "0", "0", "0", "0", adc_bits, "3000", discriminator, "BT00",
        ]
    )


def test_active_channel_requires_positive_licel_bin_width() -> None:
    with pytest.raises(ValueError, match="BinW"):
        _parse_channel_metadata(_header(bin_width="0"), "invalid-bin-width.licel")


def test_analog_channel_requires_positive_adc_bit_depth() -> None:
    with pytest.raises(ValueError, match="ADC bit depth"):
        _parse_channel_metadata(_header(adc_bits="0"), "invalid-adc.licel")


def test_analog_channel_requires_explicit_positive_discriminator_range() -> None:
    with pytest.raises(ValueError, match="Discriminator/DAQ range"):
        _parse_channel_metadata(_header(discriminator="0"), "invalid-range.licel")


def test_analog_channel_does_not_reconstruct_missing_adc_or_range_fields() -> None:
    truncated = "1 0 1 1000 1 800 7.5 532.0 0 0 0 0"
    with pytest.raises(ValueError, match="lacks required ADCbits"):
        _parse_channel_metadata(truncated, "truncated.licel")


def test_photon_counting_channel_does_not_need_analog_daq_metadata() -> None:
    metadata = _parse_channel_metadata(_header(is_pc=1, adc_bits="0", discriminator="0"), "pc.licel")
    assert metadata["is_pc"] is True
    assert metadata["adc_bits"] == 0
    assert metadata["bin_width_m"] == 7.5
