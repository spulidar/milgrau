"""Tests for surface-weather cache/retry exception boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

import milgrau.io.weather as weather_module


def test_surface_weather_propagates_unexpected_runtime_failure(tmp_path, monkeypatch) -> None:
    """Programming defects must not be retried and converted into ordinary API failure."""
    when = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    cache_file = weather_module._weather_cache_file(when, -23.5, -46.6, tmp_path)
    cache_file.write_text(json.dumps({"hourly": {"time": ["2024-01-01T12:00"]}}), encoding="utf-8")

    def fail_extraction(*_args, **_kwargs):
        raise RuntimeError("synthetic weather implementation defect")

    monkeypatch.setattr(weather_module, "_extract_surface_weather_from_payload", fail_extraction)

    with pytest.raises(RuntimeError, match="synthetic weather implementation defect"):
        weather_module.fetch_surface_weather(
            when,
            -23.5,
            -46.6,
            cache_dir=tmp_path,
        )
