"""Queue observation and operational atmosphere selection: offline tests."""

from __future__ import annotations

import argparse
import logging

import pytest

from milgrau.cli import lipancora
from milgrau.io import cds_queue


def _args(**kwargs):
    defaults = dict(
        figures_only=False,
        radiosonde_only=False,
        no_cds_queue_check=False,
        cds_queue_threshold=8000,
    )
    return argparse.Namespace(**(defaults | kwargs))


def _config():
    return {
        "level1": {
            "atmosphere": {
                "source_priority": ["era5", "radiosonde", "ussa76"],
                "radiosonde": {"max_time_delta_hours": 6.0},
            }
        }
    }


def test_dashboard_parses_visible_queue_metric():
    snapshot = cds_queue.parse_cds_queue(
        "<p>Running requests 594</p><p>Queued requests</p><p>10 512</p>"
    )
    assert snapshot.queued_requests == 10512


def test_dashboard_missing_metric_does_not_claim_low_queue():
    with pytest.raises(ValueError, match="Queued requests"):
        cds_queue.parse_cds_queue("<html><p>System OK</p></html>")


def test_dashboard_ignores_untrusted_script_text():
    with pytest.raises(ValueError, match="Queued requests"):
        cds_queue.parse_cds_queue("<script>Queued requests 0</script><p>No metrics</p>")


def test_high_queue_disables_only_new_cds_network(monkeypatch):
    monkeypatch.setattr(
        lipancora, "read_cds_queue",
        lambda: cds_queue.CDSQueueSnapshot(queued_requests=10512),
    )
    original = _config()
    selected = lipancora._apply_cds_queue_policy(
        original, _args(), logging.getLogger("test.cds.queue")
    )
    assert selected["_era5_network_allowed"] is False
    assert selected["_cds_queue_state"] == "high"
    assert selected["level1"]["atmosphere"]["source_priority"][0] == "era5"
    assert selected["level1"]["atmosphere"]["radiosonde"]["max_time_delta_hours"] == 12.0
    assert original["level1"]["atmosphere"]["radiosonde"]["max_time_delta_hours"] == 6.0


def test_low_queue_keeps_normal_era5(monkeypatch):
    monkeypatch.setattr(
        lipancora, "read_cds_queue",
        lambda: cds_queue.CDSQueueSnapshot(queued_requests=100),
    )
    selected = lipancora._apply_cds_queue_policy(
        _config(), _args(), logging.getLogger("test.cds.queue")
    )
    assert selected["_era5_network_allowed"] is True


def test_unknown_queue_is_not_interpreted_as_low(monkeypatch):
    def unavailable():
        raise OSError("unreachable")

    monkeypatch.setattr(lipancora, "read_cds_queue", unavailable)
    selected = lipancora._apply_cds_queue_policy(
        _config(), _args(), logging.getLogger("test.cds.queue")
    )
    assert selected["_cds_queue_state"] == "unknown"
    assert "_era5_network_allowed" not in selected


def test_radiosonde_flag_allows_us_standard_atmosphere():
    selected = lipancora._radiosonde_only_config(_config())
    assert selected["level1"]["atmosphere"]["source_priority"] == [
        "radiosonde", "ussa76"
    ]
    assert selected["_era5_network_allowed"] is False
