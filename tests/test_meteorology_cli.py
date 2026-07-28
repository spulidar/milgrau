"""SCI-004B prefetch CLI planning, dry-run and exit behavior."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import pytest

from milgrau.cli import meteorology as cli
from milgrau.config.loader import load_config
from milgrau.meteorology.request import AcquisitionMode
from milgrau.operations import ExecutionStatus


class ListLogger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))


def _args(**changes) -> argparse.Namespace:
    values = {
        "command": "prefetch",
        "start": datetime(2026, 7, 5, 12, 30, tzinfo=UTC),
        "end": datetime(2026, 7, 5, 13, 15, tzinfo=UTC),
        "site": "spu",
        "provider": "both",
        "radiosonde_time": [],
        "cache_only": False,
        "dry_run": True,
        "refresh_provisional": False,
        "config": "config.yaml",
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_dry_run_plans_without_acquisition(monkeypatch) -> None:
    logger = ListLogger()
    monkeypatch.setattr(
        cli,
        "get_or_acquire_meteorology",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("acquisition ran")),
    )
    summary = cli._prefetch(_args(), load_config(), logger)
    assert summary.results[0].status is ExecutionStatus.SUCCESS
    assert summary.results[0].metadata["era5_hours"] == 3
    assert any("ERA5 exact subrequests" in message for _, message in logger.messages)


def test_cache_only_flag_enters_cache_only_request_mode() -> None:
    request = cli._request_from_args(_args(cache_only=True), load_config())
    assert request.mode is AcquisitionMode.CACHE_ONLY


def test_cli_rejects_naive_timestamp_and_reversed_interval() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="offset"):
        cli._utc_datetime("2026-07-05T12:00:00")
    with pytest.raises(ValueError, match="precede"):
        cli._request_from_args(
            _args(
                start=datetime(2026, 7, 6, tzinfo=UTC),
                end=datetime(2026, 7, 5, tzinfo=UTC),
            ),
            load_config(),
        )


def test_prefetch_plan_never_contains_credentials(monkeypatch) -> None:
    monkeypatch.setenv("CDSAPI_KEY", "do-not-log-this")
    logger = ListLogger()
    cli._prefetch(_args(), load_config(), logger)
    assert "do-not-log-this" not in "\n".join(message for _, message in logger.messages)
