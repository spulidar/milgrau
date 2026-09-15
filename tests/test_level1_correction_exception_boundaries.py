"""Regression tests for Level 1 diagnostic reduction exception boundaries."""

from __future__ import annotations

import pytest

from milgrau.level1.corrections import _safe_nanmax_xarray, _safe_nanmin_xarray


class _ReductionResult:
    def __init__(self, value: object) -> None:
        self.values = value


class _InvalidReduction:
    def max(self, *, skipna: bool):
        assert skipna is True
        return _ReductionResult("not-a-number")

    def min(self, *, skipna: bool):
        assert skipna is True
        return _ReductionResult("not-a-number")


class _BrokenReduction:
    def max(self, *, skipna: bool):
        assert skipna is True
        raise RuntimeError("unexpected implementation failure")

    def min(self, *, skipna: bool):
        assert skipna is True
        raise RuntimeError("unexpected implementation failure")


def test_level1_safe_reductions_fall_back_on_expected_conversion_failure() -> None:
    data = _InvalidReduction()
    assert _safe_nanmax_xarray(data, default=2.5) == 2.5
    assert _safe_nanmin_xarray(data, default=-1.5) == -1.5


def test_level1_safe_reductions_propagate_unexpected_runtime_failure() -> None:
    data = _BrokenReduction()
    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        _safe_nanmax_xarray(data)
    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        _safe_nanmin_xarray(data)
