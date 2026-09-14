"""Regression tests for Level 2 QA statistics on sparse retrieval support."""

from __future__ import annotations

import warnings

import numpy as np

from milgrau.viz.level2_qa import _block_standard_error


def test_block_standard_error_keeps_unsupported_bins_nan_without_warning() -> None:
    blocks = np.array(
        [
            [1.0, np.nan, 4.0, np.nan],
            [3.0, np.nan, np.nan, np.nan],
            [5.0, np.nan, np.nan, 8.0],
        ],
        dtype=np.float64,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sem = _block_standard_error(blocks)

    assert caught == []
    expected_first = np.std(np.array([1.0, 3.0, 5.0]), ddof=0) / np.sqrt(3.0)
    assert np.isclose(sem[0], expected_first)
    assert np.isnan(sem[1])
    assert np.isnan(sem[2])
    assert np.isnan(sem[3])


def test_block_standard_error_respects_valid_block_mask() -> None:
    blocks = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [100.0, 200.0],
        ],
        dtype=np.float64,
    )
    valid_block = np.array([True, True, False])

    sem = _block_standard_error(blocks, valid_block)

    expected = np.std(np.array([1.0, 3.0]), ddof=0) / np.sqrt(2.0)
    assert np.isclose(sem[0], expected)
    assert np.isclose(sem[1], expected)
