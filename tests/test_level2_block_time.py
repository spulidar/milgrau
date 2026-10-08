"""Tests for clock-anchored Level 2 grouping and representative block time."""

from __future__ import annotations

import numpy as np

from milgrau.level2.block_average import block_groups


def test_block_groups_keep_clock_buckets_but_use_mean_profile_time() -> None:
    times = np.array(
        [
            "2024-01-01T00:01:00",
            "2024-01-01T00:09:00",
            "2024-01-01T00:19:00",
            "2024-01-01T00:21:00",
            "2024-01-01T00:39:00",
        ],
        dtype="datetime64[s]",
    )

    block_time, groups = block_groups(times, 20)

    assert [group.tolist() for group in groups] == [[0, 1, 2], [3, 4]]
    np.testing.assert_array_equal(
        block_time,
        np.array(
            ["2024-01-01T00:09:40", "2024-01-01T00:30:00"],
            dtype="datetime64[ns]",
        ),
    )



def test_block_groups_split_one_clock_bucket_at_solar_segment_transition() -> None:
    times = np.array(
        [
            "2024-01-01T00:01:00",
            "2024-01-01T00:08:00",
            "2024-01-01T00:11:00",
            "2024-01-01T00:18:00",
        ],
        dtype="datetime64[s]",
    )
    segment_ids = np.array(["seg00", "seg00", "seg01", "seg01"], dtype=object)

    block_time, groups = block_groups(times, 20, segment_ids=segment_ids)

    assert [group.tolist() for group in groups] == [[0, 1], [2, 3]]
    np.testing.assert_array_equal(
        block_time,
        np.array(
            ["2024-01-01T00:04:30", "2024-01-01T00:14:30"],
            dtype="datetime64[ns]",
        ),
    )
