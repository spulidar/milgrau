"""R&D tests for the method-v5 strict progressive vertical grid."""

from __future__ import annotations

import numpy as np

from milgrau.level2.adaptive_grid import (
    aggregate_to_progressive_grid,
    build_progressive_grid,
)


V5_GRID_SCHEDULE = (
    (0.0, 7.5),
    (6_000.0, 15.0),
    (10_000.0, 30.0),
    (15_000.0, 60.0),
    (20_000.0, 60.0),
    (25_000.0, 100.0),
)


def _cell_near(grid, altitude_m: float) -> int:
    return int(np.argmin(np.abs(grid.altitude_m - float(altitude_m))))


def test_progressive_grid_preserves_validated_low_column_and_caps_high_resolution() -> None:
    altitude = np.arange(0.0, 30_000.0 + 7.5, 7.5, dtype=np.float64)
    grid = build_progressive_grid(altitude, V5_GRID_SCHEDULE)

    assert grid.native_spacing_m == 7.5
    assert grid.source_start_index[0] == 0
    assert grid.source_stop_index[-1] == altitude.size
    assert np.all(grid.source_start_index[1:] == grid.source_stop_index[:-1])
    assert np.all(np.diff(grid.altitude_m) > 0.0)

    assert grid.source_count[_cell_near(grid, 5_000.0)] == 1
    assert grid.source_count[_cell_near(grid, 8_000.0)] == 2
    assert grid.source_count[_cell_near(grid, 12_000.0)] == 4
    assert grid.source_count[_cell_near(grid, 17_000.0)] == 8
    assert grid.source_count[_cell_near(grid, 22_000.0)] == 8
    assert grid.source_count[_cell_near(grid, 27_000.0)] == 13

    # Requested 100 m is represented by 13 native 7.5 m bins = 97.5 m,
    # never by interpolation to a fictitious exact 100 m cell. A final short
    # edge cell is allowed when the source profile ends mid-group.
    high = grid.altitude_m >= 25_100.0
    full_high_cells = high & (grid.source_count == 13)
    assert np.any(full_high_cells)
    assert np.all(grid.effective_resolution_m[high] <= 100.0)
    assert np.all(grid.source_count[high] <= 13)
    assert np.allclose(grid.effective_resolution_m[full_high_cells], 97.5)


def test_progressive_grid_is_exact_identity_through_established_lower_column() -> None:
    altitude = np.arange(0.0, 12_000.0 + 7.5, 7.5, dtype=np.float64)
    values = 2.0 + 0.001 * altitude
    grid = build_progressive_grid(altitude, V5_GRID_SCHEDULE)
    aggregated = aggregate_to_progressive_grid(values, grid)

    low_cells = grid.altitude_m < 6_000.0
    source_indices = grid.source_start_index[low_cells]
    assert np.all(grid.source_count[low_cells] == 1)
    assert np.array_equal(aggregated.values[low_cells], values[source_indices])
    assert np.all(aggregated.valid[low_cells])


def test_missing_native_sample_invalidates_only_its_own_output_cell() -> None:
    altitude = np.arange(0.0, 30_000.0 + 7.5, 7.5, dtype=np.float64)
    values = np.ones_like(altitude)
    grid = build_progressive_grid(altitude, V5_GRID_SCHEDULE)

    target_cell = _cell_near(grid, 22_000.0)
    start = int(grid.source_start_index[target_cell])
    values[start + 2] = np.nan

    aggregated = aggregate_to_progressive_grid(values, grid)
    assert not aggregated.valid[target_cell]
    assert np.isnan(aggregated.values[target_cell])
    assert aggregated.valid[target_cell - 1]
    assert aggregated.valid[target_cell + 1]


def test_finite_signed_rcs_can_be_averaged_without_being_called_a_gap() -> None:
    altitude = np.arange(0.0, 30_000.0 + 7.5, 7.5, dtype=np.float64)
    values = np.ones_like(altitude)
    grid = build_progressive_grid(altitude, V5_GRID_SCHEDULE)

    target_cell = _cell_near(grid, 12_000.0)
    start = int(grid.source_start_index[target_cell])
    assert grid.source_count[target_cell] == 4
    values[start] = -1.0

    signed_rcs = aggregate_to_progressive_grid(values, grid, require_positive=False)
    positive_state = aggregate_to_progressive_grid(values, grid, require_positive=True)

    assert signed_rcs.valid[target_cell]
    assert signed_rcs.values[target_cell] > 0.0
    assert not positive_state.valid[target_cell]


def test_uncertainty_dependence_limits_are_explicit() -> None:
    altitude = np.arange(0.0, 30_000.0 + 7.5, 7.5, dtype=np.float64)
    values = np.full_like(altitude, 10.0)
    sigma = np.full_like(altitude, 2.0)
    grid = build_progressive_grid(altitude, V5_GRID_SCHEDULE)

    cell = _cell_near(grid, 22_000.0)
    assert grid.source_count[cell] == 8

    independent = aggregate_to_progressive_grid(
        values,
        grid,
        uncertainty=sigma,
        uncertainty_mode="independent",
    )
    correlated = aggregate_to_progressive_grid(
        values,
        grid,
        uncertainty=sigma,
        uncertainty_mode="fully_correlated",
    )

    assert np.isclose(independent.uncertainty[cell], 2.0 / np.sqrt(8.0))
    assert np.isclose(correlated.uncertainty[cell], 2.0)
    assert independent.uncertainty[cell] < correlated.uncertainty[cell]
