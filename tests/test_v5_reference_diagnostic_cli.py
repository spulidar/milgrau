"""Lightweight contracts for the real-Level-1 v5 reference diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

from milgrau.level2.v5_reference_diagnostic_cli import _domain_summary


def _cell(*, accepted: bool, admissible: bool, altitude_m: float, cost: float):
    candidate = SimpleNamespace(accepted=accepted, diagnostic_cost=cost)
    return SimpleNamespace(
        accepted=accepted,
        nominal_path_admissible=admissible,
        altitude_m=altitude_m,
        effective_resolution_m=30.0,
        native_rayleigh_candidate=candidate,
        cell_index=int(altitude_m),
    )


def test_domain_summary_separates_rayleigh_and_path_survival(monkeypatch) -> None:
    cells = (
        _cell(accepted=True, admissible=False, altitude_m=10500.0, cost=0.05),
        _cell(accepted=False, admissible=True, altitude_m=10800.0, cost=0.04),
        _cell(accepted=True, admissible=True, altitude_m=11100.0, cost=0.03),
    )
    catalogue = SimpleNamespace(
        cells=cells,
        search_min_altitude_m=10000.0,
        search_max_altitude_m=25000.0,
    )

    def _select(*args, **kwargs):
        del args, kwargs
        return cells[2]

    monkeypatch.setattr(
        "milgrau.level2.v5_reference_diagnostic_cli.select_minimum_cost_high_column_reference",
        _select,
    )
    summary = _domain_summary(catalogue)

    assert summary["catalogued_cells"] == 3
    assert summary["rayleigh_accepted_cells"] == 2
    assert summary["path_admissible_cells"] == 2
    assert summary["accepted_and_admissible_cells"] == 1
    assert summary["selected_reference_altitude_m"] == 11100.0


def test_domain_summary_handles_no_joint_candidate() -> None:
    catalogue = SimpleNamespace(
        cells=(
            _cell(accepted=True, admissible=False, altitude_m=10500.0, cost=0.05),
            _cell(accepted=False, admissible=True, altitude_m=10800.0, cost=0.04),
        ),
        search_min_altitude_m=10000.0,
        search_max_altitude_m=25000.0,
    )
    summary = _domain_summary(catalogue)

    assert summary["accepted_and_admissible_cells"] == 0
    assert summary["selected_reference_altitude_m"] is None
