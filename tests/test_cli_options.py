"""CLI surface and release-version consistency tests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import tomllib

import yaml

from milgrau.cli import lebear, libids, lipancora, liracos
from milgrau.io.selection import parse_input_selection
from milgrau.operations import ExecutionResult
from milgrau.version import __version__


def test_primary_clis_share_input_force_and_version_options() -> None:
    for module in (libids, lipancora, liracos, lebear):
        parser = module._build_parser()
        option_strings = {option for action in parser._actions for option in action.option_strings}
        assert "--input" in option_strings
        assert "--force" in option_strings
        assert "--version" in option_strings


def test_calendar_version_is_synchronized_with_packaging_and_citation() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__
    assert str(citation["version"]) == __version__
    year, month = __version__.split(".")
    assert len(year) == 4 and year.isdigit()
    assert month.isdigit() and 1 <= int(month) <= 12


def test_lipancora_explicit_file_does_not_require_milgrau_filename(tmp_path: Path, monkeypatch) -> None:
    """External SCC raw filenames must not be used as scientific identity."""
    input_path = tmp_path / "foreign_station_scc_raw.nc"
    input_path.write_bytes(b"placeholder")
    observed: list[Path] = []

    def fake_process_single_file(args):
        path, _config, _logger = args
        observed.append(Path(path))
        return ExecutionResult.success(
            "level1.complete",
            "test",
            input_path=path,
            output_path=tmp_path / "foreign_station_L1.nc",
            duration_seconds=0.0,
        )

    monkeypatch.setattr(lipancora, "process_single_file", fake_process_single_file)
    args = argparse.Namespace(inputs=[str(input_path)], force=True)

    summary = lipancora._process_selected(args, {}, logging.getLogger("test.lipancora.external"))

    assert observed == [input_path]
    assert len(summary.results) == 1


def _selector_config() -> dict:
    return {"_station_catalog": {"station": {"id": "spu"}}}


def test_shared_input_parser_accepts_measurement_date_and_period_forms() -> None:
    config = _selector_config()

    exact = parse_input_selection([["20251107_spu_06"]], config)
    assert exact.measurement_ids == frozenset({"20251107_spu_06"})

    whole_day = parse_input_selection([["20251107"]], config)
    assert whole_day.dates == frozenset({"20251107"})

    periods = parse_input_selection([["20251107", "06", "12"]], config)
    assert periods.measurement_ids == frozenset({"20251107_spu_06", "20251107_spu_12"})

    quoted_style = parse_input_selection(["20251107 06 12"], config)
    assert quoted_style.measurement_ids == periods.measurement_ids


def test_shared_input_parser_preserves_explicit_path_with_spaces(tmp_path: Path) -> None:
    input_path = tmp_path / "foreign level 1.nc"
    input_path.write_bytes(b"placeholder")

    selection = parse_input_selection([[str(input_path)]], {})
    assert selection.paths == (input_path.resolve(),)


def test_primary_cli_parsers_accept_date_with_multiple_periods() -> None:
    for module in (libids, lipancora, liracos, lebear):
        args = module._build_parser().parse_args(["-i", "20251107", "06", "12"])
        assert args.inputs == [["20251107", "06", "12"]]


def test_lebear_time_window_option_is_explicitly_utc() -> None:
    parser = lebear._build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--time-window-utc" in option_strings
    assert "--time-window" not in option_strings
