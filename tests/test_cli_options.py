"""CLI surface and release-version consistency tests."""

from __future__ import annotations

from pathlib import Path
import tomllib

import yaml

from milgrau.cli import lebear, libids, lipancora, liracos
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
