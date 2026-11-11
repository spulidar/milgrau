#!/usr/bin/env python3
"""Fail the documentation build on missing homepage content or unrenderable math.

This validates generated HTML and KaTeX initialization, not a scientific
verification of the underlying equations or a browser screenshot.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"


def check() -> None:
    home = (PUBLIC / "index.html").read_text(encoding="utf-8")
    l1 = (PUBLIC / "docs/processing/level-1/lipancora/index.html").read_text(
        encoding="utf-8"
    )
    for text in (
        "MILGRAU",
        "One observation, three scientific levels",
        "Built for scientific scrutiny",
        "LIBIDS",
        "LIPANCORA",
        "LEBEAR",
        "SCIENTIFIC PROCESSING FLOW",
    ):
        assert text in home, f"Homepage missing: {text}"
    assert "docs/processing/level-0/libids/" in home
    assert "docs/processing/level-1/lipancora/" in home
    assert "docs/processing/level-2/lebear/" in home

    assert "katex" in l1.lower(), "No KaTeX assets in the Level 1 page"
    assert re.search(r'class=["\']?math-display', l1), (
        "No display math passthrough container"
    )
    assert re.search(r'class=["\']?math-inline', l1), (
        "No inline math passthrough container"
    )
    assert re.search(r'\$\$\s*[^$]{0,180}\\(?:frac|sigma|Delta|tfrac)', l1), (
        "Display math lost its KaTeX delimiters"
    )
    # A double escaped LaTeX command would not be interpreted as the intended
    # mathematical command. This catches a regression of the original issue.
    assert not re.search(r'\\\\(?:frac|sigma|Delta|tfrac|mathrm|tau)', l1), (
        "Doubly escaped math command still present"
    )
    print("Homepage, Level 1 math delimiters and KaTeX assets: OK")


if __name__ == "__main__":
    check()
