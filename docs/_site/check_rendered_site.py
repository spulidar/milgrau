#!/usr/bin/env python3
"""Fail the documentation build on missing homepage content or unrenderable math.

This validates generated HTML and KaTeX initialization, not a scientific
verification of the underlying equations or a browser screenshot.
"""
from __future__ import annotations

import re
import html
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
    # Only check actual math containers. The HTML can legitimately contain
    # doubled backslashes in JavaScript, JSON or highlighted source code.
    # Hugo --minify may remove the quotes around class attributes.
    math_fragments = re.findall(
        r'<(?:div|span)\b[^>]*\bclass=["\x27]?math-(?:display|inline)["\x27]?[^>]*>(.*?)</(?:div|span)>',
        l1,
        flags=re.DOTALL,
    )
    assert math_fragments, "No rendered math containers found"
    math_only = "\n".join(html.unescape(fragment) for fragment in math_fragments)
    assert re.search(r'\\(?:frac|sigma|Delta|tfrac)', math_only), (
        "Expected LaTeX commands are missing from math containers"
    )
    assert not re.search(r'\\\\(?:frac|sigma|Delta|tfrac|mathrm|tau)', math_only), (
        "Doubly escaped LaTeX command found inside a math container"
    )
    # Verify the doc footer follows the editorial sequence rather than the
    # default Hugo title ordering (which used to jump to LEBEAR).
    for current, previous, following in (
        ("getting-started/getting_started", None, "getting-started/operating_milgrau"),
        ("getting-started/operating_milgrau", "getting-started/getting_started", "getting-started/configuration_reference"),
        ("processing/level-0/libids", "getting-started/processing_levels", "processing/level-0/level0_schema"),
        ("processing/level-1/lipancora", "processing/level-0/level0_validation_workbook", "processing/level-1/level1_schema"),
    ):
        page = (PUBLIC / "docs" / current / "index.html").read_text(encoding="utf-8")
        match = re.search(r'<nav\\b[^>]*\\bid=["\\x27]?doc-nav\\b', page)
        assert match, f"No documentation navigation on {current}"
        nav = page[match.start():]
        nav = nav.split("</nav>", 1)[0]
        assert following and following in nav, f"Wrong next page for {current}: {following}"
        if previous:
            assert previous in nav, f"Wrong previous page for {current}: {previous}"

    print("Homepage, Level 1 math delimiters and KaTeX assets: OK")


if __name__ == "__main__":
    check()
