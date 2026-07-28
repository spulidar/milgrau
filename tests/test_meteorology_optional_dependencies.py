"""The SCI-004A kernel remains importable without ERA5 acquisition extras."""

from __future__ import annotations

import subprocess
import sys


def test_core_import_does_not_import_cdsapi_or_eccodes() -> None:
    code = (
        "import sys; import milgrau.meteorology; "
        "assert 'cdsapi' not in sys.modules; assert 'eccodes' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
