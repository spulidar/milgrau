"""Console entry point for the MILGRAU Streamlit NetCDF explorer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the Streamlit explorer app."""

    app_path = Path(__file__).resolve().parents[1] / "milgrau" / "explorer" / "app.py"
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]])
    )


if __name__ == "__main__":
    main()
