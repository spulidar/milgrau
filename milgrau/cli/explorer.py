"""Launch the new local FastAPI explorer (no Streamlit dependency)."""
from __future__ import annotations

import argparse

import uvicorn

from milgrau.explorer.app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="MILGRAU local scientific web explorer")
    parser.add_argument("--config", default="config.yaml", help="MILGRAU configuration")
    parser.add_argument("--host", default="127.0.0.1", help="Local bind address")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    args = parser.parse_args()
    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
