"""Small local FastAPI application for the MILGRAU Web Explorer."""
from __future__ import annotations

from datetime import date
from mimetypes import guess_type
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from plotly.offline import get_plotlyjs

from milgrau.web.catalog import Catalog
from milgrau.web.rcs import options, rcs_map

STATIC = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def _local_plotly_js() -> str:
    """Ship the installed Plotly bundle locally, without a CDN or Node.js."""
    return get_plotlyjs()


def create_app(config_path: str = "config.yaml") -> FastAPI:
    catalog = Catalog(config_path)
    app = FastAPI(title="MILGRAU Web Explorer", docs_url="/api/docs", redoc_url=None)

    # Logos already declared in config.yaml for scientific figures. Serve only
    # this pre-resolved allowlist; the browser never supplies filesystem paths.
    logo_files: list[Path] = []
    for item in catalog.config.get("visualization", {}).get("logos", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        declared = Path(item["path"])
        resolved = (declared if declared.is_absolute() else
                    catalog.config_path.parent / (
                        Path("img") / declared if len(declared.parts) == 1 else declared
                    )).resolve()
        if resolved.is_file() and resolved.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            logo_files.append(resolved)

    @app.get("/")
    def homepage() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/static/plotly.min.js", include_in_schema=False)
    def plotly_bundle() -> Response:
        return Response(
            _local_plotly_js(), media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/site")
    def site() -> dict:
        return {
            **catalog.site,
            "logos": [
                {"url": f"/api/logo/{index}", "alt": logo.stem.replace("_", " ")}
                for index, logo in enumerate(logo_files)
            ],
        }

    @app.get("/api/logo/{index}", include_in_schema=False)
    def logo(index: int) -> FileResponse:
        if index < 0 or index >= len(logo_files):
            raise HTTPException(status_code=404, detail="Unknown configured logo")
        resolved = logo_files[index]
        if not resolved.is_file() or resolved.is_symlink():
            raise HTTPException(status_code=404, detail="Configured logo unavailable")
        media_type = guess_type(resolved.name)[0] or "image/png"
        return FileResponse(resolved, media_type=media_type)

    @app.get("/api/dates")
    def dates() -> dict:
        return {"dates": catalog.dates()}

    @app.get("/api/sessions")
    def sessions(utc_date: date | None = Query(default=None, alias="date")) -> dict:
        return {"sessions": (
            catalog.sessions(utc_date) if utc_date is not None else catalog.all_sessions()
        )}

    @app.get("/api/sessions/{session_id}/products")
    def products(session_id: str) -> dict:
        try:
            return {"products": catalog.products(session_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/options")
    def product_options(session_id: str, product: str) -> dict:
        try:
            return options(catalog.resolve(session_id, product))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/rcs")
    def rcs(
        session_id: str, product: str, channel: str,
        variable: str = "range_corrected_signal",
        statistic: str = "mean",
    ) -> dict:
        try:
            return rcs_map(
                catalog.resolve(session_id, product), channel,
                variable=variable, statistic=statistic,
                gap_minutes=float(catalog.config.get("visualization", {}).get(
                    "quicklook", {}).get("max_time_gap_minutes", 10.0)),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
