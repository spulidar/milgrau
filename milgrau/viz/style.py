"""Shared visual style helpers for MILGRAU figures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
import math

import matplotlib.image as mpimg
import numpy as np

from milgrau.viz.config import resolve_visualization_config

WAVELENGTH_COLORS: dict[int, str] = {
    355: "rebeccapurple",
    387: "darkblue",
    408: "darkcyan",
    530: "orange",
    532: "forestgreen",
    1064: "crimson",
}

DEFAULT_LOGO_SPECS: tuple[tuple[str, float], ...] = (
    ("CC_BY-NC-ND.png", 0.040),
    ("lalinet_logo2.png", 0.070),
    ("LEAEL_LOGO.png", 0.065),
)



def configured_logos(config: Mapping[str, Any] | None = None) -> tuple[tuple[str, float], ...]:
    """Resolve ordered logo paths and relative axes heights from the figure recipe.

    An explicit empty list disables logos; older configs use the built-in logos.
    """
    if config is None:
        return DEFAULT_LOGO_SPECS
    visualization = config.get("visualization")
    if not isinstance(visualization, Mapping) or "logos" not in visualization:
        return DEFAULT_LOGO_SPECS
    raw = visualization["logos"]
    if not isinstance(raw, list):
        raise ValueError("visualization.logos must be a list (or [] to hide logos).")
    specs: list[tuple[str, float]] = []
    names: set[str] = set()
    for i, item in enumerate(raw):
        name = f"visualization.logos[{i}]"
        if not isinstance(item, Mapping) or set(item) != {"path", "height"}:
            raise ValueError(f"{name} must contain exactly path and height.")
        path = item["path"]
        height = item["height"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{name}.path must be a nonempty image path.")
        path = path.strip()
        if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"{name}.path must be a PNG, JPEG or WebP image.")
        if isinstance(height, bool) or not isinstance(height, (int, float)):
            raise ValueError(f"{name}.height must be a positive finite number.")
        height = float(height)
        if not math.isfinite(height) or height <= 0 or height > 0.30:
            raise ValueError(f"{name}.height must be between 0 and 0.30.")
        if path in names:
            raise ValueError(f"Duplicate visualization logo path: {path}")
        specs.append((path, height))
        names.add(path)
    return tuple(specs)


def resolve_logo_path(root_dir: str | Path, image_path: str) -> Path:
    """Resolve filenames under img/ and other relative paths under project root."""
    path = Path(image_path).expanduser()
    if path.is_absolute():
        return path
    root = Path(root_dir)
    return root / ("img" if len(path.parts) == 1 else "") / path


def logo_file_dependencies(
    config: Mapping[str, Any] | None, root_dir: str | Path
) -> list[Path]:
    """Track selected artwork so changing a logo invalidates figure caches."""
    return [
        path
        for name, _height in configured_logos(config)
        if (path := resolve_logo_path(root_dir, name)).is_file()
    ]


@lru_cache(maxsize=16)
def read_logo_image(path_str: str, _mtime_ns: int) -> np.ndarray:
    """Read and cache a logo image from disk."""
    return mpimg.imread(path_str)


def channel_color(channel_or_wavelength: str | int | float) -> str:
    """Return the MILGRAU display color associated with a wavelength."""
    try:
        if isinstance(channel_or_wavelength, str):
            wavelength = int(channel_or_wavelength.split(".")[0])
        else:
            wavelength = int(channel_or_wavelength)
    except Exception:
        return "black"
    return WAVELENGTH_COLORS.get(wavelength, "black")


def get_output_settings(config: dict[str, Any]) -> tuple[str, int]:
    """Return explicitly configured output format and DPI."""
    resolved = resolve_visualization_config(config)
    return resolved.output_format, resolved.dpi


def add_footer_and_logos(
    fig: Any,
    root_dir: str | Path,
    subtitle: str | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Add SPU-Lidar footer, optional measurement metadata, and institutional logos."""
    fig.text(
        0.08,
        0.04,
        "SPU Lidar Station - São Paulo",
        fontsize=13,
        fontweight="bold",
        color="#333333",
        va="center",
    )
    if subtitle:
        fig.text(
            0.08,
            0.018,
            subtitle,
            fontsize=10,
            color="#555555",
            va="center",
        )
    spacing = 0.010
    y_pos = 0.01
    x_right = 0.98
    for logo_name, height in configured_logos(config):
        logo_path = resolve_logo_path(root_dir, logo_name)
        if not logo_path.is_file():
            continue
        img = read_logo_image(str(logo_path), logo_path.stat().st_mtime_ns)
        img_h, img_w = img.shape[:2]
        width = height * (img_w / img_h)
        if x_right - width < 0.52:
            width = max(0.0, x_right - 0.52)
            if width <= 0.01:
                break
            height = width * img_h / img_w
        x_left = x_right - width
        ax_logo = fig.add_axes([x_left, y_pos, width, height], zorder=12)
        ax_logo.imshow(img)
        ax_logo.axis("off")
        x_right = x_left - spacing
