"""Diagnostic images, without a plotting dependency.

matplotlib is not in the core requirements and a headless container has no
display, so this module writes PNGs directly (stdlib ``zlib`` + ``struct``).
Enough to answer the questions that actually come up: does the retina see the
frame, do the motion detectors light up where they should, is the network alive.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def save_png(image: np.ndarray, path: str | Path) -> Path:
    """Write a greyscale ``(h, w)`` or RGB ``(h, w, 3)`` image as PNG."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = to_uint8(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected (h, w) or (h, w, 3), got {np.shape(image)}")
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def to_uint8(values: np.ndarray, *, symmetric: bool = False) -> np.ndarray:
    """Normalise to 0-255.  ``symmetric`` centres 0 at mid-grey (for signed data)."""
    arr = np.asarray(values, dtype=float)
    if symmetric:
        peak = float(np.max(np.abs(arr))) or 1.0
        scaled = (arr / peak + 1.0) / 2.0
    else:
        low, high = float(np.min(arr)), float(np.max(arr))
        scaled = (arr - low) / (high - low) if high > low else np.zeros_like(arr)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def signed_colormap(values: np.ndarray) -> np.ndarray:
    """Signed data -> blue/black/orange RGB, so direction is readable at a glance."""
    arr = np.asarray(values, dtype=float)
    peak = float(np.max(np.abs(arr))) or 1.0
    x = np.clip(arr / peak, -1, 1)
    pos, neg = np.clip(x, 0, 1), np.clip(-x, 0, 1)
    rgb = np.stack([pos, 0.55 * pos + 0.35 * neg, neg], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def facet_image(
    coords_deg: np.ndarray,
    values: np.ndarray,
    *,
    size: int = 240,
    radius_deg: float = 3.0,
    signed: bool = False,
) -> np.ndarray:
    """Render per-facet values as a retinal map (nearest facet per pixel)."""
    coords = np.asarray(coords_deg, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    if len(coords) != len(vals):
        raise ValueError(f"{len(coords)} facets but {len(vals)} values")
    if len(coords) == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    lo = coords.min(axis=0) - radius_deg
    hi = coords.max(axis=0) + radius_deg
    span = np.maximum(hi - lo, 1e-6)
    ys = np.linspace(hi[1], lo[1], size)
    xs = np.linspace(lo[0], hi[0], int(size * span[0] / span[1]) or size)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    d = np.linalg.norm(grid[:, None, :] - coords[None, :, :], axis=-1)
    nearest = np.argmin(d, axis=1)
    within = d[np.arange(len(grid)), nearest] <= radius_deg
    picture = np.where(within, vals[nearest], np.nan).reshape(gy.shape)
    filled = np.nan_to_num(picture, nan=0.0)
    rgb = signed_colormap(filled) if signed else np.stack([to_uint8(filled)] * 3, axis=-1)
    rgb[~np.isfinite(picture)] = 24  # background
    return rgb


def bar_chart(
    values: np.ndarray,
    *,
    width: int = 240,
    height: int = 120,
    signed: bool = True,
    background: int = 24,
) -> np.ndarray:
    """Render a row of values as bars -- e.g. descending-neuron activity."""
    vals = np.asarray(values, dtype=float).ravel()
    if len(vals) == 0:
        return np.full((height, width, 3), background, dtype=np.uint8)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    peak = float(np.max(np.abs(vals))) or 1.0
    bar_w = max(1, width // len(vals))
    mid = height // 2 if signed else height - 1
    for k, value in enumerate(vals):
        x0 = k * bar_w
        x1 = min(width, x0 + max(1, bar_w - 1))
        extent = int((value / peak) * (height / 2 - 2)) if signed else int(
            (value / peak) * (height - 2)
        )
        if signed:
            y0, y1 = (mid - extent, mid) if extent >= 0 else (mid, mid - extent)
            colour = (255, 170, 60) if extent >= 0 else (80, 140, 255)
        else:
            y0, y1 = mid - extent, mid
            colour = (255, 170, 60)
        y0, y1 = max(0, min(y0, height - 1)), max(1, min(y1, height))
        canvas[y0:y1, x0:x1] = colour
    canvas[mid : mid + 1, :] = 90
    return canvas


def draw_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    colour: tuple[int, int, int] = (255, 80, 80),
    thickness: int = 2,
) -> np.ndarray:
    """Outline a fractional ``(x0, y0, x1, y1)`` region on an RGB image."""
    out = np.array(image, dtype=np.uint8, copy=True)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    h, w = out.shape[:2]
    x0, y0, x1, y1 = box
    px0, px1 = int(x0 * w), int(np.ceil(x1 * w))
    py0, py1 = int(y0 * h), int(np.ceil(y1 * h))
    px0, px1 = max(0, px0), min(w, max(px0 + 1, px1))
    py0, py1 = max(0, py0), min(h, max(py0 + 1, py1))
    t = max(1, thickness)
    out[py0 : py0 + t, px0:px1] = colour
    out[max(py0, py1 - t) : py1, px0:px1] = colour
    out[py0:py1, px0 : px0 + t] = colour
    out[py0:py1, max(px0, px1 - t) : px1] = colour
    return out


def mark_points(
    image: np.ndarray,
    points: np.ndarray,
    *,
    colour: tuple[int, int, int] = (90, 220, 120),
    radius: int = 1,
) -> np.ndarray:
    """Dot every point (pixel coordinates) on an RGB image."""
    out = np.array(image, dtype=np.uint8, copy=True)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    h, w = out.shape[:2]
    for x, y in np.asarray(points, dtype=float):
        if not np.isfinite([x, y]).all():
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            out[max(0, yi - radius) : yi + radius + 1, max(0, xi - radius) : xi + radius + 1] = colour
    return out


def filmstrip(images: list[np.ndarray], *, pad: int = 4, background: int = 40) -> np.ndarray:
    """Concatenate images horizontally, padding to the tallest."""
    prepared = []
    for img in images:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = to_uint8(arr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        prepared.append(arr)
    height = max(a.shape[0] for a in prepared)
    total = sum(a.shape[1] for a in prepared) + pad * (len(prepared) + 1)
    canvas = np.full((height + 2 * pad, total, 3), background, dtype=np.uint8)
    x = pad
    for arr in prepared:
        canvas[pad : pad + arr.shape[0], x : x + arr.shape[1]] = arr
        x += arr.shape[1] + pad
    return canvas


__all__ = [
    "bar_chart",
    "draw_box",
    "facet_image",
    "filmstrip",
    "mark_points",
    "save_png",
    "signed_colormap",
    "to_uint8",
]
