"""Give photoreceptors an eye coordinate.

Step 3 needs to know *where each photoreceptor looks*.  Depending on the
dataset that information arrives in one of three forms, in decreasing order of
preference:

1. Explicit hex column annotations (``column_p``/``column_q``, ``hex1``/``hex2``
   in the optic-lobe annotation tables) -- used directly.
2. Only soma positions.  Photoreceptor somata of one eye lie on a curved sheet;
   projecting them onto their best-fit plane and rescaling to the
   interommatidial angle recovers a usable retinotopic map.  That is what
   :func:`eye_coords_from_soma` does.
3. Nothing.  Then there is no retinotopy to be had and the caller must fall
   back to the synthetic lattice -- we raise rather than invent coordinates.
"""

from __future__ import annotations

import numpy as np

from flyds1.hexlattice import axial_to_cartesian


def eye_coords_from_hex_columns(
    hex1: np.ndarray, hex2: np.ndarray, spacing_deg: float = 5.0
) -> np.ndarray:
    """Convert integer hex column annotations to degrees of visual angle."""
    axial = np.stack([np.asarray(hex1, dtype=float), np.asarray(hex2, dtype=float)], axis=1)
    return axial_to_cartesian(axial, spacing_deg)


def eye_coords_from_soma(
    soma_xyz: np.ndarray, spacing_deg: float = 5.0, *, flip_x: bool = False
) -> np.ndarray:
    """Project soma positions onto their best-fit plane and scale to degrees.

    The scale is set so the median nearest-neighbour distance equals
    ``spacing_deg``, i.e. neighbouring ommatidia end up one interommatidial
    angle apart.  This is a retinotopic *approximation*: it preserves
    neighbourhood relations, which is what the sampling and the motion
    detectors need, but it is not a calibrated optical axis map.
    """
    xyz = np.asarray(soma_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected (n, 3) soma positions, got {xyz.shape}")
    if len(xyz) < 3:
        raise ValueError("need at least 3 photoreceptors to fit an eye plane")

    centred = xyz - xyz.mean(axis=0)
    # principal plane: the two directions of largest spread
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    plane = centred @ vt[:2].T  # (n, 2)

    step = _median_nearest_neighbour(plane)
    if step <= 0:  # pragma: no cover - degenerate input
        raise ValueError("photoreceptor positions are degenerate")
    coords = plane * (spacing_deg / step)
    if flip_x:
        coords[:, 0] *= -1.0
    return coords


def _median_nearest_neighbour(points: np.ndarray, sample: int = 2000) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) > sample:
        idx = np.random.default_rng(0).choice(len(pts), size=sample, replace=False)
        pts_s = pts[idx]
    else:
        pts_s = pts
    d = np.linalg.norm(pts_s[:, None, :] - pts_s[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def snap_to_hex_lattice(coords: np.ndarray, spacing_deg: float = 5.0) -> np.ndarray:
    """Round continuous eye coordinates onto the nearest hex lattice site.

    Useful when wiring Reichardt detectors between "neighbouring" facets of a
    measured, slightly irregular eye: the detectors need a discrete lattice.
    """
    coords = np.asarray(coords, dtype=float)
    # invert axial_to_cartesian: x = s(q + r/2), y = s*sqrt(3)/2*r
    r = coords[:, 1] / (spacing_deg * np.sqrt(3.0) / 2.0)
    q = coords[:, 0] / spacing_deg - 0.5 * r
    axial = np.stack(_round_axial(q, r), axis=1)
    return axial


def _round_axial(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cube-coordinate rounding: the correct way to round hex coordinates."""
    x, z = np.asarray(q, float), np.asarray(r, float)
    y = -x - z
    rx, ry, rz = np.round(x), np.round(y), np.round(z)
    dx, dy, dz = np.abs(rx - x), np.abs(ry - y), np.abs(rz - z)
    fix_x = (dx > dy) & (dx > dz)
    fix_z = (~fix_x) & (dz > dy)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(fix_z, -rx - ry, rz)
    return rx.astype(np.int64), rz.astype(np.int64)


__all__ = [
    "eye_coords_from_hex_columns",
    "eye_coords_from_soma",
    "snap_to_hex_lattice",
]
