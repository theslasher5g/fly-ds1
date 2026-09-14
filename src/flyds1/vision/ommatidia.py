"""Step 3a: map screen pixels onto ommatidia.

A frame is not an image to the fly: each ommatidium integrates light over a
Gaussian acceptance cone of ~5 deg half-width, and the cones tile a hexagonal
lattice at ~5 deg spacing.  So the mapping screen -> retina is a fixed linear
operator, and building it once as a sparse matrix makes the per-frame cost a
single sparse mat-vec.

Geometry: the monitor is a flat rectangle at a known distance, so a facet
looking along ``(azimuth, elevation)`` hits it at the gnomonic (pinhole)
projection

    px = cx + f * tan(azimuth),   py = cy - f * tan(elevation),
    f  = (width / 2) / tan(fov_h / 2)

Facets whose gaze misses the monitor get no weights at all -- the fly sees the
wall behind your desk there, and pretending otherwise would smear the screen
edge across the retina.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

#: Drosophila R1-6 acceptance angle (FWHM), degrees.  Gaussian sigma is
#: FWHM / 2.355.
ACCEPTANCE_FWHM_DEG = 5.7
FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


@dataclass(frozen=True)
class ScreenGeometry:
    """How much visual angle the monitor covers, and at what resolution."""

    width: int = 320
    height: int = 180
    fov_h_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("screen must be at least 2x2 pixels")
        if not 1.0 < self.fov_h_deg < 179.0:
            raise ValueError("fov_h_deg must be in (1, 179)")

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / np.tan(np.deg2rad(self.fov_h_deg) / 2.0)

    @property
    def fov_v_deg(self) -> float:
        return float(2.0 * np.rad2deg(np.arctan((self.height / 2.0) / self.focal_px)))

    @property
    def n_pixels(self) -> int:
        return self.width * self.height

    def project(self, coords_deg: np.ndarray) -> np.ndarray:
        """Gaze directions (degrees) -> pixel coordinates ``(n, 2)`` as (x, y)."""
        coords = np.asarray(coords_deg, dtype=float)
        az = np.deg2rad(coords[:, 0])
        el = np.deg2rad(coords[:, 1])
        f = self.focal_px
        x = self.width / 2.0 + f * np.tan(az)
        y = self.height / 2.0 - f * np.tan(el)
        behind = np.abs(az) >= np.pi / 2 - 1e-6
        x = np.where(behind, np.nan, x)
        y = np.where(behind, np.nan, y)
        return np.stack([x, y], axis=1)

    def degrees_per_pixel(self, coords_deg: np.ndarray) -> np.ndarray:
        """Local angular scale at each gaze direction, ``(n, 2)`` in deg/px.

        A flat screen is not angle-linear: the same pixel step spans less
        visual angle at the edges than in the centre (the ``cos^2`` factor).
        """
        coords = np.asarray(coords_deg, dtype=float)
        az = np.deg2rad(coords[:, 0])
        el = np.deg2rad(coords[:, 1])
        f = self.focal_px
        d_az = np.rad2deg(np.cos(az) ** 2) / f
        d_el = np.rad2deg(np.cos(el) ** 2) / f
        return np.stack([d_az, d_el], axis=1)


class OmmatidiaSampler:
    """Fixed linear map from a frame to per-facet light intensity.

    Parameters
    ----------
    eye_coords_deg:
        ``(n_facets, 2)`` gaze directions, degrees, relative to screen centre.
    screen:
        Resolution and field of view the frames are delivered in.  Frames are
        expected already resized to ``(screen.height, screen.width)``.
    acceptance_fwhm_deg:
        Angular width of one facet's acceptance function.
    truncate_sigma:
        Gaussian support radius; 3 sigma keeps >99% of the mass.
    """

    def __init__(
        self,
        eye_coords_deg: np.ndarray,
        screen: ScreenGeometry | None = None,
        *,
        acceptance_fwhm_deg: float = ACCEPTANCE_FWHM_DEG,
        truncate_sigma: float = 3.0,
    ) -> None:
        self.coords = np.asarray(eye_coords_deg, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(f"expected (n, 2) eye coordinates, got {self.coords.shape}")
        self.screen = screen or ScreenGeometry()
        self.acceptance_fwhm_deg = float(acceptance_fwhm_deg)
        self.truncate_sigma = float(truncate_sigma)
        self.matrix, self.in_screen = self._build_matrix()

    @property
    def n_facets(self) -> int:
        return len(self.coords)

    def _build_matrix(self) -> tuple[sp.csr_matrix, np.ndarray]:
        screen = self.screen
        centres = screen.project(self.coords)
        scale = screen.degrees_per_pixel(self.coords)  # deg per px, per axis
        sigma_deg = self.acceptance_fwhm_deg * FWHM_TO_SIGMA
        sigma_px = sigma_deg / np.maximum(scale, 1e-9)  # (n, 2) px per facet

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []
        in_screen = np.zeros(self.n_facets, dtype=bool)

        for k in range(self.n_facets):
            cx, cy = centres[k]
            if not np.isfinite([cx, cy]).all():
                continue
            sx, sy = sigma_px[k]
            rx = max(1, int(np.ceil(self.truncate_sigma * sx)))
            ry = max(1, int(np.ceil(self.truncate_sigma * sy)))
            x0, x1 = int(np.floor(cx - rx)), int(np.ceil(cx + rx))
            y0, y1 = int(np.floor(cy - ry)), int(np.ceil(cy + ry))
            x0c, x1c = max(0, x0), min(screen.width - 1, x1)
            y0c, y1c = max(0, y0), min(screen.height - 1, y1)
            if x1c < x0c or y1c < y0c:
                continue  # facet looks off-screen entirely
            xs = np.arange(x0c, x1c + 1)
            ys = np.arange(y0c, y1c + 1)
            gx = np.exp(-0.5 * ((xs + 0.5 - cx) / sx) ** 2)
            gy = np.exp(-0.5 * ((ys + 0.5 - cy) / sy) ** 2)
            weights = np.outer(gy, gx)
            total = weights.sum()
            if total <= 1e-12:
                continue
            weights /= total  # unit gain: a uniform grey frame gives that grey
            pix = (ys[:, None] * screen.width + xs[None, :]).ravel()
            rows.append(np.full(pix.size, k, dtype=np.int64))
            cols.append(pix)
            data.append(weights.ravel())
            in_screen[k] = True

        if not rows:
            return sp.csr_matrix((self.n_facets, screen.n_pixels)), in_screen
        matrix = sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_facets, screen.n_pixels),
        ).tocsr()
        return matrix, in_screen

    def sample(self, frame: np.ndarray) -> np.ndarray:
        """Sample one frame ``(H, W)`` or a batch ``(T, H, W)``.

        Returns ``(n_facets,)`` or ``(T, n_facets)`` intensities in the units of
        the frame (use :func:`luminance` first for RGB input).
        """
        arr = np.asarray(frame, dtype=float)
        expected = (self.screen.height, self.screen.width)
        if arr.shape[-2:] != expected:
            raise ValueError(f"frame has shape {arr.shape[-2:]}, expected {expected}")
        flat = arr.reshape(-1, self.screen.n_pixels)
        out = (self.matrix @ flat.T).T
        return out[0] if arr.ndim == 2 else out

    def coverage(self) -> float:
        """Fraction of facets that see any part of the screen."""
        return float(self.in_screen.mean())

    def facet_pixel_centres(self) -> np.ndarray:
        """Where each facet looks, in pixels -- handy for overlay plots."""
        return self.screen.project(self.coords)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luma from an ``(..., 3)`` uint8 or float frame, scaled to [0, 1].

    Flies are not trichromats like us; R1-6 are broadband.  A single luma
    channel is the honest summary of what they get from an sRGB screen.
    """
    arr = np.asarray(rgb, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) RGB, got {arr.shape}")
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Dependency-free nearest-neighbour resize for ``(H, W)`` / ``(H, W, C)``.

    Enough for "get the captured frame into screen resolution"; if you have
    OpenCV or Pillow installed, prefer an area-averaging resize.
    """
    arr = np.asarray(frame)
    h, w = arr.shape[:2]
    ys = np.clip((np.arange(height) + 0.5) * h / height, 0, h - 1).astype(int)
    xs = np.clip((np.arange(width) + 0.5) * w / width, 0, w - 1).astype(int)
    return arr[ys[:, None], xs[None, :]]


__all__ = [
    "ACCEPTANCE_FWHM_DEG",
    "OmmatidiaSampler",
    "ScreenGeometry",
    "luminance",
    "resize_nearest",
]
