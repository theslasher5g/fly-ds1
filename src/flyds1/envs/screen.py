"""Grabbing frames from a monitor or a game window.

``mss`` is the fast path (it does a single blit per grab, no PIL round-trip).
``DummyCapture`` renders a synthetic frame instead, so the whole game env can be
exercised without a screen -- which is how ``tests/test_game_env.py`` runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CaptureRegion:
    """Pixel rectangle to grab.  ``monitor`` is the mss monitor index."""

    left: int = 0
    top: int = 0
    width: int = 1280
    height: int = 720
    monitor: int = 1

    def as_mss_dict(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "mon": self.monitor,
        }


class FrameSource(ABC):
    @abstractmethod
    def grab(self) -> np.ndarray:
        """Return an ``(h, w, 3)`` uint8 RGB frame."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class MSSCapture(FrameSource):  # pragma: no cover - needs a display
    """Screen capture through ``mss``."""

    def __init__(self, region: CaptureRegion | None = None) -> None:
        try:
            import mss
        except ImportError as exc:
            raise ImportError("screen capture needs mss: pip install -e '.[game]'") from exc
        self._mss = mss.mss()
        self.region = region or CaptureRegion()

    def grab(self) -> np.ndarray:
        raw = self._mss.grab(self.region.as_mss_dict())
        arr = np.asarray(raw)  # BGRA
        return arr[..., 2::-1].copy()  # -> RGB

    def close(self) -> None:
        self._mss.close()


class DummyCapture(FrameSource):
    """Synthetic frames: a drifting grating plus a shrinking HP bar.

    Enough structure that the motion detectors respond and the health-bar
    reader has something to read, so the game env can be tested end to end
    without a game.
    """

    def __init__(self, width: int = 320, height: int = 180, *, drift_px: float = 4.0) -> None:
        self.width, self.height = int(width), int(height)
        self.drift_px = float(drift_px)
        self.t = 0
        self.player_hp = 1.0

    def grab(self) -> np.ndarray:
        x = np.arange(self.width)
        row = 0.5 + 0.35 * np.sin(2 * np.pi * (x - self.drift_px * self.t) / 37.0)
        frame = np.repeat(row[None, :], self.height, axis=0)
        rgb = np.stack([frame] * 3, axis=-1)
        # red HP bar, draining slowly
        self.player_hp = max(0.0, 1.0 - self.t / 400.0)
        y0, y1 = int(0.045 * self.height), int(np.ceil(0.062 * self.height))
        x0 = int(0.035 * self.width)
        x1 = x0 + int((0.245 - 0.035) * self.width * self.player_hp)
        rgb[y0:y1, x0:x1] = (0.7, 0.05, 0.05)
        self.t += 1
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def make_frame_source(kind: str = "dummy", region: CaptureRegion | None = None) -> FrameSource:
    kind = kind.lower()
    if kind in ("mss", "screen", "real"):
        return MSSCapture(region)
    if kind in ("dummy", "synthetic", "test"):
        r = region or CaptureRegion()
        return DummyCapture(r.width, r.height)
    raise ValueError(f"unknown frame source {kind!r}")


__all__ = ["CaptureRegion", "DummyCapture", "FrameSource", "MSSCapture", "make_frame_source"]
