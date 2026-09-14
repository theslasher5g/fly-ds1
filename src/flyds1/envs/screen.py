"""Grabbing frames from a monitor, a game window, or a recording.

Three sources:

* :class:`MSSCapture` -- the live screen, the fast path (one blit per grab, no
  PIL round-trip).
* :class:`ReplayCapture` -- frames from disk. This is how you check the whole
  game path against *real* Dark Souls frames without the game running: record a
  clip, dump frames into a folder, and the retina, the brain and the reward
  read-out all run against them headless, deterministically, as often as you
  like. Debugging a perception pipeline against a live game is miserable --
  nothing is reproducible.
* :class:`DummyCapture` -- synthetic frames, for tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

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


class ReplayCapture(FrameSource):
    """Frames from a directory of images, or from a ``.npy`` stack.

    Images are read with Pillow if it is installed, otherwise ``.npy``/``.npz``
    only -- so the dependency stays optional. ``loop`` decides whether the
    recording repeats or the source reports exhaustion by raising
    ``StopIteration``.
    """

    def __init__(self, path: str | Path, *, loop: bool = True, stride: int = 1) -> None:
        self.path = Path(path)
        self.loop = bool(loop)
        self.stride = max(1, int(stride))
        self.index = 0
        self._frames: np.ndarray | None = None
        self._files: list[Path] = []

        if self.path.is_dir():
            patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            self._files = sorted(
                f for pattern in patterns for f in self.path.glob(pattern)
            )[:: self.stride]
            if not self._files:
                npy = sorted(self.path.glob("*.npy")) + sorted(self.path.glob("*.npz"))
                if npy:
                    self._frames = _load_stack(npy[0])[:: self.stride]
        elif self.path.suffix in (".npy", ".npz"):
            self._frames = _load_stack(self.path)[:: self.stride]
        else:
            self._files = [self.path]

        if self._frames is None and not self._files:
            raise FileNotFoundError(
                f"no frames in {self.path} (looked for png/jpg/bmp images and .npy/.npz stacks)"
            )

    def __len__(self) -> int:
        return len(self._frames) if self._frames is not None else len(self._files)

    def grab(self) -> np.ndarray:
        n = len(self)
        if self.index >= n:
            if not self.loop:
                raise StopIteration(f"recording exhausted after {n} frames")
            self.index = 0
        frame = (
            self._frames[self.index]
            if self._frames is not None
            else _load_image(self._files[self.index])
        )
        self.index += 1
        return np.asarray(frame)

    def reset(self) -> None:
        self.index = 0


def _load_stack(path: Path) -> np.ndarray:
    data = np.load(path)
    if hasattr(data, "files"):  # npz
        data = data[data.files[0]]
    arr = np.asarray(data)
    if arr.ndim == 3:  # a single frame
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"{path} must hold (n, h, w, 3) frames, got shape {arr.shape}")
    return arr[..., :3]


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            f"reading {path.suffix} frames needs Pillow (pip install pillow); "
            "or convert the recording to a .npy stack of (n, h, w, 3) uint8"
        ) from exc
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def make_frame_source(
    kind: str = "dummy",
    region: CaptureRegion | None = None,
    *,
    path: str | Path | None = None,
) -> FrameSource:
    kind = kind.lower()
    if kind in ("mss", "screen", "real"):
        return MSSCapture(region)
    if kind in ("replay", "recording", "folder"):
        if path is None:
            raise ValueError("the replay source needs a path to the recording")
        return ReplayCapture(path)
    if kind in ("dummy", "synthetic", "test"):
        r = region or CaptureRegion()
        return DummyCapture(r.width, r.height)
    raise ValueError(f"unknown frame source {kind!r}")


__all__ = [
    "CaptureRegion",
    "DummyCapture",
    "FrameSource",
    "MSSCapture",
    "ReplayCapture",
    "make_frame_source",
]
