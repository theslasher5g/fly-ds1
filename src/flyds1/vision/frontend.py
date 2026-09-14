"""Step 3c: the retina as the environment sees it.

This is the *fixed*, non-trainable part of vision -- optics and elementary
motion detection are physics, not parameters -- so it lives on the environment
side and its output is the observation:

    screen frame -> per-photoreceptor contrast  (+ optional motion channels)

The trainable encoder (observation -> synaptic drive) is separate, in
:mod:`flyds1.vision.encoder`, because that part the optimiser is allowed to
touch.

Two eyes: photoreceptor coordinates are centred per eye in the connectome, so
each eye's gaze is offset in azimuth.  A real Drosophila looks mostly sideways
with ~17 deg of frontal binocular overlap; a monitor sits in a ~60 deg patch
straight ahead.  We therefore aim both eyes at the screen with a configurable
azimuth offset -- a deliberate deviation from fly anatomy, made so the animal
can see the game at all, and the one place where "biologically faithful" had to
yield to "the task exists".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyds1.connectome.graph import WiredNetwork
from flyds1.connectome.retinotopy import snap_to_hex_lattice
from flyds1.vision.ommatidia import (
    ACCEPTANCE_FWHM_DEG,
    OmmatidiaSampler,
    ScreenGeometry,
    luminance,
    resize_nearest,
)
from flyds1.vision.reichardt import ReichardtBank, ReichardtConfig


@dataclass
class FrontEndConfig:
    """Optics and pre-processing of the retina."""

    screen: ScreenGeometry = field(default_factory=ScreenGeometry)
    acceptance_fwhm_deg: float = ACCEPTANCE_FWHM_DEG
    #: Azimuth offset of each eye's centre, degrees.  Left eye gets ``-offset``.
    eye_azimuth_offset_deg: float = 15.0
    spacing_deg: float = 5.0
    reichardt: ReichardtConfig = field(default_factory=ReichardtConfig)
    #: ``"retinotopic"`` keeps motion per facet, ``"pooled"`` averages it per
    #: eye (6 numbers instead of 6 * n_facets), ``"off"`` drops motion.
    motion_mode: str = "retinotopic"
    #: Weber-contrast the facet intensities before they leave the retina.  This
    #: is the photoreceptor's own light adaptation; without it the network sees
    #: absolute screen brightness and a dark room changes everything.
    adapt_contrast: bool = True
    contrast_tau: float = 0.250
    contrast_eps: float = 0.05


@dataclass
class ObsLayout:
    """Where things sit in the flat observation vector."""

    n_photoreceptors: int
    motion_shape: tuple[int, ...]
    motion_mode: str
    photoreceptor_slice: slice
    motion_slice: slice
    size: int

    def split(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat observation -> ``(photoreceptor, motion)`` with motion reshaped."""
        arr = np.asarray(obs)
        photo = arr[..., self.photoreceptor_slice]
        motion = arr[..., self.motion_slice]
        if self.motion_shape:
            motion = motion.reshape(arr.shape[:-1] + self.motion_shape)
        return photo, motion


class RetinaFrontEnd:
    """Turns frames into observations for one :class:`WiredNetwork`."""

    def __init__(self, network: WiredNetwork, config: FrontEndConfig | None = None) -> None:
        self.cfg = config or FrontEndConfig()
        self.network = network
        coords = network.eye_coords()
        if len(coords) == 0:
            raise ValueError(
                "this network has no photoreceptors with eye coordinates; "
                "step 3 needs retinotopy (see flyds1.connectome.retinotopy)"
            )
        sides = network.neurons.side[network.input_index]

        self.eyes: dict[str, dict] = {}
        for side in sorted({str(s) for s in sides}):
            slots = np.flatnonzero(sides == side)
            eye_coords = coords[slots].copy()
            offset = self.cfg.eye_azimuth_offset_deg * (-1.0 if side == "left" else 1.0)
            gaze = eye_coords + np.array([offset, 0.0])
            sampler = OmmatidiaSampler(
                gaze,
                self.cfg.screen,
                acceptance_fwhm_deg=self.cfg.acceptance_fwhm_deg,
            )
            axial = snap_to_hex_lattice(eye_coords, self.cfg.spacing_deg)
            bank = ReichardtBank(axial, self.cfg.reichardt, spacing_deg=self.cfg.spacing_deg)
            self.eyes[side] = {
                "slots": slots,
                "coords": eye_coords,
                "sampler": sampler,
                "bank": bank,
                "adapt": None,
            }

        self.n_photoreceptors = len(coords)
        self.n_axes = next(iter(self.eyes.values()))["bank"].n_axes
        self.n_motion_channels = next(iter(self.eyes.values()))["bank"].n_channels
        self.layout = self._build_layout()
        self.reset()

    # ------------------------------------------------------------------
    def _build_layout(self) -> ObsLayout:
        n_photo = self.n_photoreceptors
        mode = self.cfg.motion_mode
        if mode == "off":
            motion_shape: tuple[int, ...] = ()
            motion_size = 0
        elif mode == "pooled":
            motion_shape = (len(self.eyes), self.n_motion_channels)
            motion_size = int(np.prod(motion_shape))
        elif mode == "retinotopic":
            motion_shape = (n_photo, self.n_motion_channels)
            motion_size = int(np.prod(motion_shape))
        else:
            raise ValueError(f"unknown motion_mode {mode!r}")
        return ObsLayout(
            n_photoreceptors=n_photo,
            motion_shape=motion_shape,
            motion_mode=mode,
            photoreceptor_slice=slice(0, n_photo),
            motion_slice=slice(n_photo, n_photo + motion_size),
            size=n_photo + motion_size,
        )

    @property
    def obs_size(self) -> int:
        return self.layout.size

    def coverage(self) -> dict[str, float]:
        """Fraction of each eye's facets that can see the monitor."""
        return {side: eye["sampler"].coverage() for side, eye in self.eyes.items()}

    def reset(self) -> None:
        for eye in self.eyes.values():
            eye["bank"].reset(1)
            eye["adapt"] = None

    # ------------------------------------------------------------------
    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Bring a captured frame to ``(H, W)`` luma at screen resolution."""
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = luminance(arr[..., :3])
        elif arr.ndim == 3 and arr.shape[0] in (3, 4):  # channel-first
            arr = luminance(np.moveaxis(arr[:3], 0, -1))
        arr = np.asarray(arr, dtype=float)
        if arr.max(initial=0.0) > 1.5:
            arr = arr / 255.0
        target = (self.cfg.screen.height, self.cfg.screen.width)
        if arr.shape != target:
            arr = resize_nearest(arr, *target)
        return arr

    def process(self, frame: np.ndarray, dt: float) -> np.ndarray:
        """Frame -> flat observation vector (float32)."""
        luma = self.prepare_frame(frame)
        photo = np.zeros(self.n_photoreceptors, dtype=float)
        motion_parts: list[np.ndarray] = []

        for side in sorted(self.eyes):
            eye = self.eyes[side]
            intensity = eye["sampler"].sample(luma)  # (n_facets,)
            signal = self._adapt(eye, intensity, dt)
            photo[eye["slots"]] = signal
            if self.cfg.motion_mode == "off":
                continue
            motion = eye["bank"].step(intensity[None, :], dt)[0]  # (n_facets, ch)
            motion_parts.append(motion)

        if self.cfg.motion_mode == "retinotopic":
            motion_out = np.zeros((self.n_photoreceptors, self.n_motion_channels), dtype=float)
            for side, motion in zip(sorted(self.eyes), motion_parts):
                motion_out[self.eyes[side]["slots"]] = motion
            flat_motion = motion_out.ravel()
        elif self.cfg.motion_mode == "pooled":
            flat_motion = np.stack([m.mean(axis=0) for m in motion_parts]).ravel()
        else:
            flat_motion = np.zeros(0)

        return np.concatenate([photo, flat_motion]).astype(np.float32)

    def _adapt(self, eye: dict, intensity: np.ndarray, dt: float) -> np.ndarray:
        if not self.cfg.adapt_contrast:
            return intensity
        state = eye["adapt"]
        if state is None or state.shape != intensity.shape:
            state = intensity.copy()
        alpha = float(np.clip(dt / self.cfg.contrast_tau, 0.0, 1.0))
        state = state + alpha * (intensity - state)
        eye["adapt"] = state
        return (intensity - state) / (state + self.cfg.contrast_eps)

    # ------------------------------------------------------------------
    def mean_flow(self) -> dict[str, np.ndarray]:
        """Last frame's mean optic flow per eye (diagnostics / reward shaping)."""
        out = {}
        for side, eye in self.eyes.items():
            bank: ReichardtBank = eye["bank"]
            out[side] = bank.mean_flow(np.zeros((1, len(eye["slots"]), bank.n_channels)))[0]
        return out

    def describe(self) -> str:
        cov = ", ".join(f"{s}={v:.0%}" for s, v in sorted(self.coverage().items()))
        return (
            f"RetinaFrontEnd: {self.n_photoreceptors} photoreceptors over {len(self.eyes)} eyes, "
            f"screen {self.cfg.screen.width}x{self.cfg.screen.height} "
            f"({self.cfg.screen.fov_h_deg:.0f}x{self.cfg.screen.fov_v_deg:.0f} deg)\n"
            f"  screen coverage per eye: {cov}\n"
            f"  motion: {self.cfg.motion_mode} "
            f"({self.n_motion_channels} channels/facet) -> obs size {self.obs_size}"
        )


__all__ = ["FrontEndConfig", "ObsLayout", "RetinaFrontEnd"]
