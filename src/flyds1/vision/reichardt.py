"""Step 3b: elementary motion detection in front of the network.

The fly visual system is motion-driven, and the canonical model of an
elementary motion detector (EMD) is the Hassenstein-Reichardt correlator: take
two neighbouring facets, delay one, multiply with the undelayed other, and
subtract the mirrored term:

    out(i -> j) = lowpass(s_i) * s_j - lowpass(s_j) * s_i

Positive output means a pattern moved from facet ``i`` towards facet ``j``.
The full chain implemented here mirrors the biology in three stages:

1. **Light adaptation / contrast** -- photoreceptors and lamina cells respond to
   *relative* change, not absolute luminance.  A slow luminance lowpass per
   facet gives Weber contrast ``(I - I_slow) / (I_slow + eps)``, which is what
   makes the detectors invariant to a dark cave vs. a bright bonfire.
2. **ON/OFF split** -- half-wave rectification into brightness increments and
   decrements, the T4 (ON) and T5 (OFF) pathways.
3. **Correlation** -- along each of the three hexagonal lattice axes.

The output is *optional* for the network: the connectome built in step 1
already contains the T4/T5 motif, so these detectors are a pre-processing
short-cut, not a requirement.  Keep them when you want motion handed to the
network on a plate; switch them off when you want to ask whether the
connectome's own circuitry can do it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyds1.hexlattice import HEX_AXES, axis_directions, neighbour_pairs


@dataclass
class ReichardtConfig:
    """Time constants of the detector chain, in seconds."""

    tau_delay: float = 0.035      # delay line; sets the preferred velocity
    tau_adapt: float = 0.250      # luminance adaptation (contrast normalisation)
    contrast_eps: float = 0.05    # avoids division blow-up in near-black frames
    on_off_split: bool = True     # separate ON and OFF pathways (T4 / T5)
    gain: float = 1.0

    def __post_init__(self) -> None:
        for name in ("tau_delay", "tau_adapt"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class ReichardtBank:
    """A bank of EMDs, one per (facet, lattice axis) with a neighbour.

    Stateful: call :meth:`reset` per episode and :meth:`step` per frame.
    """

    def __init__(
        self,
        axial: np.ndarray,
        config: ReichardtConfig | None = None,
        *,
        spacing_deg: float = 5.0,
    ) -> None:
        self.axial = np.asarray(axial, dtype=np.int64)
        self.n_facets = len(self.axial)
        self.cfg = config or ReichardtConfig()
        self.pairs = neighbour_pairs(self.axial)
        self.spacing_deg = float(spacing_deg)
        self._axis_dirs = axis_directions(spacing_deg)
        self.reset()

    # ------------------------------------------------------------------
    @property
    def n_axes(self) -> int:
        return len(HEX_AXES)

    @property
    def n_channels(self) -> int:
        """Motion channels per facet: 3 axes, doubled if ON/OFF are split."""
        return self.n_axes * (2 if self.cfg.on_off_split else 1)

    def reset(self, batch_size: int = 1) -> None:
        shape = (batch_size, self.n_facets)
        self._adapt: np.ndarray | None = None
        self._delay = np.zeros(shape + (2 if self.cfg.on_off_split else 1,), dtype=float)
        self._batch = batch_size

    # ------------------------------------------------------------------
    def contrast(self, intensity: np.ndarray, dt: float) -> np.ndarray:
        """Weber contrast against an adapting luminance estimate."""
        x = np.atleast_2d(np.asarray(intensity, dtype=float))
        if self._adapt is None or self._adapt.shape != x.shape:
            self._adapt = x.copy()  # first frame: no transient from cold start
        alpha = float(np.clip(dt / self.cfg.tau_adapt, 0.0, 1.0))
        self._adapt = self._adapt + alpha * (x - self._adapt)
        return (x - self._adapt) / (self._adapt + self.cfg.contrast_eps)

    def step(self, intensity: np.ndarray, dt: float) -> np.ndarray:
        """One frame in, ``(batch, n_facets, n_channels)`` motion out.

        Channels are ordered ``[axis0_on, axis1_on, axis2_on, axis0_off, ...]``
        when ``on_off_split`` is set, otherwise one signed channel per axis.
        A positive value means motion along the positive direction of that
        lattice axis (see :data:`flyds1.hexlattice.HEX_AXES`).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        x = np.atleast_2d(np.asarray(intensity, dtype=float))
        if x.shape[1] != self.n_facets:
            raise ValueError(f"expected {self.n_facets} facets, got {x.shape[1]}")
        if x.shape[0] != self._batch:
            self.reset(x.shape[0])

        c = self.contrast(x, dt)
        if self.cfg.on_off_split:
            signal = np.stack([np.maximum(c, 0.0), np.maximum(-c, 0.0)], axis=-1)
        else:
            signal = c[..., None]

        alpha = float(np.clip(dt / self.cfg.tau_delay, 0.0, 1.0))
        self._delay = self._delay + alpha * (signal - self._delay)
        delayed = self._delay

        out = np.zeros((x.shape[0], self.n_facets, self.n_channels), dtype=float)
        n_path = signal.shape[-1]
        for axis in range(self.n_axes):
            pairs = self.pairs.get(axis, np.zeros((0, 2), dtype=np.int64))
            if len(pairs) == 0:
                continue
            i, j = pairs[:, 0], pairs[:, 1]
            for path in range(n_path):
                s_i, s_j = signal[:, i, path], signal[:, j, path]
                d_i, d_j = delayed[:, i, path], delayed[:, j, path]
                # correlate delayed with undelayed, both ways round
                corr = d_i * s_j - d_j * s_i
                out[:, i, path * self.n_axes + axis] = self.cfg.gain * corr
        return out

    # ------------------------------------------------------------------
    def flow_field(self, motion: np.ndarray) -> np.ndarray:
        """Collapse axis channels into a 2D flow vector per facet.

        ``(batch, n_facets, n_channels)`` -> ``(batch, n_facets, 2)`` in eye
        coordinates.  Diagnostics and reward shaping, not network input.
        """
        m = np.asarray(motion, dtype=float)
        if self.cfg.on_off_split:
            m = m[..., : self.n_axes] + m[..., self.n_axes :]
        return m @ self._axis_dirs

    def mean_flow(self, motion: np.ndarray) -> np.ndarray:
        """Average flow vector over the retina -- a crude self-motion estimate."""
        flow = self.flow_field(motion)
        return flow.mean(axis=1)


def rectified_direction_channels(motion: np.ndarray, n_axes: int = 3) -> np.ndarray:
    """Split signed axis channels into half-wave rectified direction channels.

    Turns 3 signed values into the 6 non-negative ones a T4a-d/T5a-d style
    population would carry, which is what a rate network with non-negative
    activations actually wants.
    """
    m = np.asarray(motion, dtype=float)
    return np.concatenate([np.maximum(m, 0.0), np.maximum(-m, 0.0)], axis=-1)


__all__ = ["ReichardtBank", "ReichardtConfig", "rectified_direction_channels"]
