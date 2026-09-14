"""Shared definition of the rate dynamics.

One equation, used by both backends (numpy reference and PyTorch), so they
cannot drift apart:

    tau_i dv_i/dt = -v_i + sum_j W_ij r_j + b_i + I_i(t),      r_i = f(v_i)

``v`` is a membrane-potential-like state, ``r >= 0`` a firing rate, ``W`` the
signed connectome matrix (``W[post, pre]``), ``I`` the external drive from the
eye.  Forward Euler with step ``dt``:

    v <- v + (dt / tau) * (-v + W r + b + I)

The integration is stable only while ``dt <= tau``; :func:`check_timestep`
enforces that, because the failure mode otherwise is a network that oscillates
at the Nyquist frequency and looks like "interesting dynamics".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Activation = Literal["relu", "softplus", "tanh", "sigmoid", "identity"]


@dataclass
class RateConfig:
    """Simulation parameters of the rate network."""

    dt: float = 1.0 / 240.0      # integration step (s); 4 substeps per 60 Hz frame
    activation: Activation = "relu"
    softplus_beta: float = 4.0
    v_init: float = 0.0
    #: Clip rates to keep a connectome with an unlucky gain from exploding.
    #: ``None`` disables clipping.
    rate_clip: float | None = 100.0
    #: Substeps of length ``dt`` per environment frame.
    steps_per_frame: int = 4

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.steps_per_frame < 1:
            raise ValueError("steps_per_frame must be >= 1")

    @property
    def frame_dt(self) -> float:
        """Wall-clock duration one environment frame covers."""
        return self.dt * self.steps_per_frame


def check_timestep(dt: float, tau: np.ndarray, *, margin: float = 1.0) -> None:
    """Raise if forward Euler would be unstable for the fastest neuron."""
    tau_min = float(np.min(tau))
    if dt > margin * tau_min:
        raise ValueError(
            f"dt={dt:.5f}s exceeds the smallest time constant tau={tau_min:.5f}s; "
            f"forward Euler needs dt <= tau (try dt={tau_min / 2:.5f} or raise tau)"
        )


def activate(v: np.ndarray, cfg: RateConfig) -> np.ndarray:
    """Apply the configured activation function (numpy)."""
    kind = cfg.activation
    if kind == "relu":
        r = np.maximum(v, 0.0)
    elif kind == "softplus":
        beta = cfg.softplus_beta
        r = np.logaddexp(0.0, beta * v) / beta
    elif kind == "tanh":
        r = np.tanh(v)
    elif kind == "sigmoid":
        r = 1.0 / (1.0 + np.exp(-v))
    elif kind == "identity":
        r = v
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown activation {kind!r}")
    if cfg.rate_clip is not None:
        r = np.minimum(r, cfg.rate_clip)
    return r


__all__ = ["Activation", "RateConfig", "activate", "check_timestep"]
