"""Resting bias -- why the network needs one to be alive at all.

Photoreceptors are histaminergic and histamine *inhibits* L1/L2: in the fly,
light hyperpolarises the first interneurons rather than exciting them.  With
rectified rates and zero bias that circuit is silent by construction -- there
is nothing for an inhibitory input to reduce -- and the whole brain outputs
zero regardless of what is on screen.

Biologically the answer is that lamina and central neurons are tonically
active; the bias term is that resting drive.  It is a genuine model parameter,
not a fudge: the connectome fixes *who talks to whom*, never the resting
potential, so this has to come from somewhere else (physiology, or training).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from flyds1.connectome.graph import WiredNetwork

#: Resting drive per super-class.  Sensory neurons get none -- their drive is
#: the screen.  The rest sit at a positive baseline so inhibition has room to
#: act in both directions.
DEFAULT_RESTING_BIAS: Mapping[str, float] = {
    "sensory": 0.0,
    "optic": 1.0,
    "visual_projection": 1.0,
    "central": 1.0,
    "descending": 1.0,
    "ascending": 1.0,
    "motor": 1.0,
}
DEFAULT_RESTING_FALLBACK = 1.0


def resting_bias(
    network: WiredNetwork,
    *,
    table: Mapping[str, float] | None = None,
    fallback: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Per-neuron resting bias from its super-class."""
    table = DEFAULT_RESTING_BIAS if table is None else table
    fb = DEFAULT_RESTING_FALLBACK if fallback is None else fallback
    values = np.array(
        [float(table.get(sc, fb)) for sc in network.neurons.super_class], dtype=float
    )
    return values * float(scale)


def homeostatic_bias(
    network: WiredNetwork,
    target_rate: float = 1.0,
    *,
    iterations: int = 200,
    lr: float = 0.5,
    config=None,
) -> np.ndarray:
    """Fit a bias that puts every neuron near ``target_rate`` at rest.

    A cheap stand-in for the physiology we do not have: run the network with no
    input and push each neuron's bias up or down until its spontaneous rate is
    close to the target.  Gives a network that is neither silent nor saturated
    before any training starts.
    """
    from flyds1.net.dynamics import RateConfig
    from flyds1.net.reference import RateNetwork

    cfg = config or RateConfig()
    bias = resting_bias(network)
    for _ in range(iterations):
        sim = RateNetwork(network, cfg, bias=bias)
        rates, _ = sim.fixed_point(None, max_steps=2000)
        error = target_rate - rates[0]
        bias = bias + lr * error
        # sensory neurons stay at zero: they are driven by the eye, not by bias
        bias[network.input_index] = 0.0
        if np.max(np.abs(error)) < 1e-3:
            break
    return bias


__all__ = ["DEFAULT_RESTING_BIAS", "homeostatic_bias", "resting_bias"]
