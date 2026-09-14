"""Does a signal actually reach the descending neurons?

The question that decides whether any of this can learn.  A connectome-derived
matrix is not automatically a working network: scale the weights too low and the
input dies out layer by layer (the DN rates end up identical no matter what is
on screen, and PPO sees a constant observation); scale them too high and the
network saturates or runs away.

:func:`signal_transmission` measures the first failure directly -- drive the eye
with different random patterns and look at how much each population's rates
differ between them -- and :func:`gain_sweep` turns that into a one-line answer
to "what gain should I use?".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyds1.connectome.graph import WiredNetwork, build_wired_network
from flyds1.connectome.schema import Connectome
from flyds1.net.dynamics import RateConfig
from flyds1.net.reference import RateNetwork


@dataclass
class TransmissionReport:
    """Per-population response variability across different stimuli."""

    per_class: dict[str, float]
    descending_std: float
    descending_mean: float
    max_rate: float
    spectral_radius: float

    def describe(self) -> str:
        chain = " -> ".join(
            f"{name}:{value:.2e}"
            for name, value in sorted(self.per_class.items(), key=lambda kv: -kv[1])
        )
        return (
            f"signal transmission (std of rates across stimuli)\n"
            f"  {chain}\n"
            f"  descending: mean rate {self.descending_mean:.3f}, "
            f"stimulus-driven std {self.descending_std:.2e}, "
            f"peak rate {self.max_rate:.2f}, spectral radius {self.spectral_radius:.3f}"
        )

    @property
    def usable(self) -> bool:
        """Is the DN signal big enough to survive standardisation and PPO?

        The threshold is the standardiser's epsilon floor (1e-5 on the variance,
        so ~3e-3 on the standard deviation) with an order of magnitude of margin.
        """
        return self.descending_std > 3e-4 and np.isfinite(self.max_rate) and self.max_rate < 1e3


def signal_transmission(
    network: WiredNetwork,
    config: RateConfig | None = None,
    *,
    n_stimuli: int = 24,
    steps: int = 12,
    amplitude: float = 2.0,
    seed: int = 0,
) -> TransmissionReport:
    """Drive the eye with random patterns; report how far the differences reach."""
    cfg = config or RateConfig()
    rng = np.random.default_rng(seed)
    sim = RateNetwork(network, cfg)
    stimuli = rng.random((n_stimuli, network.n_inputs)) * amplitude

    rates = []
    for stimulus in stimuli:
        sim.reset()
        drive = sim.drive_from_inputs(stimulus[None, :])
        for _ in range(steps):
            sim.step(drive)
        rates.append(sim.r[0].copy())
    rates = np.stack(rates)

    per_class: dict[str, float] = {}
    for name in sorted(set(network.neurons.super_class)):
        mask = np.array([sc == name for sc in network.neurons.super_class])
        if mask.any():
            per_class[str(name)] = float(rates[:, mask].std(axis=0).mean())

    dn = rates[:, network.output_index]
    return TransmissionReport(
        per_class=per_class,
        descending_std=float(dn.std(axis=0).mean()) if dn.size else 0.0,
        descending_mean=float(dn.mean()) if dn.size else 0.0,
        max_rate=float(rates.max()),
        spectral_radius=network.spectral_radius(),
    )


def gain_sweep(
    connectome: Connectome,
    gains: list[float] | tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
    *,
    normalise: str = "in_degree",
    config: RateConfig | None = None,
    min_synapse_count: float = 5.0,
    **probe_kwargs,
) -> list[tuple[float, TransmissionReport]]:
    """Rebuild the network at several gains and measure each."""
    out = []
    for gain in gains:
        net = build_wired_network(
            connectome,
            normalise=normalise,  # type: ignore[arg-type]
            gain=float(gain),
            min_synapse_count=min_synapse_count,
        )
        out.append((float(gain), signal_transmission(net, config, **probe_kwargs)))
    return out


def recommend_gain(
    sweep: list[tuple[float, TransmissionReport]], *, max_radius: float = 0.95
) -> float | None:
    """Pick the gain with the strongest DN signal that stays sub-critical.

    Sub-critical (spectral radius < 1) matters: above it the linearised dynamics
    have no stable fixed point, and what looks like a responsive network is often
    just a runaway one.
    """
    usable = [
        (report.descending_std, gain)
        for gain, report in sweep
        if report.spectral_radius < max_radius and report.usable
    ]
    if not usable:
        return None
    return max(usable)[1]


def format_sweep(sweep: list[tuple[float, TransmissionReport]]) -> str:
    lines = [f"{'gain':>6s} {'radius':>8s} {'DN mean':>9s} {'DN std':>10s} {'peak rate':>10s}  usable"]
    for gain, report in sweep:
        lines.append(
            f"{gain:6.2f} {report.spectral_radius:8.3f} {report.descending_mean:9.3f} "
            f"{report.descending_std:10.2e} {report.max_rate:10.2f}  {'yes' if report.usable else 'no'}"
        )
    return "\n".join(lines)


__all__ = [
    "TransmissionReport",
    "format_sweep",
    "gain_sweep",
    "recommend_gain",
    "signal_transmission",
]
