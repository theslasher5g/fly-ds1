"""The other step-2 route: the same connectome as a spiking network (Brian2).

The rate model is the one that trains; this exists because the roadmap's second
option is worth keeping open, and because some questions (spike timing, the
giant fibre's all-or-nothing escape, refractory effects) simply are not askable
of a rate model.

What carries over unchanged: the graph, the signs, the synapse counts and the
time constants -- everything in :class:`~flyds1.connectome.graph.WiredNetwork`.
What has to be invented, because a connectome does not contain it: resting and
threshold potentials, reset behaviour, refractory period, and the conversion
from "synapse count" to "post-synaptic potential amplitude".  Those are the
parameters of :class:`LIFParameters`, and they are guesses with a comment, not
measurements.

Caveats worth being explicit about:

* Rates and spikes are not interchangeable. A rate unit sitting at 1.0 is not
  "one spike per second" unless you make it so; ``rate_to_hz`` is that choice.
* Brian2 runs its own clock. There is no attempt here to interleave it with a
  game loop at 30 fps -- for closed-loop control use the rate model and treat
  this as an offline analysis tool.
* Brian2 currently requires numpy < 2, so this module is exercised only where
  that holds; its test skips otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyds1.connectome.graph import WiredNetwork


@dataclass
class LIFParameters:
    """Leaky integrate-and-fire parameters, in millivolts and milliseconds.

    None of these come from the connectome; they are standard LIF values chosen
    to put a neuron at a few tens of Hz under realistic drive.
    """

    v_rest_mV: float = -52.0        # fly neurons rest depolarised relative to mammals
    v_threshold_mV: float = -45.0
    v_reset_mV: float = -52.0
    refractory_ms: float = 2.0
    #: Millivolts of post-synaptic potential per unit of normalised weight.
    #: With in-degree normalisation a weight of 1.0 means "this neuron's entire
    #: input budget", so a few mV per unit puts a handful of synchronous inputs
    #: at threshold.
    psp_mV_per_weight: float = 5.0
    #: Resting drive, converted from the rate model's bias the same way.
    bias_mV_per_unit: float = 3.0
    #: Millivolts of injected current per unit of external drive.
    drive_mV_per_unit: float = 4.0
    #: Synaptic delay; keeps the network from resolving a loop within one step.
    delay_ms: float = 1.0


def require_brian2():
    try:
        import brian2
    except Exception as exc:  # noqa: BLE001 - brian2 fails on numpy>=2 with AttributeError
        raise ImportError(
            "the spiking backend needs a working Brian2 install "
            "(pip install -e '.[spiking]'; Brian2 currently requires numpy<2)"
        ) from exc
    return brian2


def build_spiking_network(
    network: WiredNetwork,
    params: LIFParameters | None = None,
    *,
    bias: np.ndarray | None = None,
    drive_mV: np.ndarray | None = None,
    record_all: bool = False,
):
    """Build a Brian2 ``Network`` from a :class:`WiredNetwork`.

    Returns ``(net, parts)`` where ``parts`` holds the ``NeuronGroup``, the
    ``Synapses``, and the monitors, so callers can inspect them after running.
    """
    b2 = require_brian2()
    params = params or LIFParameters()
    from flyds1.net.bias import resting_bias

    bias_units = resting_bias(network) if bias is None else np.asarray(bias, dtype=float)
    drive = np.zeros(network.n_neurons) if drive_mV is None else np.asarray(drive_mV, dtype=float)

    equations = """
    dv/dt = (v_rest - v + I_bias + I_drive) / tau : volt (unless refractory)
    tau : second
    I_bias : volt
    I_drive : volt
    """
    group = b2.NeuronGroup(
        network.n_neurons,
        equations,
        threshold="v > v_th",
        reset="v = v_reset",
        refractory=params.refractory_ms * b2.ms,
        method="euler",
        namespace={
            "v_rest": params.v_rest_mV * b2.mV,
            "v_th": params.v_threshold_mV * b2.mV,
            "v_reset": params.v_reset_mV * b2.mV,
        },
    )
    group.v = params.v_rest_mV * b2.mV
    group.tau = network.tau * b2.second
    group.I_bias = (bias_units * params.bias_mV_per_unit) * b2.mV
    group.I_drive = (drive * params.drive_mV_per_unit) * b2.mV

    coo = network.W.tocoo()
    synapses = b2.Synapses(group, group, model="w : volt", on_pre="v_post += w", delay=params.delay_ms * b2.ms)
    # W[post, pre]: rows are targets, columns are sources -- Brian2 wants (i=pre, j=post)
    synapses.connect(i=coo.col.astype(int), j=coo.row.astype(int))
    synapses.w = (coo.data * params.psp_mV_per_weight) * b2.mV

    monitors = {
        "descending": b2.SpikeMonitor(group[: 0]) if network.n_outputs == 0 else b2.SpikeMonitor(group),
    }
    if record_all:
        monitors["voltage"] = b2.StateMonitor(group, "v", record=True)

    net = b2.Network(group, synapses, *monitors.values())
    parts = {"group": group, "synapses": synapses, **monitors}
    return net, parts


def descending_rates_hz(parts: dict, network: WiredNetwork, duration_s: float) -> np.ndarray:
    """Spike counts of the descending neurons over the run, as Hz."""
    monitor = parts["descending"]
    counts = np.asarray(monitor.count)
    return counts[network.output_index] / max(duration_s, 1e-9)


def run_spiking(
    network: WiredNetwork,
    duration_ms: float = 500.0,
    *,
    params: LIFParameters | None = None,
    drive_units: np.ndarray | None = None,
    record_all: bool = False,
) -> dict:
    """Convenience: build, run, and report descending firing rates."""
    b2 = require_brian2()
    drive = None
    if drive_units is not None:
        drive = np.zeros(network.n_neurons)
        drive[network.input_index] = np.asarray(drive_units, dtype=float).ravel()
    net, parts = build_spiking_network(network, params, drive_mV=drive, record_all=record_all)
    net.run(duration_ms * b2.ms)
    return {
        "parts": parts,
        "descending_hz": descending_rates_hz(parts, network, duration_ms / 1000.0),
        "total_spikes": int(np.asarray(parts["descending"].count).sum()),
    }


__all__ = ["LIFParameters", "build_spiking_network", "descending_rates_hz", "run_spiking"]
