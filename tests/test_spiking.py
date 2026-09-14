"""The Brian2 route.  Skipped wherever Brian2 does not import cleanly --
it currently requires numpy<2, so most modern environments skip this file."""

import numpy as np
import pytest


def _brian2_or_skip():
    try:
        import brian2  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - numpy>=2 raises AttributeError, not ImportError
        pytest.skip(f"Brian2 unavailable in this environment: {type(exc).__name__}")
    return brian2


def test_require_brian2_explains_itself():
    from flyds1.net.spiking import require_brian2

    try:
        require_brian2()
    except ImportError as exc:
        assert "spiking" in str(exc)


def test_spiking_network_mirrors_the_graph(network):
    _brian2_or_skip()
    from flyds1.net.spiking import build_spiking_network

    _, parts = build_spiking_network(network)
    assert len(parts["group"]) == network.n_neurons
    assert len(parts["synapses"]) == network.W.nnz
    # Brian2 indexes (i=pre, j=post); the matrix is W[post, pre]
    coo = network.W.tocoo()
    assert set(zip(parts["synapses"].i[:], parts["synapses"].j[:])) == set(
        zip(coo.col.tolist(), coo.row.tolist())
    )


def test_spiking_network_responds_to_drive(network):
    _brian2_or_skip()
    from flyds1.net.spiking import run_spiking

    quiet = run_spiking(network, 200.0)
    driven = run_spiking(network, 200.0, drive_units=np.full(network.n_inputs, 3.0))
    assert driven["total_spikes"] != quiet["total_spikes"]
