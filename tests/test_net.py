import numpy as np
import pytest

from flyds1.net.bias import homeostatic_bias, resting_bias
from flyds1.net.diagnostics import gain_sweep, recommend_gain, signal_transmission
from flyds1.net.dynamics import RateConfig, check_timestep
from flyds1.net.reference import RateNetwork

torch = pytest.importorskip("torch")
from flyds1.net.rate_rnn import ConnectomeRNN, TrainableParts  # noqa: E402


def test_timestep_guard(network):
    check_timestep(0.001, network.tau)
    with pytest.raises(ValueError, match="exceeds the smallest"):
        check_timestep(1.0, network.tau)


def test_resting_state_is_alive_not_saturated(network):
    sim = RateNetwork(network, RateConfig())
    rates, converged = sim.fixed_point(None, max_steps=8000)
    assert converged
    assert rates.max() > 0.0, "a silent network cannot compute anything"
    assert np.isfinite(rates).all()


def test_zero_bias_would_silence_the_network(network):
    """The reason flyds1.net.bias exists: histaminergic photoreceptors."""
    sim = RateNetwork(network, RateConfig(), bias=np.zeros(network.n_neurons))
    rates, _ = sim.fixed_point(None, max_steps=2000)
    assert rates.max() == 0.0
    assert resting_bias(network).max() > 0.0


def test_photoreceptors_get_no_resting_bias(network):
    bias = resting_bias(network)
    assert np.allclose(bias[network.input_index], 0.0)


def test_homeostatic_bias_hits_its_target(network):
    bias = homeostatic_bias(network, target_rate=1.0, iterations=80)
    sim = RateNetwork(network, RateConfig(), bias=bias)
    rates, _ = sim.fixed_point(None, max_steps=4000)
    assert abs(float(rates.mean()) - 1.0) < 0.2


def test_output_depends_on_input(network):
    sim = RateNetwork(network, RateConfig())
    rng = np.random.default_rng(0)
    outs = []
    for _ in range(4):
        sim.reset()
        drive = sim.drive_from_inputs(rng.random((1, network.n_inputs)) * 2)
        for _ in range(15):
            sim.step(drive)
        outs.append(sim.outputs()[0].copy())
    spread = np.stack(outs).std(axis=0).mean()
    assert spread > 1e-4, f"descending neurons ignore the stimulus (spread {spread:.2e})"


def test_backends_agree(network):
    cfg = RateConfig()
    rng = np.random.default_rng(1)
    drives = rng.random((10, network.n_inputs))

    sim = RateNetwork(network, cfg)
    reference = np.stack([sim.step(sim.drive_from_inputs(d))[0] for d in drives])

    rnn = ConnectomeRNN(network, cfg, dtype=torch.float64)
    v = rnn.init_state(1)
    rates = []
    for d in drives:
        drive = rnn.scatter_drive(torch.tensor(d, dtype=torch.float64).unsqueeze(0))
        v, r = rnn(v, drive)
        rates.append(r.detach().numpy()[0])
    assert np.abs(reference - np.stack(rates)).max() < 1e-10


def test_connectome_weights_are_frozen(network):
    rnn = ConnectomeRNN(network, RateConfig(), TrainableParts(bias=True, plastic_edges=10))
    assert not rnn.w_val.requires_grad
    trainable = {name for name, p in rnn.named_parameters() if p.requires_grad}
    assert "bias" in trainable
    assert "plastic_val" in trainable
    assert not any("w_val" in name for name in trainable)


def test_gradients_reach_trainable_parts(network):
    rnn = ConnectomeRNN(network, RateConfig(), TrainableParts(bias=True, plastic_edges=20))
    v = rnn.init_state(3)
    drive = torch.rand(3, network.n_inputs, requires_grad=True)
    for _ in range(3):
        v, r = rnn(v, rnn.scatter_drive(drive))
    rnn.read_outputs(r).pow(2).mean().backward()
    assert drive.grad.abs().sum() > 0
    assert rnn.bias.grad.abs().sum() > 0
    assert rnn.plastic_val.grad.abs().sum() > 0


def test_plastic_edges_do_not_duplicate_existing_ones(network):
    rnn = ConnectomeRNN(network, RateConfig(), TrainableParts(plastic_edges=40, seed=1))
    existing = set(zip(rnn.w_post.tolist(), rnn.w_pre.tolist()))
    new = list(zip(rnn.p_post.tolist(), rnn.p_pre.tolist()))
    assert len(new) == len(set(new))
    assert not (set(new) & existing)


def test_detached_state_stops_gradient_flow(network):
    rnn = ConnectomeRNN(network, RateConfig(), TrainableParts(bias=True))
    v = rnn.init_state(1)
    v, _ = rnn.step(v, None, detach=True)
    assert not v.requires_grad


def test_signal_transmission_report(network):
    report = signal_transmission(network, RateConfig(), n_stimuli=6, steps=8)
    assert set(report.per_class) >= {"sensory", "central", "descending"}
    assert report.spectral_radius < 1.0
    assert report.usable


def test_gain_sweep_recommends_a_subcritical_gain(connectome):
    sweep = gain_sweep(connectome, (0.5, 1.0, 3.0), n_stimuli=4, steps=6)
    best = recommend_gain(sweep)
    assert best in (0.5, 1.0, 3.0)
    chosen = dict((g, r) for g, r in sweep)[best]
    assert chosen.spectral_radius < 0.95
    # a higher gain must transmit at least as much signal as a lower one
    stds = [r.descending_std for _, r in sweep]
    assert stds[-1] > stds[0]
