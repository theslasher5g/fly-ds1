"""Steps 5 + 6: encoder, decoder, and the two ways to attach the brain."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from flyds1.agent.stacking import ObsStackWrapper  # noqa: E402
from flyds1.envs.arena import ArenaConfig, FlyArenaEnv  # noqa: E402
from flyds1.envs.wrappers import RetinaWrapper  # noqa: E402
from flyds1.motor.decoder import DescendingDecoder, NumpyDecoder, RunningStandardiser  # noqa: E402
from flyds1.net.dynamics import RateConfig  # noqa: E402
from flyds1.net.rate_rnn import TrainableParts  # noqa: E402
from flyds1.vision.encoder import DriveEncoder, build_encoder_wiring, encode_numpy  # noqa: E402
from flyds1.vision.frontend import FrontEndConfig, RetinaFrontEnd  # noqa: E402

SMALL_ARENA = ArenaConfig(width=96, height=54, max_steps=60, seed=2)


@pytest.fixture
def retina(network):
    return RetinaFrontEnd(network, FrontEndConfig(motion_mode="pooled"))


def test_encoder_wiring_targets_direction_cells(network, retina):
    wiring = build_encoder_wiring(network, retina.layout)
    assert wiring.n_photoreceptors == network.n_inputs
    assert wiring.n_motion_targets > 0
    assert set(wiring.group_names) >= {"t4a", "t4b", "t5a", "t5b"}
    # every target is a real neuron index, every source a real observation slot
    assert wiring.motion_target.max() < network.n_neurons
    assert wiring.motion_source.max() < np.prod(retina.layout.motion_shape)


def test_motion_injection_can_be_disabled(network, retina):
    wiring = build_encoder_wiring(network, retina.layout, inject_motion=False)
    assert wiring.n_motion_targets == 0


def test_encoder_backends_agree(network, retina, rng):
    wiring = build_encoder_wiring(network, retina.layout)
    obs = rng.normal(size=(5, retina.obs_size)).astype(np.float32)
    numpy_drive = encode_numpy(obs, retina.layout, wiring)
    encoder = DriveEncoder(wiring, retina.layout)
    torch_drive = encoder(torch.tensor(obs)).detach().numpy()
    assert np.abs(numpy_drive - torch_drive).max() < 1e-5


def test_encoder_drives_only_its_targets(network, retina, rng):
    wiring = build_encoder_wiring(network, retina.layout)
    drive = encode_numpy(rng.normal(size=retina.obs_size), retina.layout, wiring)
    touched = np.flatnonzero(np.abs(drive[0]) > 0)
    allowed = set(wiring.photoreceptor_index.tolist()) | set(wiring.motion_target.tolist())
    assert set(touched.tolist()) <= allowed


def test_standardiser_recovers_temporal_statistics():
    """Regression: batch-only variance made the features vanish."""
    norm = RunningStandardiser(3)
    rng = np.random.default_rng(0)
    for t in range(500):
        base = np.sin(t / 15.0) * np.array([1.0, 2.0, 0.5]) + 5.0
        norm(torch.tensor(base + rng.normal(scale=1e-4, size=(2, 3)), dtype=torch.float32))
    norm.eval()
    out = []
    for t in range(200):
        base = np.sin(t / 15.0) * np.array([1.0, 2.0, 0.5]) + 5.0
        out.append(norm(torch.tensor(base[None], dtype=torch.float32)).numpy())
    z = np.concatenate(out)
    assert np.allclose(z.mean(axis=0), 0, atol=0.3)
    assert np.allclose(z.std(axis=0), 1.0, rtol=0.4)


def test_standardiser_survives_a_batch_of_one():
    norm = RunningStandardiser(2)
    out = norm(torch.tensor([[5.0, -3.0]]))
    assert torch.isfinite(out).all()
    for _ in range(10):
        out = norm(torch.tensor([[5.0, -3.0]]))
    assert torch.isfinite(out).all()


def test_decoder_is_linear_and_trainable():
    decoder = DescendingDecoder(8, 4)
    rates = torch.rand(16, 8) + 1.0
    out = decoder(rates)
    assert out.shape == (16, 4)
    out.pow(2).mean().backward()
    assert decoder.linear.weight.grad.abs().sum() > 0
    assert sum(p.numel() for p in decoder.parameters() if p.requires_grad) < 100


def test_numpy_decoder_matches_torch():
    decoder = DescendingDecoder(6, 3)
    rates = torch.rand(10, 6) + 1.0
    decoder.eval()
    expected = decoder(rates).detach().numpy()
    got = NumpyDecoder.from_torch(decoder).decode(rates.numpy())
    assert np.abs(expected - got).max() < 1e-5


def test_fly_extractor_unrolls_the_stack(network):
    from flyds1.agent.fly_policy import FlyBrainExtractor

    base = RetinaWrapper(FlyArenaEnv(SMALL_ARENA), network, FrontEndConfig(motion_mode="pooled"))
    env = ObsStackWrapper(base, k=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, base.front_end.obs_size)
    # Move first: on the very first frame the Weber contrast is exactly zero by
    # construction (the adaptation state starts at that frame), so the gradient
    # of a multiplicative gain would legitimately be zero too.
    for _ in range(5):
        obs, *_ = env.step(np.array([1, 0, 0, 1], dtype=np.float32))
    assert np.abs(obs).max() > 0

    extractor = FlyBrainExtractor(
        env.observation_space, network, base.layout, trainable=TrainableParts(plastic_edges=10)
    )
    features = extractor(torch.tensor(obs[None], dtype=torch.float32))
    assert features.shape == (1, network.n_outputs)
    features.pow(2).mean().backward()
    assert extractor.encoder.photo_gain.grad.abs().sum() > 0
    assert extractor.rnn.plastic_val.grad.abs().sum() > 0
    assert not extractor.rnn.w_val.requires_grad


def test_fly_extractor_is_a_pure_function_of_the_stack(network):
    """Required for PPO: shuffled minibatch replay must be reproducible."""
    from flyds1.agent.fly_policy import FlyBrainExtractor

    base = RetinaWrapper(FlyArenaEnv(SMALL_ARENA), network, FrontEndConfig(motion_mode="pooled"))
    env = ObsStackWrapper(base, k=3)
    obs, _ = env.reset(seed=0)
    x = torch.tensor(obs[None], dtype=torch.float32)
    extractor = FlyBrainExtractor(env.observation_space, network, base.layout, standardise=False)
    extractor.eval()
    with torch.no_grad():
        first, second = extractor(x), extractor(x)
    assert torch.allclose(first, second)


def test_fly_extractor_rejects_unstacked_observations(network):
    from flyds1.agent.fly_policy import FlyBrainExtractor

    env = RetinaWrapper(FlyArenaEnv(SMALL_ARENA), network, FrontEndConfig(motion_mode="pooled"))
    with pytest.raises(ValueError, match="ObsStackWrapper"):
        FlyBrainExtractor(env.observation_space, network, env.layout)


def test_brain_wrapper_outputs_descending_rates(network):
    from flyds1.agent.brain_wrapper import FlyBrainWrapper

    base = RetinaWrapper(FlyArenaEnv(SMALL_ARENA), network, FrontEndConfig(motion_mode="pooled"))
    env = FlyBrainWrapper(base, network, base.layout, rate_config=RateConfig())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (network.n_outputs,)
    observations = [obs]
    for _ in range(30):
        obs, *_ = env.step(np.array([1, 0, 0, 1], dtype=np.float32))
        observations.append(obs)
    stack = np.stack(observations)
    assert np.isfinite(stack).all()
    assert stack.std() > 1e-6, "descending rates do not move with the world"


def test_stack_wrapper_pads_the_first_step(network):
    base = RetinaWrapper(FlyArenaEnv(SMALL_ARENA), network, FrontEndConfig(motion_mode="pooled"))
    env = ObsStackWrapper(base, k=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 4
    assert np.allclose(obs[0], obs[-1])  # padded with the first frame
