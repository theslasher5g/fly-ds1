"""Step 6: the fly brain as a Stable-Baselines3 feature extractor.

Layout of the policy:

    obs stack (k, obs_size)
        -> DriveEncoder            (trainable, tiny: gain/offset per facet)
        -> ConnectomeRNN x k steps  (FROZEN connectome weights)
        -> descending-neuron rates
        -> standardiser            (running mean/var)
        -> SB3 action head          (the linear decoder; net_arch=[])

With ``net_arch=[]`` the policy's own head *is* the linear decoder of roadmap
step 5, so nothing between the DNs and the keys is hidden in an MLP.  The value
function does get an MLP: value estimation is an artefact of the RL algorithm,
not part of the animal, and starving it only makes the gradient noisier.
"""

from __future__ import annotations

from flyds1.connectome.graph import WiredNetwork
from flyds1.net.dynamics import RateConfig
from flyds1.net.rate_rnn import ConnectomeRNN, TrainableParts, require_torch
from flyds1.vision.encoder import DriveEncoder, build_encoder_wiring
from flyds1.vision.frontend import ObsLayout

try:
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _SB3_AVAILABLE = False

    class BaseFeaturesExtractor:  # type: ignore[no-redef]
        pass

    torch = None  # type: ignore[assignment]

    class _NNStub:
        class Module:
            pass

    nn = _NNStub()  # type: ignore[assignment]


class FlyBrainExtractor(BaseFeaturesExtractor):
    """Unrolls the connectome over an observation stack; outputs DN rates."""

    def __init__(
        self,
        observation_space,
        network: WiredNetwork,
        layout: ObsLayout,
        *,
        rate_config: RateConfig | None = None,
        trainable: TrainableParts | None = None,
        inject_motion: bool = True,
        standardise: bool = True,
        warmup_steps: int = 2,
    ) -> None:
        if not _SB3_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs stable-baselines3: pip install -e '.[rl]'")
        require_torch()
        if len(observation_space.shape) != 2:
            raise ValueError(
                f"FlyBrainExtractor expects a stacked observation (k, obs_size), "
                f"got {observation_space.shape}; wrap the env in ObsStackWrapper"
            )
        n_outputs = int(network.n_outputs)
        if n_outputs == 0:
            raise ValueError(
                "this network has no descending neurons to read out; check the "
                "connectome's annotations (super_class 'descending')"
            )
        super().__init__(observation_space, features_dim=n_outputs)

        self.k = int(observation_space.shape[0])
        self.warmup_steps = int(warmup_steps)
        self.rate_config = rate_config or RateConfig()
        wiring = build_encoder_wiring(network, layout, inject_motion=inject_motion)
        self.encoder = DriveEncoder(wiring, layout)
        self.rnn = ConnectomeRNN(network, self.rate_config, trainable or TrainableParts())
        self.wiring = wiring

        if standardise:
            from flyds1.motor.decoder import RunningStandardiser

            self.norm: nn.Module | None = RunningStandardiser(n_outputs)
        else:
            self.norm = None

    # ------------------------------------------------------------------
    def forward(self, observations: "torch.Tensor") -> "torch.Tensor":
        """``(batch, k, obs_size)`` -> ``(batch, n_descending)``."""
        if observations.dim() == 2:  # single stack
            observations = observations.unsqueeze(0)
        batch = observations.shape[0]
        v = self.rnn.init_state(batch, device=observations.device)

        # settle the network at its resting state before the first frame, so the
        # first observation does not land on a cold, all-zero network
        for _ in range(self.warmup_steps):
            v, r = self.rnn(v, None)
        for t in range(self.k):
            drive = self.encoder(observations[:, t])
            v, r = self.rnn(v, drive)
        rates = self.rnn.read_outputs(r)
        return self.norm(rates) if self.norm is not None else rates

    def describe(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = int(self.rnn.w_val.numel())
        return (
            f"FlyBrainExtractor: stack k={self.k}, "
            f"{self.rnn.n_neurons} neurons, {self.features_dim} descending outputs\n"
            f"  frozen connectome weights: {frozen}\n"
            f"  trainable parameters (encoder + optional net parts): {trainable}\n"
            f"  {self.wiring.describe()}"
        )


def fly_policy_kwargs(
    network: WiredNetwork,
    layout: ObsLayout,
    *,
    rate_config: RateConfig | None = None,
    trainable: TrainableParts | None = None,
    inject_motion: bool = True,
    value_net_arch: list[int] | None = None,
) -> dict:
    """``policy_kwargs`` for ``PPO("MlpPolicy", env, **fly_policy_kwargs(...))``.

    ``net_arch`` gives the policy head no hidden layers (the linear decoder) and
    the value head a small MLP.
    """
    return {
        "features_extractor_class": FlyBrainExtractor,
        "features_extractor_kwargs": {
            "network": network,
            "layout": layout,
            "rate_config": rate_config,
            "trainable": trainable,
            "inject_motion": inject_motion,
        },
        "net_arch": {"pi": [], "vf": value_net_arch or [64, 64]},
        "share_features_extractor": True,
    }


__all__ = ["FlyBrainExtractor", "fly_policy_kwargs"]
