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
    import numpy as np
    import torch
    import torch.nn as nn
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp

    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _SB3_AVAILABLE = False

    class BaseFeaturesExtractor:  # type: ignore[no-redef]
        pass

    class ActorCriticPolicy:  # type: ignore[no-redef]
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


class RawObservationExtractor(BaseFeaturesExtractor):
    """The critic's view: the observation itself, flattened.

    The value function is not part of the animal -- it is bookkeeping the RL
    algorithm needs -- so there is no reason to squeeze it through the same
    descending-neuron bottleneck as the policy.  Doing so makes it estimate
    returns from 8-24 numbers that have already thrown most of the scene away,
    and a critic that cannot predict the return turns every advantage into
    noise.
    """

    def __init__(self, observation_space) -> None:
        if not _SB3_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs stable-baselines3: pip install -e '.[rl]'")
        super().__init__(observation_space, features_dim=int(np.prod(observation_space.shape)))
        self.flatten = nn.Flatten()

    def forward(self, observations: "torch.Tensor") -> "torch.Tensor":
        return self.flatten(observations)


class SplitMlpExtractor(nn.Module):
    """Like SB3's ``MlpExtractor``, but actor and critic may differ in width.

    SB3's own version takes a single ``feature_dim`` for both heads, which rules
    out an actor reading 24 descending neurons and a critic reading the whole
    observation.
    """

    def __init__(self, pi_dim: int, vf_dim: int, net_arch: dict, activation_fn) -> None:
        super().__init__()
        arch_pi = list(net_arch.get("pi", []))
        arch_vf = list(net_arch.get("vf", []))
        self.policy_net = nn.Sequential(*create_mlp(pi_dim, -1, arch_pi, activation_fn))
        self.value_net = nn.Sequential(*create_mlp(vf_dim, -1, arch_vf, activation_fn))
        self.latent_dim_pi = arch_pi[-1] if arch_pi else pi_dim
        self.latent_dim_vf = arch_vf[-1] if arch_vf else vf_dim

    def forward(self, features):  # pragma: no cover - only used when sharing
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: "torch.Tensor") -> "torch.Tensor":
        return self.policy_net(features)

    def forward_critic(self, features: "torch.Tensor") -> "torch.Tensor":
        return self.value_net(features)


class FlyActorCriticPolicy(ActorCriticPolicy):
    """Actor through the connectome, critic straight off the observation.

    Use via ``PPO(FlyActorCriticPolicy, env, policy_kwargs=fly_policy_kwargs(...,
    critic_sees="observation"))``.
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        if not _SB3_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs stable-baselines3: pip install -e '.[rl]'")
        kwargs["share_features_extractor"] = False
        kwargs.setdefault("features_extractor_class", FlyBrainExtractor)
        # SB3 builds the actor's extractor first and the critic's second (see
        # ActorCriticPolicy.__init__); the counter is how we hand out a
        # different class for the second call without constructing a whole
        # second connectome and throwing it away.
        self._extractor_calls = 0
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def make_features_extractor(self):
        self._extractor_calls += 1
        if self._extractor_calls == 1:
            return super().make_features_extractor()
        return RawObservationExtractor(self.observation_space)

    def _build_mlp_extractor(self) -> None:
        net_arch = self.net_arch if isinstance(self.net_arch, dict) else {"pi": self.net_arch, "vf": self.net_arch}
        self.mlp_extractor = SplitMlpExtractor(
            pi_dim=self.pi_features_extractor.features_dim,
            vf_dim=self.vf_features_extractor.features_dim,
            net_arch=net_arch,
            activation_fn=self.activation_fn,
        ).to(self.device)

    def describe(self) -> str:
        return (
            f"{self.pi_features_extractor.describe()}\n"
            f"  critic reads the raw observation "
            f"({self.vf_features_extractor.features_dim} numbers), not the DN bottleneck"
        )


def fly_policy_kwargs(
    network: WiredNetwork,
    layout: ObsLayout,
    *,
    rate_config: RateConfig | None = None,
    trainable: TrainableParts | None = None,
    inject_motion: bool = True,
    value_net_arch: list[int] | None = None,
    critic_sees: str = "observation",
) -> dict:
    """``policy_kwargs`` for PPO.

    ``net_arch`` gives the policy head no hidden layers -- that head *is* the
    linear decoder of step 5 -- and the value head a small MLP.

    ``critic_sees``:

    * ``"observation"`` (default) -- pair with :class:`FlyActorCriticPolicy`:
      the critic reads the raw observation while the actor goes through the
      connectome.
    * ``"features"`` -- pair with ``"MlpPolicy"``: both read the descending
      neurons. Simpler, and a much harder job for the critic.
    """
    if critic_sees not in ("observation", "features"):
        raise ValueError(f"critic_sees must be 'observation' or 'features', got {critic_sees!r}")
    kwargs = {
        "features_extractor_class": FlyBrainExtractor,
        "features_extractor_kwargs": {
            "network": network,
            "layout": layout,
            "rate_config": rate_config,
            "trainable": trainable,
            "inject_motion": inject_motion,
        },
        "net_arch": {"pi": [], "vf": value_net_arch or [64, 64]},
    }
    if critic_sees == "features":
        kwargs["share_features_extractor"] = True
    return kwargs


__all__ = [
    "FlyActorCriticPolicy",
    "FlyBrainExtractor",
    "RawObservationExtractor",
    "SplitMlpExtractor",
    "fly_policy_kwargs",
]
