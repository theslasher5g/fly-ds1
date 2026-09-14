"""The other honest option: run the brain inside the environment.

Here the connectome network lives in a wrapper, keeps its state across the whole
episode (no ``k``-frame horizon), and the observation the RL algorithm sees *is*
the descending-neuron activity.  PPO then trains the linear decoder and the
value function only -- the encoder stays fixed, because no gradient can reach
through the environment boundary.

Use this when episode-long dynamics matter more than a learned encoder, or as a
sanity check: if a frozen-encoder fly can already play, the encoder was never
the interesting part.
"""

from __future__ import annotations

import numpy as np

from flyds1.connectome.graph import WiredNetwork
from flyds1.net.dynamics import RateConfig
from flyds1.net.reference import RateNetwork
from flyds1.vision.encoder import EncoderWiring, build_encoder_wiring, encode_numpy
from flyds1.vision.frontend import ObsLayout

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _GYM_AVAILABLE = False

    class _Stub:
        class Wrapper:
            pass

    gym = _Stub()  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


class FlyBrainWrapper(gym.Wrapper):
    """Observation: retina -> connectome -> descending rates."""

    def __init__(
        self,
        env,
        network: WiredNetwork,
        layout: ObsLayout,
        *,
        rate_config: RateConfig | None = None,
        photo_gain: float = 1.0,
        motion_gain: float = 1.0,
        inject_motion: bool = True,
        standardise: bool = True,
        standardise_momentum: float = 0.01,
    ) -> None:
        if not _GYM_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs gymnasium: pip install -e '.[rl]'")
        super().__init__(env)
        self.network = network
        self.layout = layout
        self.cfg = rate_config or RateConfig()
        self.sim = RateNetwork(network, self.cfg)
        self.wiring: EncoderWiring = build_encoder_wiring(network, layout, inject_motion=inject_motion)
        self.photo_gain = float(photo_gain)
        self.motion_gain = float(motion_gain)
        self.standardise = bool(standardise)
        self.momentum = float(standardise_momentum)
        self._mean = np.zeros(network.n_outputs)
        self._var = np.ones(network.n_outputs)
        self._seen = 0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(network.n_outputs,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _brain_step(self, retina_obs: np.ndarray) -> np.ndarray:
        drive = encode_numpy(
            retina_obs,
            self.layout,
            self.wiring,
            photo_gain=self.photo_gain,
            motion_gain=self.motion_gain,
        )
        self.sim.step(drive)
        rates = self.sim.outputs()[0]
        return self._normalise(rates).astype(np.float32)

    def _normalise(self, rates: np.ndarray) -> np.ndarray:
        """Exponentially weighted z-score (Welford form).

        ``delta`` is taken against the *old* mean and multiplied by the residual
        against the *new* one; that is the EWMA analogue of Welford's update and
        it tracks the variance over time, which is the variance that matters
        here -- see :class:`flyds1.motor.decoder.RunningStandardiser`.
        """
        if not self.standardise:
            return rates
        if self._seen == 0:
            self._mean = rates.copy()
            self._seen = 1
            return np.zeros_like(rates)
        m = self.momentum
        delta = rates - self._mean
        self._mean = self._mean + m * delta
        self._var = (1 - m) * self._var + m * delta * (rates - self._mean)
        self._seen += 1
        return delta / np.sqrt(self._var + 1e-5)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.sim.reset()
        return self._brain_step(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._brain_step(obs), reward, terminated, truncated, info


__all__ = ["FlyBrainWrapper"]
