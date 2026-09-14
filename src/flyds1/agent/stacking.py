"""Short observation history, so the brain can be unrolled deterministically.

Why this exists: PPO replays its rollout buffer in *shuffled* minibatches.  A
network that carries hidden state across environment steps cannot be replayed
that way -- by the time a stored observation comes up for a gradient step, the
live state belongs to a different moment, and the gradient is wrong.

Two honest ways out, both provided:

* **This one.** Stack the last ``k`` observations; the policy integrates the
  connectome for ``k`` frames *from rest* on every forward pass.  The brain is
  then a pure function of the stack, shuffled replay is correct, and encoder and
  decoder both get gradients.  Cost: ``k`` times the compute, and no memory
  beyond ``k`` frames.
* :class:`flyds1.agent.brain_wrapper.FlyBrainWrapper` -- run the brain inside
  the environment with unbounded state, and train only the decoder.

``k`` is therefore the working-memory horizon of the model: at 30 fps, ``k=6``
is 200 ms of history, about the duration of a fly's optomotor response.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _GYM_AVAILABLE = False

    class _Stub:
        class ObservationWrapper:
            pass

    gym = _Stub()  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


class ObsStackWrapper(gym.ObservationWrapper):
    """Stacks the last ``k`` vector observations into ``(k, obs_size)``."""

    def __init__(self, env, k: int = 6) -> None:
        if not _GYM_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs gymnasium: pip install -e '.[rl]'")
        super().__init__(env)
        if k < 1:
            raise ValueError("k must be >= 1")
        base = env.observation_space
        if len(base.shape) != 1:
            raise ValueError(f"expected a vector observation, got shape {base.shape}")
        self.k = int(k)
        self.obs_size = int(base.shape[0])
        self._frames: deque[np.ndarray] = deque(maxlen=self.k)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.k, self.obs_size), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self._frames.append(np.asarray(observation, dtype=np.float32))
        while len(self._frames) < self.k:  # first steps of an episode
            self._frames.appendleft(self._frames[0])
        return np.stack(self._frames, axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        return self.observation(obs), info


__all__ = ["ObsStackWrapper"]
