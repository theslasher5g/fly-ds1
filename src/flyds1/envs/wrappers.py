"""Glue: frame-producing environments -> retina observations.

Keeping the retina in a wrapper (rather than inside each env) means the arena
and the real game share exactly one implementation of step 3, and you can look
at the raw frames any time by dropping the wrapper.
"""

from __future__ import annotations

import numpy as np

from flyds1.connectome.graph import WiredNetwork
from flyds1.vision.frontend import FrontEndConfig, RetinaFrontEnd

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _GYM_AVAILABLE = False

    class _Stub:
        class Wrapper:
            pass

        class ObservationWrapper:
            pass

    gym = _Stub()  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


class RetinaWrapper(gym.Wrapper):
    """Replaces frame observations with per-photoreceptor contrast + motion.

    The wrapped env must produce ``(H, W)`` or ``(H, W, 3)`` frames.  ``dt`` is
    taken from the env (``env.dt`` or ``env.cfg.dt``) so the motion detectors
    integrate over the true frame interval; pass it explicitly for envs that
    expose neither.
    """

    def __init__(
        self,
        env,
        network: WiredNetwork,
        config: FrontEndConfig | None = None,
        *,
        dt: float | None = None,
        keep_frame_in_info: bool = False,
    ) -> None:
        if not _GYM_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs gymnasium: pip install -e '.[rl]'")
        super().__init__(env)
        cfg = config or FrontEndConfig()
        frame_shape = env.observation_space.shape
        if len(frame_shape) < 2:
            raise ValueError(f"expected a frame observation, got shape {frame_shape}")
        # The retina's screen geometry must match the frames it will be fed.
        cfg.screen = type(cfg.screen)(
            width=int(frame_shape[1]), height=int(frame_shape[0]), fov_h_deg=cfg.screen.fov_h_deg
        )
        self.front_end = RetinaFrontEnd(network, cfg)
        self.dt = float(dt if dt is not None else self._infer_dt(env))
        self.keep_frame_in_info = bool(keep_frame_in_info)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.front_end.obs_size,), dtype=np.float32
        )

    @staticmethod
    def _infer_dt(env) -> float:
        for attr in ("dt",):
            value = getattr(env, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
        cfg = getattr(env, "cfg", None)
        value = getattr(cfg, "dt", None)
        if isinstance(value, (int, float)):
            return float(value)
        fps = getattr(cfg, "target_fps", None)
        if isinstance(fps, (int, float)) and fps > 0:
            return 1.0 / float(fps)
        raise ValueError("cannot infer dt from env; pass dt=... explicitly")

    @property
    def layout(self):
        return self.front_end.layout

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        self.front_end.reset()
        obs = self.front_end.process(frame, self.dt)
        if self.keep_frame_in_info:
            info = {**info, "frame": frame}
        return obs, info

    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)
        obs = self.front_end.process(frame, self.dt)
        if self.keep_frame_in_info:
            info = {**info, "frame": frame}
        return obs, reward, terminated, truncated, info


__all__ = ["RetinaWrapper"]
