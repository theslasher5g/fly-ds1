"""Step 4c: the bridge to a real game.

    keys/mouse  --(input backend)-->  game  --(screen capture)-->  frames
                                                    |
                                            health bars, optic flow -> reward

Everything game-specific is configuration: the capture rectangle, the key
bindings (:mod:`flyds1.motor.actions`), and the HUD regions the reward is read
from (:mod:`flyds1.envs.reward`).  Retargeting to another title means editing a
config, not this file.

Practical notes, learned the boring way:

* A real game cannot be reset from code.  After death the game reloads at the
  last bonfire on its own; :meth:`ScreenGameEnv.reset` just releases every key,
  waits for the reload, and starts a new episode.  Episodes are therefore
  wall-clock bounded, not state bounded.
* Real time is the hard limit: ~30 steps/s, no faster.  A PPO run that needs a
  million steps needs a day of real play.  Do the algorithm work in the
  synthetic arena (:mod:`flyds1.envs.arena`), which runs ~2000 steps/s, and come
  here for the final policy.
* ``dry_run=True`` (the default) computes everything and sends no input.  Turn
  it off deliberately, with the game in a single-player offline session, and
  keep the ``pydirectinput`` failsafe (slam the mouse into a screen corner)
  available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from flyds1.envs.input_backends import InputBackend, make_input_backend
from flyds1.envs.reward import RewardConfig, ScreenRewardEstimator, frame_flow_magnitude
from flyds1.envs.screen import CaptureRegion, FrameSource, make_frame_source
from flyds1.motor.actions import DARKSOULS_ACTIONS, ActionSpec
from flyds1.vision.ommatidia import luminance, resize_nearest

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _GYM_AVAILABLE = False

    class _Stub:
        class Env:
            pass

    gym = _Stub()  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


@dataclass
class GameConfig:
    """How to see the game, how to press its keys, how fast to run."""

    capture: CaptureRegion = field(default_factory=CaptureRegion)
    frame_source: str = "dummy"        # "mss" live, "replay" from a recording
    replay_path: str | None = None     # folder of frames or .npy stack
    #: "auto" picks pydirectinput on Windows, xdotool on Linux. dry_run is the
    #: single switch for whether input is sent at all; this only chooses which
    #: real backend to use once dry_run=False (an explicit "dry" here still
    #: forces DryRunBackend even if dry_run were somehow False, but there is
    #: no reason to set that -- use dry_run for that).
    input_backend: str = "auto"
    window_name: str | None = None     # for xdotool targeting
    dry_run: bool = True               # True: never send input, regardless of input_backend
    target_fps: float = 30.0
    frame_width: int = 320             # resolution handed to the retina
    frame_height: int = 180
    max_steps: int = 3600              # 2 minutes at 30 fps
    terminate_on_death: bool = True
    respawn_wait_s: float = 8.0        # time the game needs to reload
    warmup_frames: int = 3
    reward: RewardConfig = field(default_factory=RewardConfig)


class ScreenGameEnv(gym.Env):
    """Gymnasium env over a live game: frames in, keys out."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: GameConfig | None = None,
        action_spec: ActionSpec | None = None,
        *,
        frame_source: FrameSource | None = None,
        input_backend: InputBackend | None = None,
    ) -> None:
        if not _GYM_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs gymnasium: pip install -e '.[rl]'")
        super().__init__()
        self.cfg = config or GameConfig()
        self.spec_actions = action_spec or DARKSOULS_ACTIONS
        self.bindings = {b.name: b.key for b in self.spec_actions.buttons}
        self.source = frame_source or make_frame_source(
            self.cfg.frame_source, self.cfg.capture, path=self.cfg.replay_path
        )
        if input_backend is not None:
            self.input = input_backend
        else:
            kind = "dry" if self.cfg.dry_run else self.cfg.input_backend
            self.input = make_input_backend(kind, window=self.cfg.window_name)
        self.reward_fn = ScreenRewardEstimator(self.cfg.reward)

        self.action_space = self.spec_actions.gym_space()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.cfg.frame_height, self.cfg.frame_width), dtype=np.float32
        )
        self._period = 1.0 / float(self.cfg.target_fps)
        self._prev_luma: np.ndarray | None = None
        self._last_rgb: np.ndarray | None = None
        self.steps = 0

    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        """Nominal seconds per step -- what the retina integrates over."""
        return self._period

    def _grab(self) -> tuple[np.ndarray, np.ndarray]:
        rgb = self.source.grab()
        self._last_rgb = rgb
        luma = luminance(rgb)
        if luma.shape != (self.cfg.frame_height, self.cfg.frame_width):
            luma = resize_nearest(luma, self.cfg.frame_height, self.cfg.frame_width)
        return rgb, np.asarray(luma, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.input.release_all()
        self.reward_fn.reset()
        self.steps = 0
        if options and options.get("skip_wait"):
            pass
        elif self.cfg.respawn_wait_s > 0 and self.cfg.frame_source not in ("dummy", "synthetic", "test"):
            time.sleep(self.cfg.respawn_wait_s)  # pragma: no cover - real game only
        luma = None
        for _ in range(max(1, self.cfg.warmup_frames)):
            _, luma = self._grab()
        self._prev_luma = luma
        self._next_deadline = time.perf_counter() + self._period
        return luma, {"player_hp": 1.0, "boss_hp": 1.0, "steps": 0}

    def step(self, action):
        held, (dx, dy) = self.spec_actions.decode(action)
        self.input.apply(held, self.bindings)
        self.input.move_mouse(dx, dy)

        self._pace()
        rgb, luma = self._grab()
        flow = 0.0 if self._prev_luma is None else frame_flow_magnitude(self._prev_luma, luma)
        self._prev_luma = luma
        reward, info = self.reward_fn(rgb, flow)

        self.steps += 1
        terminated = bool(self.cfg.terminate_on_death and info["died"])
        truncated = self.steps >= self.cfg.max_steps
        info.update(steps=self.steps, held_keys=self.input.held_keys, dry_run=self.cfg.dry_run)
        if terminated or truncated:
            self.input.release_all()
        return luma, float(reward), terminated, truncated, info

    def _pace(self) -> None:
        """Hold the loop at ``target_fps``; report when we cannot keep up."""
        now = time.perf_counter()
        remaining = self._next_deadline - now
        if remaining > 0:
            time.sleep(remaining)
            self._next_deadline += self._period
        else:
            # behind schedule: resync instead of accumulating debt
            self._next_deadline = now + self._period

    def render(self):
        if self._last_rgb is None:
            return None
        return self._last_rgb

    def close(self) -> None:
        self.input.release_all()
        self.source.close()


__all__ = ["GameConfig", "ScreenGameEnv"]
