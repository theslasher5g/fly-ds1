"""Step 4d: one boss, over and over.

A boss fight is the smallest unit of a souls game that is worth learning: it has
a start, two possible endings, a dense-ish reward (the boss's health bar), and
it repeats.  Everything outside the fight -- dying, reloading, walking back to
the fog gate -- is overhead this module automates and does not score.

    WAITING  --boss bar appears-->  FIGHTING  --bar gone / death screen-->  ENDED
       ^                                                                     |
       +-------------------- run-back macro ---------------------------------+

What this is honest about:

* **Real time is the budget.** At 30 fps with ``action_repeat=4``, one decision
  is 133 ms, an attempt lasting a minute is ~450 decisions, and the run-back
  costs another 30-60 s of wall clock. That is roughly 30-40 attempts an hour,
  i.e. ~15k decisions an hour. Published single-boss RL runs use millions.
  :meth:`BossFightEnv.budget` prints this arithmetic for your settings, because
  it decides whether an experiment is feasible before it decides anything else.
* **The run-back is a replayed route, not navigation.** The way back from the
  bonfire never changes, so it needs reliable execution rather than
  pathfinding: record it once with ``flyds1 route --record`` and every segment
  gets a visual checkpoint that is verified on the way (see
  :mod:`flyds1.envs.navigation`). A run-back that quietly went the wrong way
  turns every following attempt into noise, so a lost route abandons the
  attempt instead.
* **The detectors are pixel heuristics.** They are tested against synthetic
  footage here; against your actual capture they need calibrating.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from flyds1.envs.detectors import DetectorConfig, FightStateDetector
from flyds1.envs.input_backends import InputBackend, make_input_backend
from flyds1.envs.navigation import Route
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
class Macro:
    """A scripted key sequence: ``[(buttons, seconds), ...]``.

    Used for the parts of the loop the agent is not learning -- getting back to
    the fog gate, walking through it. Buttons are names from the
    :class:`~flyds1.motor.actions.ActionSpec`.
    """

    steps: tuple[tuple[tuple[str, ...], float], ...] = ()
    name: str = "macro"

    @property
    def duration_s(self) -> float:
        return float(sum(seconds for _, seconds in self.steps))

    def play(self, backend: InputBackend, bindings: dict[str, str], sleep_fn=time.sleep) -> None:
        for buttons, seconds in self.steps:
            backend.apply({name: True for name in buttons}, bindings)
            sleep_fn(max(0.0, float(seconds)))
        backend.release_all()


#: Placeholder run-back: walk forward for a few seconds, then through the fog.
#: Every route is different -- record yours.
DEFAULT_RUNBACK = Macro(
    steps=((("forward",), 4.0), ((), 0.5), (("forward",), 3.0)),
    name="placeholder run-back",
)
DEFAULT_ENTER_FOG = Macro(steps=((("forward",), 2.0),), name="through the fog gate")


@dataclass
class BossRewardConfig:
    """What a boss attempt is scored on."""

    boss_damage: float = 20.0      # per unit of boss health removed
    player_damage: float = 5.0     # per unit of own health lost
    kill: float = 50.0
    death: float = 10.0
    survival: float = 0.005        # per decision; small, this is not the goal


@dataclass
class BossConfig:
    """Capture, input, timing and reward for a repeated boss fight."""

    capture: CaptureRegion = field(default_factory=CaptureRegion)
    frame_source: str = "dummy"
    replay_path: str | None = None
    #: "auto" picks pydirectinput on Windows, xdotool on Linux. dry_run is the
    #: single switch for whether input is sent at all -- see GameConfig for why
    #: this does not also default to "dry".
    input_backend: str = "auto"
    window_name: str | None = None
    dry_run: bool = True

    target_fps: float = 30.0
    #: Frames per decision.  Raise it: the descending neurons need 8-16 frames
    #: to hear about a stimulus at all (docs/results.md), so deciding every
    #: single frame asks the brain to answer before it has been told.
    action_repeat: int = 4
    frame_width: int = 320
    frame_height: int = 180

    max_fight_steps: int = 900          # decisions; 900 * 133 ms = 2 minutes
    max_wait_steps: int = 600           # give up waiting for the fight to start
    wait_after_death_s: float = 12.0    # the game's own reload
    runback: Macro = field(default_factory=lambda: DEFAULT_RUNBACK)
    enter_fog: Macro = field(default_factory=lambda: DEFAULT_ENTER_FOG)
    #: A recorded, checkpoint-verified route (``flyds1 route --record``). When
    #: set it replaces the blind ``runback``/``enter_fog`` macros: same idea,
    #: but it notices when it has gone wrong.
    route_path: str | None = None

    detectors: DetectorConfig = field(default_factory=DetectorConfig)
    reward: BossRewardConfig = field(default_factory=BossRewardConfig)


class BossFightEnv(gym.Env):
    """One attempt per episode, with the overhead automated away."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: BossConfig | None = None,
        action_spec: ActionSpec | None = None,
        *,
        frame_source: FrameSource | None = None,
        input_backend: InputBackend | None = None,
        sleep_fn=None,
    ) -> None:
        if not _GYM_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs gymnasium: pip install -e '.[rl]'")
        super().__init__()
        self.cfg = config or BossConfig()
        self.spec_actions = action_spec or DARKSOULS_ACTIONS
        self.bindings = {b.name: b.key for b in self.spec_actions.buttons}
        self.source = frame_source or make_frame_source(
            self.cfg.frame_source, self.cfg.capture, path=self.cfg.replay_path
        )
        if input_backend is not None:
            self.input = input_backend
        else:
            self.input = make_input_backend(
                "dry" if self.cfg.dry_run else self.cfg.input_backend,
                window=self.cfg.window_name,
            )
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        self.detector = FightStateDetector(self.cfg.detectors)
        self.route = Route.load(self.cfg.route_path) if self.cfg.route_path else None
        #: Result of the most recent run-back, or None if it was not run.
        self.last_route_result = None
        self.routes_lost = 0

        self.action_space = self.spec_actions.gym_space()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.cfg.frame_height, self.cfg.frame_width), dtype=np.float32
        )
        self._period = 1.0 / float(self.cfg.target_fps)
        self._last_rgb: np.ndarray | None = None
        self.attempts = 0
        self.kills = 0
        self.steps = 0
        self.state = "waiting"

    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        """Seconds one decision covers -- what the retina integrates over."""
        return self._period * self.cfg.action_repeat

    def budget(self, hours: float = 1.0) -> dict:
        """The arithmetic that decides whether a run is feasible at all."""
        attempt_s = self.cfg.max_fight_steps * self.dt
        overhead_s = (
            self.cfg.wait_after_death_s
            + self.cfg.runback.duration_s
            + self.cfg.enter_fog.duration_s
        )
        per_attempt = attempt_s + overhead_s
        attempts = hours * 3600.0 / per_attempt
        return {
            "seconds_per_decision": self.dt,
            "seconds_per_attempt_max": per_attempt,
            "attempts_per_hour": attempts,
            "decisions_per_hour": attempts * self.cfg.max_fight_steps,
            "hours_for_1M_decisions": 1e6 / max(1e-9, attempts * self.cfg.max_fight_steps),
        }

    # ------------------------------------------------------------------
    def _grab(self) -> tuple[np.ndarray, np.ndarray]:
        rgb = self.source.grab()
        self._last_rgb = rgb
        luma = luminance(rgb)
        if luma.shape != (self.cfg.frame_height, self.cfg.frame_width):
            luma = resize_nearest(luma, self.cfg.frame_height, self.cfg.frame_width)
        return rgb, np.asarray(luma, dtype=np.float32)

    def _pace(self) -> None:
        now = time.perf_counter()
        remaining = self._next_deadline - now
        if remaining > 0:
            self._sleep(remaining)
            self._next_deadline += self._period
        else:
            self._next_deadline = now + self._period

    def _advance_frames(self, n: int) -> tuple[list[np.ndarray], dict]:
        """Grab ``n`` frames at the target rate; returns luma frames and state."""
        frames: list[np.ndarray] = []
        state: dict = {}
        for _ in range(max(1, n)):
            self._pace()
            rgb, luma = self._grab()
            state = self.detector.update(rgb)
            frames.append(luma)
            if state["fight_ended"]:
                break
        return frames, state

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.input.release_all()
        self.steps = 0
        self._next_deadline = time.perf_counter() + self._period

        skip = bool(options and options.get("skip_macros"))
        if self.state in ("dead", "ended") and not skip:
            self._sleep(self.cfg.wait_after_death_s)
            self._run_back()

        self.detector.reset()
        self.state = "waiting"
        frames, state = [], {}
        for _ in range(self.cfg.max_wait_steps):
            frames, state = self._advance_frames(1)
            if state.get("fight_active"):
                self.state = "fighting"
                break
        self.attempts += 1
        info = {
            **state,
            "state": self.state,
            "attempts": self.attempts,
            "kills": self.kills,
            "routes_lost": self.routes_lost,
            "route": self.last_route_result.describe() if self.last_route_result else None,
        }
        return frames[-1], info

    def _run_back(self, steer=None) -> None:
        """Get from the bonfire to the fog gate.

        Prefers the recorded route, which verifies where it ends up; falls back
        to the blind macros when no route has been recorded yet.
        """
        if self.route is not None:
            self.last_route_result = self.route.run(
                self.input,
                self.bindings,
                self.source.grab,
                sleep_fn=self._sleep,
                fps=self.cfg.target_fps,
                steer=steer,
            )
            if not self.last_route_result.completed:
                self.routes_lost += 1
            return
        self.cfg.runback.play(self.input, self.bindings, self._sleep)
        self.cfg.enter_fog.play(self.input, self.bindings, self._sleep)

    def step(self, action):
        held, (dx, dy) = self.spec_actions.decode(action)
        self.input.apply(held, self.bindings)
        self.input.move_mouse(dx, dy)

        before_boss = self.detector.boss_hp
        before_player = self.detector.player_hp
        frames, state = self._advance_frames(self.cfg.action_repeat)
        self.steps += 1

        cfg = self.cfg.reward
        boss_damage = max(0.0, before_boss - state["boss_hp"])
        player_damage = max(0.0, before_player - state["player_hp"])
        reward = (
            cfg.survival
            + cfg.boss_damage * boss_damage
            - cfg.player_damage * player_damage
        )

        terminated = False
        ended = state["fight_ended"]
        if ended == "boss_killed":
            reward += cfg.kill
            self.kills += 1
            self.state = "ended"
            terminated = True
        elif ended == "player_died":
            reward -= cfg.death
            self.state = "dead"
            terminated = True
        elif ended == "aborted":
            self.state = "ended"
            terminated = True

        truncated = self.steps >= self.cfg.max_fight_steps
        if terminated or truncated:
            self.input.release_all()

        info = {
            **state,
            "state": self.state,
            "attempts": self.attempts,
            "kills": self.kills,
            "boss_damage": boss_damage,
            "player_damage": player_damage,
            "frames": frames,          # RetinaWrapper integrates every frame
            "held_keys": self.input.held_keys,
            "dry_run": self.cfg.dry_run,
            "steps": self.steps,
        }
        return frames[-1], float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._last_rgb

    def close(self) -> None:
        self.input.release_all()
        self.source.close()


__all__ = [
    "DEFAULT_ENTER_FOG",
    "DEFAULT_RUNBACK",
    "BossConfig",
    "BossFightEnv",
    "BossRewardConfig",
    "Macro",
]
