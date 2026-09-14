"""The boss-fight loop: state machine, reward, macros, and the time budget."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from flyds1.envs.boss import BossConfig, BossFightEnv, BossRewardConfig, Macro  # noqa: E402
from flyds1.envs.detectors import DetectorConfig, FightStateDetector  # noqa: E402
from flyds1.envs.input_backends import DryRunBackend  # noqa: E402
from flyds1.envs.screen import FrameSource  # noqa: E402
from flyds1.motor.actions import DARKSOULS_ACTIONS  # noqa: E402

H, W = 180, 320


def hud_frame(player=1.0, boss=None, death=False):
    frame = np.full((H, W, 3), 0.2)
    x0 = int(0.035 * W)
    frame[int(0.045 * H) : int(np.ceil(0.062 * H)), x0 : x0 + int(0.21 * W * player)] = (
        0.7, 0.05, 0.05,
    )
    if boss is not None:
        b0 = int(0.255 * W)
        frame[int(0.885 * H) : int(np.ceil(0.9 * H)), b0 : b0 + int(0.49 * W * boss)] = (
            0.7, 0.05, 0.05,
        )
    if death:
        frame[:] = (0.18, 0.02, 0.02)
    return (frame * 255).astype(np.uint8)


class FakeFight(FrameSource):
    """A boss fight that reacts to the keys the agent is actually holding.

    Closes the loop: attacking hurts the boss, standing in front of it hurts
    you. Enough to test the env's state machine end to end without a game.
    """

    def __init__(self, backend: DryRunBackend, *, frames_before_fight=4, boss_dps=0.05,
                 player_dps=0.0):
        self.backend = backend
        self.frames_before_fight = frames_before_fight
        self.boss_dps = boss_dps
        self.player_dps = player_dps
        self.t = 0
        self.boss_hp = 1.0
        self.player_hp = 1.0
        self.finished = False

    def grab(self):
        self.t += 1
        if self.t <= self.frames_before_fight:
            return hud_frame(self.player_hp, None)
        if self.finished:
            return hud_frame(self.player_hp, None, death=self.player_hp <= 0.0)
        if "mouse:left" in self.backend.held_keys:
            self.boss_hp = max(0.0, self.boss_hp - self.boss_dps)
        self.player_hp = max(0.0, self.player_hp - self.player_dps)
        if self.boss_hp <= 0.0 or self.player_hp <= 0.0:
            self.finished = True
        return hud_frame(self.player_hp, self.boss_hp)


def make_env(**kwargs):
    backend = DryRunBackend()
    cfg = BossConfig(
        target_fps=10_000.0,
        action_repeat=2,
        max_fight_steps=200,
        max_wait_steps=50,
        wait_after_death_s=0.0,
        frame_width=64,
        frame_height=36,
        **kwargs,
    )
    source = FakeFight(backend)
    env = BossFightEnv(
        cfg, DARKSOULS_ACTIONS, frame_source=source, input_backend=backend, sleep_fn=lambda _s: None
    )
    return env, source, backend


def attack_action():
    action = np.zeros(DARKSOULS_ACTIONS.size, dtype=np.int8)
    action[DARKSOULS_ACTIONS.button_index("attack")] = 1
    return action


def idle_action():
    return np.zeros(DARKSOULS_ACTIONS.size, dtype=np.int8)


def test_episode_waits_for_the_fight_then_runs_it():
    env, source, _ = make_env()
    obs, info = env.reset(options={"skip_macros": True})
    assert obs.shape == (36, 64)
    assert info["state"] == "fighting"
    assert info["attempts"] == 1

    obs, reward, terminated, truncated, info = env.step(attack_action())
    assert info["boss_damage"] > 0
    assert reward > 0
    assert not terminated


def test_killing_the_boss_terminates_and_pays_the_bonus():
    env, source, _ = make_env()
    env.reset(options={"skip_macros": True})
    total = 0.0
    for _ in range(300):
        _, reward, terminated, truncated, info = env.step(attack_action())
        total += reward
        if terminated or truncated:
            break
    assert info["fight_ended"] == "boss_killed"
    assert terminated and env.kills == 1
    assert total > env.cfg.reward.kill


def test_dying_terminates_and_costs():
    env, source, _ = make_env()
    source.player_dps = 0.06
    source.boss_dps = 0.0
    env.reset(options={"skip_macros": True})
    total = 0.0
    for _ in range(300):
        _, reward, terminated, truncated, info = env.step(idle_action())
        total += reward
        if terminated or truncated:
            break
    assert info["fight_ended"] == "player_died"
    assert terminated and env.kills == 0
    assert total < 0


def test_action_repeat_hands_every_frame_to_the_retina():
    env, _, _ = make_env()
    env.reset(options={"skip_macros": True})
    _, _, _, _, info = env.step(attack_action())
    assert len(info["frames"]) == env.cfg.action_repeat
    assert env.dt == pytest.approx(env.cfg.action_repeat / env.cfg.target_fps)


def test_dry_run_sends_nothing_and_releases_keys():
    env, _, backend = make_env()
    env.reset(options={"skip_macros": True})
    env.step(attack_action())
    assert set(backend.held_keys) == {"mouse:left"}
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(attack_action())
        if terminated or truncated:
            break
    assert backend.held_keys == ()
    assert env.cfg.dry_run is True
    assert backend.log, "the dry-run backend records what it would have pressed"


def test_runback_macro_plays_between_attempts():
    env, source, backend = make_env()
    env.reset(options={"skip_macros": True})
    env.state = "dead"
    source.t = 0
    source.finished = False
    source.boss_hp = 1.0
    source.player_hp = 1.0
    backend.log.clear()
    env.reset()
    pressed = [key for kind, key in backend.log if kind == "press"]
    assert "w" in pressed, "the run-back macro should have walked forward"


def test_budget_arithmetic_is_reported():
    env, _, _ = make_env()
    budget = env.budget(hours=1.0)
    assert budget["seconds_per_decision"] == pytest.approx(env.dt)
    assert budget["attempts_per_hour"] > 0
    assert budget["hours_for_1M_decisions"] > 0


def test_detector_config_rejects_impossible_thresholds():
    with pytest.raises(ValueError, match="boss_bar_present"):
        DetectorConfig(boss_bar_present=0.5, boss_dead_below=0.1)


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ([(1.0, None)] * 4 + [(1.0, 1.0)] * 4 + [(0.9, 0.3)] * 3 + [(0.9, 0.02)] * 2
         + [(0.9, None)] * 4, ["started", "boss_killed"]),
        ([(1.0, None)] * 4 + [(1.0, 1.0)] * 4 + [(1.0, 1.0)] * 3 + [(1.0, None)] * 4,
         ["started", "aborted"]),
        ([(1.0, None)] * 3 + [(1.0, 1.0)] * 1 + [(1.0, None)] * 4, []),
    ],
    ids=["killed", "walked-out", "one-frame-flicker"],
)
def test_fight_state_transitions(sequence, expected):
    detector = FightStateDetector()
    events = []
    for player, boss in sequence:
        state = detector.update(hud_frame(player, boss))
        if state["fight_started"]:
            events.append("started")
        if state["fight_ended"]:
            events.append(state["fight_ended"])
    assert events == expected


def test_reward_weights_are_configurable():
    env, _, _ = make_env()
    env.cfg.reward = BossRewardConfig(boss_damage=100.0, kill=0.0, survival=0.0, player_damage=0.0)
    env.reset(options={"skip_macros": True})
    _, reward, _, _, info = env.step(attack_action())
    assert reward == pytest.approx(100.0 * info["boss_damage"])


def test_macro_duration():
    macro = Macro(steps=((("forward",), 1.5), ((), 0.5)))
    assert macro.duration_s == pytest.approx(2.0)


def test_boss_env_runs_a_recorded_route_on_reset(tmp_path):
    from flyds1.envs.input_backends import DryRunBackend
    from flyds1.envs.navigation import Route, Waypoint

    backend = DryRunBackend()
    source = FakeFight(backend)

    # record a two-leg route against this fake game, then point the env at it
    route = Route(
        waypoints=(
            Waypoint(buttons=("forward",), seconds=0.01, name="leg 0", retries=0),
            Waypoint(buttons=("forward",), seconds=0.01, name="leg 1", retries=0),
        ),
        name="test",
    )
    bindings = {b.name: b.key for b in DARKSOULS_ACTIONS.buttons}
    recorded = route.record(
        backend, bindings, source.grab, sleep_fn=lambda _s: None, fps=10_000.0
    )
    path = recorded.save(tmp_path / "route.npz")

    env, source2, backend2 = make_env(route_path=str(path))
    assert env.route is not None and len(env.route.waypoints) == 2

    env.reset(options={"skip_macros": True})
    env.state = "dead"          # force the run-back path on the next reset
    source2.t = 0
    source2.finished = False
    source2.boss_hp = 1.0
    source2.player_hp = 1.0
    backend2.log.clear()

    _, info = env.reset()
    pressed = [key for kind, key in backend2.log if kind == "press"]
    assert "w" in pressed, "the recorded route should have walked forward"
    assert info["route"] is not None
    assert env.last_route_result is not None


def test_boss_env_passes_death_handling_config_to_the_route():
    """route_handle_death / route_max_respawns / wait_after_death_s must reach
    Route.run, not just live as unused config -- verified by intercepting the
    call rather than staging a real death, which needs a lot of fake frames."""
    from unittest.mock import MagicMock

    from flyds1.envs.navigation import Route, RouteResult, Waypoint

    backend = DryRunBackend()
    source = FakeFight(backend)
    cfg = BossConfig(
        target_fps=10_000.0, action_repeat=2, max_fight_steps=200, max_wait_steps=50,
        wait_after_death_s=7.5, route_handle_death=False, route_max_respawns=9,
        frame_width=64, frame_height=36,
    )
    env = BossFightEnv(cfg, DARKSOULS_ACTIONS, frame_source=source, input_backend=backend,
                       sleep_fn=lambda _s: None)
    env.route = Route(waypoints=(Waypoint(buttons=(), seconds=0.001),))
    env.route.run = MagicMock(return_value=RouteResult(completed=True, deaths=2))

    env.reset(options={"skip_macros": True})
    env.state = "dead"
    source.t = 0
    source.finished = False
    source.boss_hp = source.player_hp = 1.0
    env.reset()

    _, kwargs = env.route.run.call_args
    assert kwargs["handle_death"] is False
    assert kwargs["max_respawns"] == 9
    assert kwargs["respawn_wait_s"] == 7.5
    assert env.route_deaths == 2
