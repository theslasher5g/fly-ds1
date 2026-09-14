import numpy as np
import pytest

pytest.importorskip("gymnasium")

from flyds1.envs.arena import ArenaConfig, FlyArenaEnv  # noqa: E402
from flyds1.envs.game import GameConfig, ScreenGameEnv  # noqa: E402
from flyds1.envs.input_backends import make_input_backend  # noqa: E402
from flyds1.envs.reward import (  # noqa: E402
    BarRegion,
    HealthBarReader,
    ScreenRewardEstimator,
    frame_flow_magnitude,
)
from flyds1.envs.screen import make_frame_source  # noqa: E402
from flyds1.envs.wrappers import RetinaWrapper  # noqa: E402
from flyds1.motor.actions import ARENA_ACTIONS, DARKSOULS_ACTIONS  # noqa: E402
from flyds1.vision.frontend import FrontEndConfig  # noqa: E402

SMALL_ARENA = ArenaConfig(width=96, height=54, max_steps=50, seed=1)


def hud_frame(player=1.0, boss=1.0, h=90, w=160):
    frame = np.zeros((h, w, 3))
    x0, x1 = int(0.035 * w), int(0.245 * w)
    y0, y1 = int(0.045 * h), int(np.ceil(0.062 * h))
    frame[y0:y1, x0 : x0 + int((x1 - x0) * player)] = (0.7, 0.05, 0.05)
    bx0, bx1 = int(0.255 * w), int(0.745 * w)
    by0, by1 = int(0.885 * h), int(np.ceil(0.900 * h))
    frame[by0:by1, bx0 : bx0 + int((bx1 - bx0) * boss)] = (0.7, 0.05, 0.05)
    return frame


def test_arena_conforms_to_the_gym_api():
    from gymnasium.utils.env_checker import check_env

    check_env(FlyArenaEnv(SMALL_ARENA), skip_render_check=True)


def test_arena_frames_have_structure_and_change():
    env = FlyArenaEnv(SMALL_ARENA)
    obs, _ = env.reset(seed=1)
    frames = [obs]
    act = np.zeros(ARENA_ACTIONS.size, dtype=np.float32)
    act[0] = act[3] = 1.0
    for _ in range(20):
        obs, *_ = env.step(act)
        frames.append(obs)
    stack = np.stack(frames)
    assert stack.var() > 1e-4, "a featureless world gives the motion detectors nothing"
    assert np.abs(np.diff(stack, axis=0)).mean() > 1e-4


def test_arena_walls_block_movement():
    env = FlyArenaEnv(SMALL_ARENA)
    env.reset(seed=1)
    env.pos = np.array([1.5, 1.5])
    env.angle = np.pi  # straight at the west wall
    act = np.zeros(ARENA_ACTIONS.size, dtype=np.float32)
    act[0] = 1.0
    for _ in range(10):
        env.step(act)
    assert env.pos[0] >= 1.0


def test_arena_enemy_damages_the_player():
    # the chaser only exists in the survive task (or with target_enemy set)
    env = FlyArenaEnv(ArenaConfig(width=96, height=54, max_steps=200, seed=1, task="survive"))
    env.reset(seed=1)
    env.enemy = env.pos.copy() + 0.1
    for _ in range(20):
        env.step(np.zeros(ARENA_ACTIONS.size, dtype=np.float32))
    assert env.hp < 100.0


def test_retina_wrapper_produces_the_declared_observation(network):
    env = RetinaWrapper(FlyArenaEnv(SMALL_ARENA), network, FrontEndConfig(motion_mode="pooled"))
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()
    obs2, *_ = env.step(np.ones(ARENA_ACTIONS.size, dtype=np.float32))
    assert obs2.shape == obs.shape


def test_retina_wrapper_matches_screen_geometry_to_frames(network):
    env = RetinaWrapper(FlyArenaEnv(SMALL_ARENA), network, FrontEndConfig())
    assert env.front_end.cfg.screen.width == SMALL_ARENA.width
    assert env.front_end.cfg.screen.height == SMALL_ARENA.height


def test_health_bar_reader():
    reader = HealthBarReader(BarRegion(0.035, 0.045, 0.245, 0.062))
    assert reader.read(hud_frame(player=1.0)) > 0.95
    assert reader.read(hud_frame(player=0.5)) == pytest.approx(0.5, abs=0.1)
    assert reader.read(hud_frame(player=0.0)) < 0.05


def test_reward_reacts_to_damage_and_death():
    est = ScreenRewardEstimator()
    est(hud_frame(1.0, 1.0))
    hurt, info = est(hud_frame(0.6, 0.8))
    assert info["player_damage"] > 0 and info["boss_damage"] > 0
    dead, info = est(hud_frame(0.0, 0.8))
    assert info["died"] is True
    assert dead < hurt


def test_flow_magnitude():
    assert frame_flow_magnitude(np.zeros((4, 4)), np.zeros((4, 4))) == 0.0
    assert frame_flow_magnitude(np.zeros((4, 4)), np.ones((4, 4))) == 1.0


def test_dry_run_backend_tracks_held_keys():
    backend = make_input_backend("dry")
    bindings = {b.name: b.key for b in DARKSOULS_ACTIONS.buttons}
    backend.apply({"forward": True, "attack": True}, bindings)
    assert set(backend.held_keys) == {"w", "mouse:left"}
    backend.apply({"forward": True}, bindings)
    assert set(backend.held_keys) == {"w"}
    backend.release_all()
    assert backend.held_keys == ()


def test_game_env_runs_against_a_dummy_screen():
    env = ScreenGameEnv(
        GameConfig(frame_source="dummy", target_fps=10_000, max_steps=5, dry_run=True)
    )
    obs, _ = env.reset(options={"skip_wait": True})
    assert obs.shape == (env.cfg.frame_height, env.cfg.frame_width)
    action = np.zeros(DARKSOULS_ACTIONS.size, dtype=np.float32)
    action[0] = 1.0
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    assert info["dry_run"] is True
    assert info["steps"] == 5
    env.close()
    assert env.input.held_keys == ()


def test_game_env_never_sends_input_in_dry_run():
    env = ScreenGameEnv(GameConfig(frame_source="dummy", target_fps=10_000, dry_run=True))
    env.reset(options={"skip_wait": True})
    action = np.ones(DARKSOULS_ACTIONS.size, dtype=np.float32)
    env.step(action)
    from flyds1.envs.input_backends import DryRunBackend

    assert isinstance(env.input, DryRunBackend)


def test_action_spec_decoding():
    """Buttons are binary; the camera is four more buttons by default."""
    assert DARKSOULS_ACTIONS.is_binary
    action = np.zeros(DARKSOULS_ACTIONS.size, dtype=int)
    for name in ("forward", "left", "attack"):
        action[DARKSOULS_ACTIONS.button_index(name)] = 1
    action[DARKSOULS_ACTIONS.n_buttons + 1] = 1  # camera_right
    action[DARKSOULS_ACTIONS.n_buttons + 2] = 1  # camera_up
    held, (dx, dy) = DARKSOULS_ACTIONS.decode(action)
    assert held["forward"] and held["left"] and held["attack"]
    assert not held["back"] and not held["block"]
    assert dx == pytest.approx(DARKSOULS_ACTIONS.camera_scale_px)
    assert dy == pytest.approx(-DARKSOULS_ACTIONS.camera_scale_px)
    with pytest.raises(ValueError):
        DARKSOULS_ACTIONS.decode(np.zeros(3))


def test_action_spec_continuous_camera_still_works():
    spec = DARKSOULS_ACTIONS.with_camera("continuous")
    assert not spec.is_binary and spec.size == DARKSOULS_ACTIONS.n_buttons + 2
    held, (dx, dy) = spec.decode(np.r_[np.ones(spec.n_buttons), 0.5, -0.25])
    assert held["forward"]
    assert dx == pytest.approx(20.0) and dy == pytest.approx(-10.0)


def test_action_spaces_are_binary():
    from gymnasium import spaces

    assert isinstance(FlyArenaEnv(SMALL_ARENA).action_space, spaces.MultiBinary)
    assert isinstance(DARKSOULS_ACTIONS.gym_space(), spaces.MultiBinary)
    assert isinstance(DARKSOULS_ACTIONS.with_camera("continuous").gym_space(), spaces.Box)


def test_dummy_frame_source_moves():
    src = make_frame_source("dummy")
    assert not np.array_equal(src.grab(), src.grab())


def _target_oracle(env):
    """Steer at the marker: the behaviour the target task is meant to reward."""
    e = env.unwrapped
    rel = e.target - e.pos
    err = (np.arctan2(rel[1], rel[0]) - e.angle + np.pi) % (2 * np.pi) - np.pi
    action = np.zeros(4, dtype=np.int8)
    if abs(err) > 0.15:
        action[3 if err > 0 else 2] = 1
    if abs(err) < 1.0:
        action[0] = 1
    return action


def _episode(env, policy, seed):
    env.reset(seed=seed)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(policy(env))
        total += reward
        if terminated or truncated:
            return total, info


def test_target_task_picks_the_open_map():
    from flyds1.envs.arena import OPEN_MAP

    cfg = ArenaConfig(task="target")
    assert cfg.tile_map == tuple(OPEN_MAP)
    assert ArenaConfig(task="survive").tile_map != tuple(OPEN_MAP)
    with pytest.raises(ValueError, match="task must be"):
        ArenaConfig(task="nonsense")


def test_target_is_visible_and_reachable():
    env = FlyArenaEnv(ArenaConfig(width=160, height=90, max_steps=300, task="target"))
    env.reset(seed=3)
    rel = env.target - env.pos
    env.angle = float(np.arctan2(rel[1], rel[0]))
    frame = env._observe()
    assert frame.max() > 0.97, "the marker must be the brightest thing in view"
    assert (frame > 0.97).sum() > 10

    _, info = _episode(env, _target_oracle, seed=10_000)
    assert info["targets_reached"] >= 1


def test_target_task_separates_seeing_from_not_seeing():
    """The point of this task: a policy that steers at the marker must score
    far above one that does not, or it cannot teach anything."""
    env = FlyArenaEnv(ArenaConfig(width=160, height=90, max_steps=300, task="target"))
    oracle = [_episode(env, _target_oracle, 10_000 + e)[0] for e in range(5)]
    idle = [_episode(env, lambda _e: np.zeros(4, dtype=np.int8), 10_000 + e)[0] for e in range(5)]
    assert np.mean(oracle) > np.mean(idle) + 20


def test_survive_task_still_works():
    env = FlyArenaEnv(ArenaConfig(width=96, height=54, max_steps=60, task="survive"))
    obs, info = env.reset(seed=1)
    assert "enemy_distance" in info
    _, info = _episode(env, lambda _e: np.array([1, 0, 0, 0], dtype=np.int8), seed=1)
    assert info["steps"] == 60 or info["hp"] <= 0


def test_dry_run_false_alone_is_enough_to_select_a_real_backend(monkeypatch):
    """Regression: input_backend defaulted to the string "dry" as its own
    field, independently of dry_run. Someone flipping dry_run=False without
    also touching input_backend -- exactly what the dry-run notice tells them
    to do -- still got a DryRunBackend and silently sent nothing."""
    from flyds1.envs.game import GameConfig, ScreenGameEnv

    captured = {}

    def fake_make_input_backend(kind, window=None):
        captured["kind"] = kind
        from flyds1.envs.input_backends import DryRunBackend

        return DryRunBackend()

    monkeypatch.setattr("flyds1.envs.game.make_input_backend", fake_make_input_backend)
    ScreenGameEnv(GameConfig(frame_source="dummy", dry_run=False))
    assert captured["kind"] != "dry"


def test_dry_run_true_forces_dry_regardless_of_input_backend(monkeypatch):
    from flyds1.envs.game import GameConfig, ScreenGameEnv

    captured = {}

    def fake_make_input_backend(kind, window=None):
        captured["kind"] = kind
        from flyds1.envs.input_backends import DryRunBackend

        return DryRunBackend()

    monkeypatch.setattr("flyds1.envs.game.make_input_backend", fake_make_input_backend)
    ScreenGameEnv(GameConfig(frame_source="dummy", dry_run=True, input_backend="xdotool"))
    assert captured["kind"] == "dry"
