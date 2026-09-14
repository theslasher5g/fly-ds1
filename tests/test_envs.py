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
    env = FlyArenaEnv(ArenaConfig(**{**SMALL_ARENA.__dict__, "max_steps": 200}))
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
    held, (dx, dy) = DARKSOULS_ACTIONS.decode(
        np.array([1, -1, 1, -1, -1, 1, -1, -1, -1, 0.5, -0.25])
    )
    assert held["forward"] and held["left"] and held["attack"]
    assert not held["back"] and not held["block"]
    assert dx == pytest.approx(20.0) and dy == pytest.approx(-10.0)
    with pytest.raises(ValueError):
        DARKSOULS_ACTIONS.decode(np.zeros(3))


def test_dummy_frame_source_moves():
    src = make_frame_source("dummy")
    assert not np.array_equal(src.grab(), src.grab())
