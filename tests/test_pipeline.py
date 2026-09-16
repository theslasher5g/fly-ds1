"""End-to-end: config, pipeline assembly and the CLI."""

import numpy as np
import pytest

from flyds1.cli import main
from flyds1.config import ExperimentConfig
from flyds1.plotting import facet_image, filmstrip, save_png, to_uint8

SMALL = {
    "connectome.spec": "synthetic:rings=2",
    # a 2-ring eye spans +-10 deg per eye; the screen has to match or the agent
    # sees only the middle of it (which the selftest now checks for)
    "frontend.screen.fov_h_deg": 45.0,
    "env.arena.width": 96,
    "env.arena.height": 54,
    "env.arena.max_steps": 40,
    "frontend.motion_mode": "pooled",
    "env.stack_k": 2,
}


def small_config() -> ExperimentConfig:
    return ExperimentConfig().apply_overrides(SMALL)


def test_config_round_trip(tmp_path):
    cfg = small_config()
    path = cfg.dump_yaml(tmp_path / "c.yaml")
    back = ExperimentConfig.from_yaml(path)
    assert back.connectome.spec == cfg.connectome.spec
    assert back.frontend.screen.width == cfg.frontend.screen.width
    assert back.rate.dt == cfg.rate.dt


def test_config_rejects_unknown_keys():
    with pytest.raises(KeyError, match="no field"):
        ExperimentConfig.from_mapping({"connectome": {"nonsense": 1}})
    with pytest.raises(KeyError, match="unknown config key"):
        ExperimentConfig().apply_overrides({"training.nope": 1})


def test_pipeline_assembles_both_brain_locations():
    from flyds1.pipeline import build_network, make_env

    cfg = small_config()
    network = build_network(cfg)
    env, layout = make_env(cfg, network, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (cfg.env.stack_k, layout.size)

    wrapped, _ = make_env(cfg.apply_overrides({"env.brain_location": "wrapper"}), network, seed=0)
    obs2, _ = wrapped.reset(seed=0)
    assert obs2.shape == (network.n_outputs,)


def test_network_cache_is_reused(tmp_path):
    from flyds1.pipeline import build_network

    cfg = small_config().apply_overrides({"connectome.cache": str(tmp_path / "net.npz")})
    first = build_network(cfg)
    assert (tmp_path / "net.npz").exists()
    second = build_network(cfg)
    assert second.n_neurons == first.n_neurons
    assert (second.W != first.W).nnz == 0


def test_png_writer(tmp_path):
    image = (np.random.default_rng(0).random((8, 12)) * 255).astype(np.uint8)
    path = save_png(image, tmp_path / "x.png")
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in data[:20] and data[-8:-4] == b"IEND"


def test_facet_image_and_filmstrip():
    coords = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.3]])
    img = facet_image(coords, np.array([-1.0, 0.0, 1.0]), size=40, signed=True)
    assert img.shape[-1] == 3 and img.dtype == np.uint8
    strip = filmstrip([img, to_uint8(np.zeros((10, 10)))])
    assert strip.ndim == 3


def _cli(*args) -> int:
    return main(list(args))


def test_cli_doctor_and_fetch(capsys):
    assert _cli("doctor") == 0
    assert "torch" in capsys.readouterr().out
    assert _cli("fetch") == 0
    assert "codex.flywire.ai" in capsys.readouterr().out


def test_cli_info_and_build(tmp_path, capsys):
    args = []
    for key, value in SMALL.items():
        args += ["--set", f"{key}={value}"]
    assert _cli(*args, "info") == 0
    out = capsys.readouterr().out
    assert "photoreceptors" in out and "spectral radius" in out

    assert _cli(*args, "build", "--out", str(tmp_path / "net.npz")) == 0
    assert (tmp_path / "net.npz").exists()


def test_cli_selftest_passes(tmp_path, capsys):
    args = []
    for key, value in SMALL.items():
        args += ["--set", f"{key}={value}"]
    assert _cli(*args, "selftest", "--png", str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "all checks passed" in out
    assert "FAIL" not in out
    assert list(tmp_path.glob("*.png"))


def test_cli_play_with_random_policy(capsys):
    args = []
    for key, value in SMALL.items():
        args += ["--set", f"{key}={value}"]
    assert _cli(*args, "play", "--episodes", "1") == 0
    assert "mean return" in capsys.readouterr().out


def test_cli_reports_bad_overrides(capsys):
    assert _cli("--set", "connectome.nope=1", "info") == 2
    assert "error" in capsys.readouterr().err


def test_pipeline_assembles_the_game_environment():
    """The real-game path, exercised against the dummy screen capture."""
    from flyds1.pipeline import action_spec_for, build_network, make_env

    cfg = small_config().apply_overrides(
        {
            "env.kind": "game",
            "env.game.frame_source": "dummy",
            "env.game.target_fps": 10_000,
            "env.game.frame_width": 96,
            "env.game.frame_height": 54,
            "env.game.max_steps": 6,
            "env.game.dry_run": True,
        }
    )
    network = build_network(cfg)
    env, layout = make_env(cfg, network, seed=0)
    obs, _ = env.reset(options={"skip_wait": True})
    assert obs.shape == (cfg.env.stack_k, layout.size)

    spec = action_spec_for(cfg)
    assert spec.size == 13  # 9 souls buttons + 4 camera buttons, not the arena's four
    action = np.zeros(spec.size, dtype=np.float32)
    action[0] = 1.0
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["dry_run"] is True
    assert np.isfinite(obs).all()


def test_retina_wrapper_exposes_its_last_observation(tmp_path):
    """The live viewer reads this back instead of re-processing the frame."""
    from flyds1.pipeline import build_network, make_env

    cfg = small_config().apply_overrides({"env.brain_location": "wrapper"})
    network = build_network(cfg)
    env, _ = make_env(cfg, network, seed=0)
    retina = env  # FlyBrainWrapper -> RetinaWrapper
    while type(retina).__name__ != "RetinaWrapper":
        retina = retina.env
    retina.keep_frame_in_info = True

    env.reset(seed=0)
    first = np.array(retina.last_observation, copy=True)
    assert not first.any(), "a still fly looking at a still scene sees nothing"

    # move: the Weber contrast is zero by construction until the scene changes
    walk = np.array([1, 0, 0, 1], dtype=np.int8)
    for _ in range(3):
        _, _, _, _, info = env.step(walk)
    assert "frame" in info
    assert retina.last_observation.shape == (retina.front_end.obs_size,)
    assert retina.last_observation.any()
    assert not np.array_equal(first, retina.last_observation)


def test_facet_layout_is_cached_between_renders():
    """Regression: the pixel->facet assignment was rebuilt every frame, which
    at a real fly's 721 facets per eye meant seconds per panel, not
    milliseconds."""
    import time

    from flyds1.hexlattice import hex_lattice
    from flyds1.plotting import _FACET_LAYOUTS, facet_image

    _FACET_LAYOUTS.clear()
    _, coords = hex_lattice(rings=6, spacing=5.0)
    values = np.random.default_rng(0).normal(size=len(coords))

    first = facet_image(coords, values, size=80, signed=True)
    assert len(_FACET_LAYOUTS) == 1

    start = time.perf_counter()
    again = facet_image(coords, values, size=80, signed=True)
    cached_seconds = time.perf_counter() - start

    assert np.array_equal(first, again)
    assert len(_FACET_LAYOUTS) == 1, "a second render must reuse the layout"
    assert cached_seconds < 0.05

    facet_image(coords, values, size=120, signed=True)
    assert len(_FACET_LAYOUTS) == 2, "a different size needs its own layout"


def test_default_config_uses_a_real_fly_eye():
    """The default eye should be the animal's, not a convenient fraction of it."""
    from flyds1.connectome.loader import load_connectome
    from flyds1.vision.frontend import DROSOPHILA_OMMATIDIA_PER_EYE

    cfg = ExperimentConfig()
    connectome = load_connectome(cfg.connectome.spec)
    photoreceptors = connectome.neurons.photoreceptor_mask()
    per_eye = photoreceptors.sum() / 2
    assert per_eye / DROSOPHILA_OMMATIDIA_PER_EYE > 0.9


def test_warns_when_overrides_target_the_wrong_env_section(capsys):
    """Regression: --set env.boss.frame_source=mss with env.kind=game was
    silently ignored, leaving the agent watching a dummy source while a real
    game sat untouched."""
    from flyds1.cli import _load_config

    class Args:
        config = None
        set = ["env.kind=game", "env.boss.frame_source=mss", "env.boss.capture.width=1280"]

    cfg = _load_config(Args())
    assert cfg.env.kind == "game"
    assert cfg.env.game.frame_source == "dummy"  # the override never touched this
    err = capsys.readouterr().err
    assert "env.boss.frame_source" in err and "env.boss.capture.width" in err
    assert "env.game.*" in err


def test_no_warning_when_overrides_match_the_env_kind(capsys):
    from flyds1.cli import _load_config

    class Args:
        config = None
        set = ["env.kind=game", "env.game.frame_source=mss"]

    _load_config(Args())
    assert capsys.readouterr().err == ""


def test_live_warns_about_dry_run(capsys, monkeypatch):
    """Regression: dry_run defaults to True on the game/boss envs, and the
    live loop ran, "steps" climbed, and the character stood still with no
    indication why -- every key press was suppressed on purpose, silently."""
    from flyds1.envs.input_backends import DryRunBackend

    # dry_run=False makes ScreenGameEnv construct a *real* input backend
    # ("auto" resolves to xdotool/pydirectinput), which is exactly the point
    # of that default -- but it means this test needs a stand-in wherever no
    # such backend exists (this container has neither).  What is under test
    # here is the printed notice, not backend construction.
    monkeypatch.setattr("flyds1.envs.game.make_input_backend", lambda *a, **k: DryRunBackend())

    args = []
    for key, value in SMALL.items():
        args += ["--set", f"{key}={value}"]
    args += ["--set", "env.kind=game", "--set", "env.game.max_steps=2"]

    assert _cli(*args, "live", "--episodes", "1", "--port", "0") == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "no keys are being sent" in out

    assert _cli(*args, "--set", "env.game.dry_run=false",
                "live", "--episodes", "1", "--port", "0") == 0
    out = capsys.readouterr().out
    assert "LIVE INPUT" in out and "keys ARE being sent" in out

    # dry_run=False but input_backend explicitly "dry" must not claim otherwise
    assert _cli(*args, "--set", "env.game.dry_run=false", "--set", "env.game.input_backend=dry",
                "live", "--episodes", "1", "--port", "0") == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "input_backend is explicitly 'dry'" in out
    assert "LIVE INPUT" not in out


def test_live_says_nothing_about_dry_run_for_the_arena(capsys):
    args = []
    for key, value in SMALL.items():
        args += ["--set", f"{key}={value}"]
    assert _cli(*args, "live", "--episodes", "1", "--port", "0") == 0
    out = capsys.readouterr().out
    assert "DRY RUN" not in out and "LIVE INPUT" not in out


def test_cli_live_writes_the_panel_by_default(tmp_path):
    args = []
    for key, value in SMALL.items():
        args += ["--set", f"{key}={value}"]
    assert _cli(*args, "live", "--episodes", "1", "--port", "0",
                "--out", str(tmp_path)) == 0
    import json

    layout = json.loads((tmp_path / "layout.json").read_text())
    state = json.loads((tmp_path / "state.json").read_text())
    assert layout["n_neurons"] > 0
    assert set(layout["eyes"]) == {"left", "right"}, "the panel shows both eyes"
    assert len(layout["dn_labels"]) == len(state["dn"])
    # the hemisphere gauge and the population-flow panel's data
    assert -1.0 <= state["balance"] <= 1.0
    assert {"sensory", "optic", "descending"} <= {s["name"] for s in state["stages"]}
    assert state["types"]["T4a"] >= 0.0
    assert (tmp_path / "fly.js").exists() and (tmp_path / "frame.png").exists()


def test_cli_live_can_still_render_the_classic_filmstrip(tmp_path):
    args = []
    for key, value in SMALL.items():
        args += ["--set", f"{key}={value}"]
    assert _cli(*args, "live", "--episodes", "1", "--port", "0",
                "--viewer", "classic", "--out", str(tmp_path)) == 0
    assert (tmp_path / "panel.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "stats.txt").read_text()
    assert not (tmp_path / "layout.json").exists()
