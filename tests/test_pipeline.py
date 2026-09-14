"""End-to-end: config, pipeline assembly and the CLI."""

import numpy as np
import pytest

from flyds1.cli import main
from flyds1.config import ExperimentConfig
from flyds1.plotting import facet_image, filmstrip, save_png, to_uint8

SMALL = {
    "connectome.spec": "synthetic:rings=2",
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
