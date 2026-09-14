"""Command line entry point: ``flyds1 <command>``.

    flyds1 doctor                      which optional dependencies are installed
    flyds1 fetch                       how to obtain a real connectome
    flyds1 build   --out data/net.npz  build and cache the wired network
    flyds1 info                        summarise config, network and retina
    flyds1 selftest --png out/         run the whole pipeline offline and check it
    flyds1 tune                        sweep the recurrent gain and recommend one
    flyds1 train   --config c.yaml     PPO training
    flyds1 play    --model m.zip       roll out a policy and report statistics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from flyds1.config import ExperimentConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    overrides = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        overrides[key.strip()] = _parse_scalar(value.strip())
    return cfg.apply_overrides(overrides) if overrides else cfg


def _parse_scalar(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    return text


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_doctor(args) -> int:
    rows = [
        ("numpy", "core", True),
        ("scipy", "core", True),
        ("yaml", "core", True),
        ("torch", "step 2 (differentiable net) + step 6", False),
        ("gymnasium", "step 4 (environments)", False),
        ("stable_baselines3", "step 6 (PPO)", False),
        ("pandas", "step 1 (natverse route)", False),
        ("navis", "step 1 (natverse route)", False),
        ("fafbseg", "step 1 (live FlyWire queries)", False),
        ("mss", "step 4 (screen capture)", False),
        ("pydirectinput", "step 5 (Windows input)", False),
        ("brian2", "step 2 (spiking alternative)", False),
    ]
    print(f"{'package':20s} {'status':9s} used for")
    missing_core = False
    for module, purpose, required in rows:
        ok = _has(module)
        if required and not ok:
            missing_core = True
        print(f"{module:20s} {'ok' if ok else 'missing':9s} {purpose}")
    print("\ninstall extras with:  pip install -e '.[torch,rl]'   (and [connectome], [game])")
    return 1 if missing_core else 0


def cmd_fetch(args) -> int:
    from flyds1.connectome.codex import download_instructions

    print(download_instructions())
    return 0


def cmd_build(args) -> int:
    from flyds1.pipeline import build_network

    cfg = _load_config(args)
    if args.out:
        cfg = cfg.apply_overrides({"connectome.cache": str(args.out)})
    net = build_network(cfg, verbose=True)
    print(net.describe())
    print(f"spectral radius: {net.spectral_radius():.3f}")
    if args.out:
        print(f"saved to {args.out}")
    return 0


def cmd_info(args) -> int:
    from flyds1.pipeline import build_network, make_front_end
    from flyds1.vision.encoder import build_encoder_wiring

    cfg = _load_config(args)
    net = build_network(cfg)
    fe = make_front_end(cfg, net)
    wiring = build_encoder_wiring(net, fe.layout, inject_motion=cfg.encoder.inject_motion)
    print(net.describe())
    print(f"spectral radius: {net.spectral_radius():.3f}")
    print(fe.describe())
    print(wiring.describe())
    print(f"rate dynamics: dt={cfg.rate.dt * 1000:.2f} ms x {cfg.rate.steps_per_frame} substeps "
          f"= {cfg.rate.frame_dt * 1000:.1f} ms per frame, activation={cfg.rate.activation}")
    print(f"env: {cfg.env.kind}, brain in {cfg.env.brain_location}, stack_k={cfg.env.stack_k}")
    return 0


def cmd_selftest(args) -> int:
    """Run every step once, offline, with numeric checks on each."""
    from flyds1.net.reference import RateNetwork
    from flyds1.pipeline import action_spec_for, build_network, make_env
    from flyds1.vision.encoder import build_encoder_wiring, encode_numpy

    cfg = _load_config(args)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f' -- {detail}' if detail else ''}")
        if not ok:
            failures.append(name)

    print("step 1/2: connectome -> wired network")
    net = build_network(cfg)
    check("neurons present", net.n_neurons > 0, f"{net.n_neurons} neurons")
    check("photoreceptors present", net.n_inputs > 0, f"{net.n_inputs}")
    check("descending neurons present", net.n_outputs > 0, f"{net.n_outputs}")
    check("both signs of synapse present", (net.W.data > 0).any() and (net.W.data < 0).any())
    radius = net.spectral_radius()
    check("spectral radius below 2", radius < 2.0, f"{radius:.3f}")

    print("step 3: screen -> eye -> motion")
    from flyds1.pipeline import make_front_end

    front_end = make_front_end(cfg, net)
    check(
        "retina covers the screen",
        front_end.screen_coverage() > 0.9,
        f"{front_end.screen_coverage():.0%} of the screen width "
        f"(raise connectome rings, or lower frontend.screen.fov_h_deg)",
    )
    env, layout = make_env(cfg, net, seed=0)
    obs, _ = env.reset(seed=0)
    check("observation finite", bool(np.isfinite(np.asarray(obs)).all()))
    spec = action_spec_for(cfg)
    forward = np.zeros(spec.size, dtype=np.float32)
    forward[0] = 1.0
    frames = []
    for k in range(30):
        act = forward.copy()
        act[3 if k % 20 < 10 else 2] = 1.0
        obs, reward, terminated, truncated, info = env.step(act)
        frames.append(np.asarray(obs, dtype=float).ravel())
        if terminated or truncated:
            break
    stack = np.stack(frames)
    check("observations vary over time", float(stack.std()) > 1e-6, f"std={stack.std():.4f}")

    print("step 2b/5: brain and read-out")
    from flyds1.net.diagnostics import signal_transmission

    report = signal_transmission(net, cfg.rate)
    check(
        "stimulus signal reaches the descending neurons",
        report.usable,
        f"DN std {report.descending_std:.2e} (raise connectome.gain if this fails; see 'flyds1 tune')",
    )
    sim = RateNetwork(net, cfg.rate)
    rest, converged = sim.fixed_point(None, max_steps=4000)
    check("resting state converges", converged, f"mean rate {rest.mean():.3f}")
    check("network not silent at rest", float(rest.max()) > 1e-6)
    wiring = build_encoder_wiring(net, layout, inject_motion=cfg.encoder.inject_motion)
    if cfg.env.brain_location == "policy":
        retina_obs = stack[-1][-layout.size :] if stack.shape[1] != layout.size else stack[-1]
        drive = encode_numpy(retina_obs, layout, wiring)
        sim.reset()
        for _ in range(20):
            sim.step(drive)
        dn = sim.outputs()
        check("descending neurons respond", float(np.abs(dn).max()) > 1e-6, f"max {np.abs(dn).max():.3f}")
        check("descending rates differ", float(dn.std()) > 1e-9, f"std {dn.std():.4f}")

    if args.png:
        out = Path(args.png)
        written = _write_diagnostics(cfg, net, out)
        print(f"step 3 diagnostics written: {', '.join(str(p) for p in written)}")

    print(f"\n{'all checks passed' if not failures else f'{len(failures)} check(s) FAILED: {failures}'}")
    return 1 if failures else 0


def _write_diagnostics(cfg: ExperimentConfig, net, out_dir: Path) -> list[Path]:
    """Render frame / retina / motion images -- the visual sanity check."""
    from flyds1.pipeline import action_spec_for, make_frame_env, make_front_end
    from flyds1.plotting import facet_image, filmstrip, save_png, to_uint8

    fe = make_front_end(cfg, net)
    env = make_frame_env(cfg, seed=1)
    spec = action_spec_for(cfg)
    frame, _ = env.reset(seed=1)
    act = np.zeros(spec.size, dtype=np.float32)
    act[0] = 1.0
    act[3] = 1.0  # walk forward while turning: guarantees optic flow
    obs = None
    motion_sum = None
    for k in range(24):
        frame, *_ = env.step(act)
        obs = fe.process(frame, fe.cfg.reichardt.tau_delay)
        _, m = fe.layout.split(np.asarray(obs))
        if k >= 8:  # skip the adaptation transient
            motion_sum = m if motion_sum is None else motion_sum + m
    photo, motion = fe.layout.split(np.asarray(obs))
    if motion_sum is not None:
        motion = motion_sum / 16.0
    side = sorted(fe.eyes)[-1]
    slots = fe.eyes[side]["slots"]
    coords = fe.eyes[side]["coords"]

    panels = [to_uint8(np.asarray(frame, dtype=float))]
    panels.append(facet_image(coords, photo[slots], signed=True))
    if fe.layout.motion_mode == "retinotopic":
        horizontal = motion[slots][:, 0] - motion[slots][:, 1]
        panels.append(facet_image(coords, horizontal, signed=True))
    strip = filmstrip(panels)
    written = [save_png(strip, out_dir / "pipeline_frame_retina_motion.png")]
    return written


def cmd_tune(args) -> int:
    """Sweep the recurrent gain and report where the signal reaches the DNs."""
    from flyds1.connectome.loader import load_connectome
    from flyds1.net.diagnostics import format_sweep, gain_sweep, recommend_gain

    cfg = _load_config(args)
    connectome = load_connectome(cfg.connectome.spec, spacing_deg=cfg.frontend.spacing_deg)
    gains = args.gains or [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    sweep = gain_sweep(
        connectome,
        gains,
        normalise=cfg.connectome.normalise,
        config=cfg.rate,
        min_synapse_count=cfg.connectome.min_synapse_count,
    )
    print(format_sweep(sweep))
    best = recommend_gain(sweep)
    if best is None:
        print("\nno gain in this range gives a usable descending signal below "
              "criticality -- widen the sweep or check the connectome's annotations")
        return 1
    print(f"\nrecommended: --set connectome.gain={best}")
    return 0


def cmd_train(args) -> int:
    from stable_baselines3.common.vec_env import DummyVecEnv

    from flyds1.pipeline import build_network, make_env, make_model

    cfg = _load_config(args)
    if args.timesteps:
        cfg = cfg.apply_overrides({"training.total_timesteps": int(args.timesteps)})
    net = build_network(cfg, verbose=True)
    out_dir = Path(cfg.training.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump_yaml(out_dir / "config.yaml")

    def factory(rank: int):
        def _make():
            env, _ = make_env(cfg, net, seed=cfg.training.seed + rank)
            return env

        return _make

    venv = DummyVecEnv([factory(i) for i in range(cfg.training.n_envs)])
    if cfg.training.normalize_reward:
        from stable_baselines3.common.vec_env import VecNormalize

        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg.training.gamma)
    _, layout = make_env(cfg, net, seed=cfg.training.seed)
    model = make_model(cfg, venv, net, layout)
    for holder in (model.policy, getattr(model.policy, "pi_features_extractor", None)):
        if hasattr(holder, "describe"):
            print(holder.describe())
            break
    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg.training.total_timesteps, progress_bar=False)
    elapsed = time.perf_counter() - t0
    path = out_dir / "model.zip"
    model.save(path)
    if cfg.training.normalize_reward:
        venv.save(str(out_dir / "vecnormalize.pkl"))
    print(f"trained {cfg.training.total_timesteps} steps in {elapsed:.1f}s -> {path}")
    return 0


def cmd_play(args) -> int:
    from flyds1.pipeline import action_spec_for, build_network, make_env

    cfg = _load_config(args)
    net = build_network(cfg)
    env, _ = make_env(cfg, net, seed=args.seed)
    policy = None
    if args.model:
        from stable_baselines3 import PPO

        policy = PPO.load(args.model, device=cfg.training.device)

    spec = action_spec_for(cfg)
    rng = np.random.default_rng(args.seed)
    returns: list[float] = []
    lengths: list[int] = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        total, steps = 0.0, 0
        while True:
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=args.deterministic)
            else:
                action = rng.uniform(-1, 1, size=spec.size).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
        print(f"episode {episode}: return {total:+.2f} over {steps} steps "
              f"({json.dumps({k: round(v, 2) for k, v in info.items() if isinstance(v, (int, float))})})")
    print(f"\nmean return {np.mean(returns):+.2f} +- {np.std(returns):.2f} "
          f"over {args.episodes} episodes, mean length {np.mean(lengths):.0f}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flyds1", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="experiment YAML (defaults to built-in values)")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE",
        help="override a config value, e.g. --set connectome.spec=synthetic:rings=8",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="report which optional dependencies are installed").set_defaults(
        func=cmd_doctor
    )
    sub.add_parser("fetch", help="print how to download a real connectome").set_defaults(
        func=cmd_fetch
    )
    p_build = sub.add_parser("build", help="build the wired network")
    p_build.add_argument("--out", type=Path, help="save the network as .npz")
    p_build.set_defaults(func=cmd_build)

    sub.add_parser("info", help="summarise network, retina and encoder").set_defaults(func=cmd_info)

    p_self = sub.add_parser("selftest", help="run every pipeline step once with checks")
    p_self.add_argument("--png", type=Path, help="also write diagnostic images to this directory")
    p_self.set_defaults(func=cmd_selftest)

    p_tune = sub.add_parser("tune", help="sweep the recurrent gain")
    p_tune.add_argument("--gains", type=float, nargs="+", help="gains to try")
    p_tune.set_defaults(func=cmd_tune)

    p_train = sub.add_parser("train", help="train with PPO")
    p_train.add_argument("--timesteps", type=int, help="override training.total_timesteps")
    p_train.set_defaults(func=cmd_train)

    p_play = sub.add_parser("play", help="roll out a policy")
    p_play.add_argument("--model", type=Path, help="trained model .zip (random policy if omitted)")
    p_play.add_argument("--episodes", type=int, default=3)
    p_play.add_argument("--seed", type=int, default=0)
    p_play.add_argument("--deterministic", action="store_true")
    p_play.set_defaults(func=cmd_play)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ImportError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
