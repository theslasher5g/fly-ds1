"""Command line entry point: ``flyds1 <command>``.

    flyds1 doctor                      which optional dependencies are installed
    flyds1 fetch                       how to obtain a real connectome
    flyds1 build   --out data/net.npz  build and cache the wired network
    flyds1 info                        summarise config, network and retina
    flyds1 selftest --png out/         run the whole pipeline offline and check it
    flyds1 calibrate --frame shot.png  check HUD regions and field of view on a real frame
    flyds1 watch --frames clip/         run the brain over recorded game footage
    flyds1 route --record r.npz        walk the run-back once and remember what it looks like
    flyds1 live --port 8000            watch the fly play in a browser, live
    flyds1 budget                      how many boss attempts an hour of real time buys
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
    config_path = getattr(args, "config", None)
    cfg = ExperimentConfig.from_yaml(config_path) if config_path else ExperimentConfig()
    overrides = {}
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        overrides[key.strip()] = _parse_scalar(value.strip())
    result = cfg.apply_overrides(overrides) if overrides else cfg
    _warn_on_overrides_for_the_wrong_env(result.env.kind, overrides)
    return result


#: env.<kind> config sections that only take effect when env.kind matches.
#: game and boss each have their own frame_source/capture/dry_run/etc., so
#: --set env.boss.frame_source=mss silently does nothing when env.kind=game
#: (and vice versa) -- there is no error, the env just keeps watching whatever
#: its own section already said, which for the unset one is a dummy source.
#: That is exactly the mistake that leaves a game import watching a synthetic
#: grating while a real game sits untouched on the desktop, so it is worth a
#: warning rather than silent data loss.
_ENV_SPECIFIC_SECTIONS = ("game", "boss")


def _warn_on_overrides_for_the_wrong_env(kind: str, overrides: dict) -> None:
    for section in _ENV_SPECIFIC_SECTIONS:
        if section == kind:
            continue
        mismatched = [key for key in overrides if key.startswith(f"env.{section}.")]
        if mismatched:
            print(
                f"warning: env.kind={kind!r}, so these overrides have no effect "
                f"(use env.{kind}.* instead): {', '.join(mismatched)}",
                file=sys.stderr,
            )


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


def cmd_calibrate(args) -> int:
    """Check the game-specific settings against one real frame.

    Reads a screenshot, reports what the health-bar readers see, and writes an
    annotated image: HUD regions outlined, and every ommatidium's gaze marked,
    so it is obvious whether the eye is looking at the game or at the wall
    behind it.  Setting these numbers by guessing is the single most common way
    to spend an evening debugging a working pipeline.
    """
    from flyds1.envs.reward import BOSS_HP_REGION, PLAYER_HP_REGION, HealthBarReader
    from flyds1.envs.screen import ReplayCapture
    from flyds1.pipeline import build_network, make_front_end
    from flyds1.plotting import draw_box, filmstrip, mark_points, save_png, to_uint8

    cfg = _load_config(args)
    frame = ReplayCapture(args.frame, loop=False).grab()
    print(f"frame: {frame.shape[1]}x{frame.shape[0]} px")

    regions = {"player hp": PLAYER_HP_REGION, "boss hp": BOSS_HP_REGION}
    annotated = to_uint8(np.asarray(frame, dtype=float) / 255.0)
    for name, region in regions.items():
        value = HealthBarReader(region).read(frame)
        print(f"  {name:10s} {value:5.0%} filled  region={region}")
        annotated = draw_box(annotated, (region.x0, region.y0, region.x1, region.y1))

    cfg = cfg.apply_overrides(
        {"env.kind": "game", "env.game.frame_width": int(frame.shape[1]),
         "env.game.frame_height": int(frame.shape[0])}
    )
    net = build_network(cfg)
    front_end = make_front_end(cfg, net)
    centres = np.concatenate(
        [eye["sampler"].facet_pixel_centres() for eye in front_end.eyes.values()]
    )
    annotated = mark_points(annotated, centres)
    print(front_end.describe())

    out = Path(args.out or "calibration.png")
    save_png(filmstrip([annotated]), out)
    print(f"\nwrote {out}: red boxes are the HUD regions the reward reads,")
    print("green dots are where the fly's ommatidia look.")
    if front_end.screen_coverage() < 0.9:
        print("the dots do not span the frame -- raise the connectome's rings or "
              "lower frontend.screen.fov_h_deg before training on this")
    return 0


def cmd_watch(args) -> int:
    """Run the brain over recorded footage and report what the DNs do.

    No game, no key presses, no learning: just the perception path on real
    frames, so you can see whether anything reaches the motor output before
    committing hours of real-time play to it.
    """
    from flyds1.envs.screen import ReplayCapture
    from flyds1.net.reference import RateNetwork
    from flyds1.pipeline import build_network, make_front_end
    from flyds1.plotting import bar_chart, facet_image, filmstrip, save_png, to_uint8
    from flyds1.vision.encoder import build_encoder_wiring, encode_numpy

    cfg = _load_config(args)
    source = ReplayCapture(args.frames, loop=False)
    first = source.grab()
    source.reset()
    cfg = cfg.apply_overrides(
        {"env.kind": "game", "env.game.frame_width": int(first.shape[1]),
         "env.game.frame_height": int(first.shape[0])}
    )
    net = build_network(cfg)
    front_end = make_front_end(cfg, net)
    wiring = build_encoder_wiring(net, front_end.layout, inject_motion=cfg.encoder.inject_motion)
    sim = RateNetwork(net, cfg.rate)
    dt = 1.0 / cfg.env.game.target_fps

    print(front_end.describe())
    print(f"reading {len(source)} frames at dt={dt * 1000:.0f} ms\n")

    dn_history, obs_history, frames = [], [], []
    for _ in range(len(source)):
        frame = source.grab()
        obs = front_end.process(frame, dt)
        sim.step(encode_numpy(obs, front_end.layout, wiring))
        dn_history.append(sim.outputs()[0].copy())
        obs_history.append(obs)
        frames.append(frame)

    dn = np.stack(dn_history)
    obs = np.stack(obs_history)
    settle = min(len(dn) - 1, 10)
    print(f"descending neurons: {dn.shape[1]}")
    print(f"  mean rate            {dn[settle:].mean():.3f}")
    print(f"  variation over time  {dn[settle:].std(axis=0).mean():.4f}")
    print(f"  most active          {np.argsort(-dn[settle:].mean(axis=0))[:5].tolist()}")
    print(f"  retinal input varies {obs[settle:, front_end.layout.photoreceptor_slice].std():.4f}")
    if dn[settle:].std(axis=0).mean() < 1e-3:
        print("\n  the descending population barely moves on this footage: nothing")
        print("  downstream can act on it (see docs/results.md, 'how far the signal gets')")

    if args.png:
        out_dir = Path(args.png)
        side = sorted(front_end.eyes)[-1]
        eye = front_end.eyes[side]
        panels_written = []
        for k in np.linspace(settle, len(frames) - 1, min(4, len(frames) - settle)).astype(int):
            photo, _ = front_end.layout.split(obs[k])
            strip = filmstrip([
                to_uint8(np.asarray(frames[k], dtype=float) / 255.0),
                facet_image(eye["coords"], photo[eye["slots"]], signed=True),
                bar_chart(dn[k] - dn[settle:].mean(axis=0)),
            ])
            panels_written.append(save_png(strip, out_dir / f"watch_{k:04d}.png"))
        print(f"\nwrote {len(panels_written)} panels to {out_dir} "
              "(frame | what the eye sees | descending activity)")
    return 0


def _find_wrapper(env, name: str):
    """Walk a Gymnasium wrapper chain looking for a class by name."""
    node = env
    seen = 0
    while node is not None and seen < 20:
        if type(node).__name__ == name:
            return node
        node = getattr(node, "env", None)
        seen += 1
    return None


def cmd_live(args) -> int:
    """Run the agent and stream what it sees to a local web page.

    Read-only: the viewer never touches the environment. Uses the wrapper brain
    location so the network's state is continuous and its descending rates are
    the observation -- which is what makes them watchable.
    """
    from flyds1.live import LiveView, format_stats
    from flyds1.pipeline import action_spec_for, build_network, make_env

    cfg = _load_config(args)
    if cfg.env.brain_location != "wrapper":
        print("live view runs the brain in the environment (brain_location=wrapper) "
              "so its state is continuous; switching for this run")
        cfg = cfg.apply_overrides({"env.brain_location": "wrapper"})

    net = build_network(cfg)
    env, _ = make_env(cfg, net, seed=args.seed)
    retina = _find_wrapper(env, "RetinaWrapper")
    brain = _find_wrapper(env, "FlyBrainWrapper")
    if retina is None or brain is None:  # pragma: no cover - defensive
        raise ValueError("could not find the retina/brain wrappers in the env chain")
    # envs that do not repeat actions (the arena) hand over a single frame, and
    # only if asked -- without this the viewer's first panel is black
    retina.keep_frame_in_info = True

    front_end = retina.front_end
    side = sorted(front_end.eyes)[-1]
    eye = front_end.eyes[side]

    view = LiveView(Path(args.out or "live"), refresh_ms=args.refresh)
    url = view.serve(args.port)
    print(f"open {url} -- panels: game | what the eye sees | motion | descending neurons")

    policy = None
    if args.model:
        from stable_baselines3 import PPO

        policy = PPO.load(args.model, device=cfg.training.device)
    spec = action_spec_for(cfg)
    rng = np.random.default_rng(args.seed)

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(
                options={"skip_macros": args.skip_macros} if cfg.env.kind == "boss" else None
            )
            total, steps = 0.0, 0
            while True:
                if policy is not None:
                    action, _ = policy.predict(obs, deterministic=args.deterministic)
                else:
                    action = (rng.random(spec.size) < 0.25).astype(np.int8)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                steps += 1

                frames = info.get("frames")
                frame = frames[-1] if frames else info.get("frame")
                if frame is None:  # pragma: no cover - defensive
                    frame = np.zeros((cfg.frontend.screen.height, cfg.frontend.screen.width))
                # The retina already processed this frame inside the wrapper;
                # re-running it here would double-count the adaptation state and
                # show a picture the brain never saw. Read its last output back.
                photo, motion = front_end.layout.split(np.asarray(retina.last_observation))
                horizontal = None
                if front_end.layout.motion_mode == "retinotopic":
                    horizontal = (motion[eye["slots"]][:, 0] - motion[eye["slots"]][:, 1])
                view.update(
                    frame,
                    eye_coords=eye["coords"],
                    photoreceptors=photo[eye["slots"]],
                    motion=horizontal,
                    descending=brain.sim.outputs()[0],
                    stats=format_stats(
                        info,
                        {"episode": episode, "return": total, "steps": steps,
                         "keys": len([k for k in np.atleast_1d(action) if k > 0])},
                    ),
                )
                if terminated or truncated:
                    break
            print(f"episode {episode}: return {total:+.2f} over {steps} decisions")
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nstopped")
    finally:
        view.stop()
        env.close()
    return 0


def cmd_route(args) -> int:
    """Record or verify the run-back from the bonfire to the fog gate.

    The way back never changes, so this is route *execution*, not pathfinding:
    walk it once with ``--record`` (standing where the run-back starts), and
    every segment gets a reference frame that later runs are checked against.
    """
    import yaml

    from flyds1.envs.input_backends import make_input_backend
    from flyds1.envs.navigation import DEFAULT_ASYLUM_ROUTE, Route, Waypoint
    from flyds1.envs.screen import make_frame_source
    from flyds1.pipeline import action_spec_for

    cfg = _load_config(args)
    boss = cfg.env.boss
    source = make_frame_source(
        args.source or boss.frame_source, boss.capture, path=boss.replay_path
    )
    backend = make_input_backend("dry" if args.dry_run else boss.input_backend,
                                 window=boss.window_name)
    spec = action_spec_for(cfg)
    bindings = {b.name: b.key for b in spec.buttons}

    if args.steps:
        steps_path = Path(args.steps)
        if not steps_path.exists():
            example = Path(__file__).resolve().parents[2] / "configs" / "route_asylum.yaml"
            hint = f" (the repository ships one at {example})" if example.exists() else ""
            raise FileNotFoundError(
                f"no waypoint file at {steps_path}{hint}. "
                "Omit --steps entirely to record the built-in placeholder route instead."
            )
        data = yaml.safe_load(steps_path.read_text())
        route = Route(
            waypoints=tuple(
                Waypoint(
                    buttons=tuple(entry.get("buttons", [])),
                    seconds=float(entry.get("seconds", 1.0)),
                    name=str(entry.get("name", "")),
                    retries=int(entry.get("retries", 2)),
                )
                for entry in data["waypoints"]
            ),
            name=str(data.get("name", "route")),
        )
    elif args.verify:
        route = Route.load(args.verify)
    else:
        route = DEFAULT_ASYLUM_ROUTE
        print("no --steps given, using the placeholder Undead Asylum route:")

    print(route.describe())
    if args.dry_run:
        print("dry run: no keys are sent, so the character will not actually move")

    if args.record:
        recorded = route.record(
            backend, bindings, source.grab, fps=boss.target_fps, threshold=args.threshold
        )
        path = recorded.save(args.record)
        print(f"\nrecorded {len(recorded.waypoints)} checkpoints -> {path}")
        print("point the boss env at it with --set env.boss.route_path=" + str(path))
        return 0

    result = route.run(backend, bindings, source.grab, fps=boss.target_fps)
    print("\n" + result.describe())
    for index, score in enumerate(result.scores):
        print(f"  waypoint {index} ({route.waypoints[index].name or '-'}): match {score:+.2f}")
    return 0 if result.completed else 1


def cmd_budget(args) -> int:
    """Print the real-time arithmetic of a boss-fight run.

    The first number to look at in a project whose environment runs at wall
    clock speed, and the one most likely to end an experiment before it starts.
    """
    from flyds1.envs.boss import BossFightEnv

    cfg = _load_config(args)
    env = BossFightEnv(cfg.env.boss, sleep_fn=lambda _s: None)
    budget = env.budget(hours=1.0)
    print("boss fight, real-time budget")
    print(f"  action_repeat {cfg.env.boss.action_repeat} at {cfg.env.boss.target_fps:.0f} fps"
          f" -> {budget['seconds_per_decision'] * 1000:.0f} ms per decision")
    print(f"  attempt (max {cfg.env.boss.max_fight_steps} decisions) plus run-back"
          f" -> {budget['seconds_per_attempt_max']:.0f} s")
    print(f"  {budget['attempts_per_hour']:.0f} attempts/hour,"
          f" {budget['decisions_per_hour']:.0f} decisions/hour")
    print(f"  1M decisions would take {budget['hours_for_1M_decisions']:.0f} hours"
          f" ({budget['hours_for_1M_decisions'] / 24:.1f} days) of real play")
    print("\nPublished single-boss RL runs use millions of steps. If that number is")
    print("uncomfortable, the levers are action_repeat, max_fight_steps and the")
    print("run-back macro -- not patience.")
    env.close()
    return 0


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
    if "kills" in info:
        kills, attempts = int(info["kills"]), int(info.get("attempts", args.episodes))
        print(f"boss: {kills} kill(s) in {attempts} attempts")
        if policy is not None:
            print("run the same number of attempts without --model for the random-action "
                  "baseline; a kill only means something above it")
    return 0


# ---------------------------------------------------------------------------
def _common_options() -> argparse.ArgumentParser:
    """``--config`` and ``--set``, accepted on either side of the subcommand.

    argparse puts global options before the subcommand, which is not how anyone
    types them: ``flyds1 live --set x=1`` is the natural form and used to fail
    with "unrecognized arguments".  Attaching the same options to every
    subparser with SUPPRESS defaults makes both orders work: put every
    ``--set`` on whichever side reads naturally.  Mixing sides in one command
    does not merge -- argparse's subparser dispatch re-parses into a fresh
    namespace and copies it wholesale over the top-level one, so ``--set``
    given after the subcommand replaces (not adds to) any given before it.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", type=Path, default=argparse.SUPPRESS,
        help="experiment YAML (defaults to built-in values)",
    )
    common.add_argument(
        "--set", action="append", metavar="KEY=VALUE", default=argparse.SUPPRESS,
        help="override a config value, e.g. --set connectome.spec=synthetic:rings=8",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="flyds1", description=__doc__.splitlines()[0], parents=[common]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub_kwargs = {"parents": [common]}

    sub.add_parser(
        "doctor", help="report which optional dependencies are installed", **sub_kwargs
    ).set_defaults(func=cmd_doctor)
    sub.add_parser(
        "fetch", help="print how to download a real connectome", **sub_kwargs
    ).set_defaults(func=cmd_fetch)
    p_build = sub.add_parser("build", help="build the wired network", **sub_kwargs)
    p_build.add_argument("--out", type=Path, help="save the network as .npz")
    p_build.set_defaults(func=cmd_build)

    sub.add_parser(
        "info", help="summarise network, retina and encoder", **sub_kwargs
    ).set_defaults(func=cmd_info)

    p_self = sub.add_parser(
        "selftest", help="run every pipeline step once with checks", **sub_kwargs
    )
    p_self.add_argument("--png", type=Path, help="also write diagnostic images to this directory")
    p_self.set_defaults(func=cmd_selftest)

    p_cal = sub.add_parser("calibrate", help="check HUD regions and field of view on a frame", **sub_kwargs)
    p_cal.add_argument("--frame", required=True, help="a screenshot (png/jpg) or .npy frame")
    p_cal.add_argument("--out", help="annotated image to write (default calibration.png)")
    p_cal.set_defaults(func=cmd_calibrate)

    p_watch = sub.add_parser("watch", help="run the brain over recorded footage", **sub_kwargs)
    p_watch.add_argument("--frames", required=True, help="folder of frames or .npy stack")
    p_watch.add_argument("--png", help="write diagnostic panels to this directory")
    p_watch.set_defaults(func=cmd_watch)

    p_route = sub.add_parser(
        "route", help="record or verify the run-back route", **sub_kwargs
    )
    p_route.add_argument("--record", help="walk the route and save checkpoints to this .npz")
    p_route.add_argument("--verify", help="load this route and walk it, checking every checkpoint")
    p_route.add_argument("--steps", help="YAML describing the waypoints to record")
    p_route.add_argument("--source", help="frame source override, e.g. mss or replay")
    p_route.add_argument("--dry-run", action="store_true", help="send no input")
    p_route.add_argument("--threshold", type=float, default=0.55,
                         help="how close a checkpoint has to match (default 0.55)")
    p_route.set_defaults(func=cmd_route)

    p_live = sub.add_parser("live", help="watch the agent play in a browser", **sub_kwargs)
    p_live.add_argument("--model", help="trained model .zip (random actions if omitted)")
    p_live.add_argument("--port", type=int, default=8000)
    p_live.add_argument("--out", help="directory for the viewer files (default ./live)")
    p_live.add_argument("--refresh", type=int, default=200, help="page refresh interval, ms")
    p_live.add_argument("--episodes", type=int, default=1)
    p_live.add_argument("--seed", type=int, default=0)
    p_live.add_argument("--deterministic", action="store_true")
    p_live.add_argument("--skip-macros", action="store_true",
                        help="boss env: do not play the run-back macro")
    p_live.set_defaults(func=cmd_live)

    sub.add_parser(
        "budget", help="real-time arithmetic of a boss-fight run", **sub_kwargs
    ).set_defaults(func=cmd_budget)

    p_tune = sub.add_parser("tune", help="sweep the recurrent gain", **sub_kwargs)
    p_tune.add_argument("--gains", type=float, nargs="+", help="gains to try")
    p_tune.set_defaults(func=cmd_tune)

    p_train = sub.add_parser("train", help="train with PPO", **sub_kwargs)
    p_train.add_argument("--timesteps", type=int, help="override training.total_timesteps")
    p_train.set_defaults(func=cmd_train)

    p_play = sub.add_parser("play", help="roll out a policy", **sub_kwargs)
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
