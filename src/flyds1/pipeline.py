"""Assembling the six steps into one runnable thing.

Everything here takes an :class:`~flyds1.config.ExperimentConfig` and returns
objects from the step modules; no new behaviour, just wiring, so the CLI and any
notebook agree on how a run is put together.
"""

from __future__ import annotations

from pathlib import Path

from flyds1.config import ExperimentConfig
from flyds1.connectome.graph import WiredNetwork, build_wired_network
from flyds1.connectome.loader import load_connectome
from flyds1.vision.frontend import ObsLayout, RetinaFrontEnd


def build_network(cfg: ExperimentConfig, *, verbose: bool = False) -> WiredNetwork:
    """Step 1 + 2a, with an optional ``.npz`` cache."""
    section = cfg.connectome
    cache = Path(section.cache) if section.cache else None
    if cache and cache.exists():
        net = WiredNetwork.load(cache)
        if verbose:
            print(f"loaded cached network from {cache}")
        return net
    connectome = load_connectome(section.spec, spacing_deg=cfg.frontend.spacing_deg)
    if verbose:
        print(connectome.summary())
    net = build_wired_network(
        connectome,
        min_synapse_count=section.min_synapse_count,
        normalise=section.normalise,  # type: ignore[arg-type]
        gain=section.gain,
        drop_modulatory=section.drop_modulatory,
        keep_largest_component=section.keep_largest_component,
    )
    if cache:
        net.save(cache)
        if verbose:
            print(f"cached network to {cache}")
    return net


def make_frame_env(cfg: ExperimentConfig, *, seed: int | None = None):
    """Step 4: the environment that produces frames."""
    if cfg.env.kind == "arena":
        from flyds1.envs.arena import FlyArenaEnv

        arena = cfg.env.arena
        if seed is not None:
            arena = type(arena)(**{**arena.__dict__, "seed": int(seed)})
        return FlyArenaEnv(arena)
    if cfg.env.kind == "game":
        from flyds1.envs.game import ScreenGameEnv

        return ScreenGameEnv(cfg.env.game)
    raise ValueError(f"unknown env kind {cfg.env.kind!r} (expected 'arena' or 'game')")


def make_env(cfg: ExperimentConfig, network: WiredNetwork, *, seed: int | None = None):
    """Full observation chain: frames -> retina -> (optionally) brain -> stack."""
    from flyds1.envs.wrappers import RetinaWrapper

    frame_env = make_frame_env(cfg, seed=seed)
    env = RetinaWrapper(frame_env, network, cfg.frontend)
    layout: ObsLayout = env.layout

    if cfg.env.brain_location == "wrapper":
        from flyds1.agent.brain_wrapper import FlyBrainWrapper

        env = FlyBrainWrapper(
            env,
            network,
            layout,
            rate_config=cfg.rate,
            photo_gain=cfg.encoder.init_photo_gain,
            motion_gain=cfg.encoder.init_motion_gain,
            inject_motion=cfg.encoder.inject_motion,
        )
        return env, layout

    if cfg.env.brain_location != "policy":
        raise ValueError(
            f"unknown brain_location {cfg.env.brain_location!r} (expected 'policy' or 'wrapper')"
        )

    from flyds1.agent.stacking import ObsStackWrapper

    return ObsStackWrapper(env, k=cfg.env.stack_k), layout


def make_front_end(cfg: ExperimentConfig, network: WiredNetwork) -> RetinaFrontEnd:
    """A standalone retina, for diagnostics that do not need an environment."""
    from flyds1.vision.frontend import FrontEndConfig

    frontend = cfg.frontend
    if cfg.env.kind == "arena":
        screen = type(frontend.screen)(
            width=cfg.env.arena.width,
            height=cfg.env.arena.height,
            fov_h_deg=frontend.screen.fov_h_deg,
        )
    else:
        screen = type(frontend.screen)(
            width=cfg.env.game.frame_width,
            height=cfg.env.game.frame_height,
            fov_h_deg=frontend.screen.fov_h_deg,
        )
    adjusted = FrontEndConfig(
        screen=screen,
        acceptance_fwhm_deg=frontend.acceptance_fwhm_deg,
        eye_azimuth_offset_deg=frontend.eye_azimuth_offset_deg,
        spacing_deg=frontend.spacing_deg,
        reichardt=frontend.reichardt,
        motion_mode=frontend.motion_mode,
        adapt_contrast=frontend.adapt_contrast,
        contrast_tau=frontend.contrast_tau,
        contrast_eps=frontend.contrast_eps,
    )
    return RetinaFrontEnd(network, adjusted)


def make_model(cfg: ExperimentConfig, env, network: WiredNetwork, layout: ObsLayout):
    """Step 6: a PPO model wired to the fly brain."""
    from stable_baselines3 import PPO

    train = cfg.training
    kwargs: dict = {}
    if cfg.env.brain_location == "policy":
        from flyds1.agent.fly_policy import fly_policy_kwargs

        kwargs = fly_policy_kwargs(
            network,
            layout,
            rate_config=cfg.rate,
            trainable=cfg.trainable,
            inject_motion=cfg.encoder.inject_motion,
            value_net_arch=list(train.value_net_arch),
        )
    else:
        # the brain runs in the env; the policy is the linear decoder plus value net
        kwargs = {"net_arch": {"pi": [], "vf": list(train.value_net_arch)}}

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=train.learning_rate,
        n_steps=train.n_steps,
        batch_size=train.batch_size,
        n_epochs=train.n_epochs,
        gamma=train.gamma,
        gae_lambda=train.gae_lambda,
        clip_range=train.clip_range,
        ent_coef=train.ent_coef,
        vf_coef=train.vf_coef,
        max_grad_norm=train.max_grad_norm,
        seed=train.seed,
        device=train.device,
        policy_kwargs=kwargs,
        verbose=1,
    )


def action_spec_for(cfg: ExperimentConfig):
    from flyds1.motor.actions import ARENA_ACTIONS, DARKSOULS_ACTIONS

    return ARENA_ACTIONS if cfg.env.kind == "arena" else DARKSOULS_ACTIONS


__all__ = [
    "action_spec_for",
    "build_network",
    "make_env",
    "make_frame_env",
    "make_front_end",
    "make_model",
]
