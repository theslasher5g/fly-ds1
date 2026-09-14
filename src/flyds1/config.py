"""YAML configuration for a whole experiment.

Every stage of the pipeline already has its own dataclass; this module nests
them into one :class:`ExperimentConfig` and fills it from YAML, so a run is
reproducible from a single file.  Unknown keys are an error rather than a silent
no-op -- a typo in a config is the most expensive kind of bug in an RL project.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin

import yaml

from flyds1.envs.arena import ArenaConfig
from flyds1.envs.game import GameConfig
from flyds1.net.dynamics import RateConfig
from flyds1.net.rate_rnn import TrainableParts
from flyds1.vision.frontend import FrontEndConfig


@dataclass
class ConnectomeSection:
    """Step 1 + 2a: where the graph comes from and how it is scaled."""

    spec: str = "synthetic:rings=6"
    min_synapse_count: float = 5.0
    normalise: str = "in_degree"
    #: Recurrent gain.  Not cosmetic: at gain 1.0 the in-degree-normalised
    #: synthetic connectome loses ~99% of its stimulus-driven variance between
    #: the projection neurons and the descending neurons, and the policy sees a
    #: constant observation.  3.0 puts the network near (but below) criticality
    #: -- spectral radius ~0.9 -- where signals survive the depth of the
    #: hierarchy.  Re-run ``flyds1 tune`` after changing the connectome.
    gain: float = 3.0
    drop_modulatory: bool = True
    keep_largest_component: bool = False
    cache: str | None = None       # optional .npz to save/load the built network


@dataclass
class EncoderSection:
    """Step 3d: observation -> drive."""

    inject_motion: bool = True
    per_facet_gain: bool = True
    init_photo_gain: float = 1.0
    init_motion_gain: float = 1.0


@dataclass
class EnvSection:
    """Step 4: which environment, and how the brain is attached."""

    kind: str = "arena"                 # "arena" | "game"
    #: ``"policy"`` = brain inside the policy (encoder trainable, k-frame memory)
    #: ``"wrapper"`` = brain inside the env (episode-long memory, frozen encoder)
    brain_location: str = "policy"
    stack_k: int = 6
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    game: GameConfig = field(default_factory=GameConfig)


@dataclass
class TrainingSection:
    """Step 6: PPO hyper-parameters."""

    total_timesteps: int = 200_000
    n_envs: int = 4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    out_dir: str = "runs/fly"
    save_every: int = 0                 # 0 = only at the end
    value_net_arch: tuple[int, ...] = (64, 64)
    device: str = "auto"
    #: Wrap the vec env in VecNormalize for the reward only.  The arena's return
    #: is dominated by rare large events (death), and an unnormalised value
    #: target left ``explained_variance`` negative -- i.e. the critic was worse
    #: than predicting the mean, so every advantage was noise.
    normalize_reward: bool = True


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one run."""

    connectome: ConnectomeSection = field(default_factory=ConnectomeSection)
    rate: RateConfig = field(default_factory=RateConfig)
    frontend: FrontEndConfig = field(default_factory=FrontEndConfig)
    encoder: EncoderSection = field(default_factory=EncoderSection)
    trainable: TrainableParts = field(default_factory=TrainableParts)
    env: EnvSection = field(default_factory=EnvSection)
    training: TrainingSection = field(default_factory=TrainingSection)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{path} must contain a YAML mapping")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        return _build(cls, data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def dump_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    def apply_overrides(self, overrides: Mapping[str, Any]) -> "ExperimentConfig":
        """Apply ``{"training.seed": 3}`` style overrides, returning a new config."""
        data = self.to_dict()
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                if part not in node:
                    raise KeyError(f"unknown config section {part!r} in {dotted!r}")
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f"unknown config key {dotted!r}")
            node[parts[-1]] = value
        return ExperimentConfig.from_mapping(data)


def _build(cls, data: Mapping[str, Any]):
    """Recursively instantiate nested dataclasses from a mapping."""
    if not is_dataclass(cls):
        return data
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(
            f"{cls.__name__} has no field(s) {sorted(unknown)}; "
            f"valid keys: {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        target = known[name].type
        target = _resolve_type(target, cls)
        if is_dataclass(target) and isinstance(value, Mapping):
            kwargs[name] = _build(target, value)
        elif get_origin(target) is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _resolve_type(annotation, owner):
    """Turn a (possibly string) annotation into a class where we can."""
    if isinstance(annotation, str):
        import sys

        module = sys.modules.get(owner.__module__)
        candidates = {**vars(module or object), **globals()}
        base = annotation.split("|")[0].strip().split("[")[0].strip()
        return candidates.get(base, annotation)
    args = get_args(annotation)
    if args and get_origin(annotation) is not tuple:
        for arg in args:
            if is_dataclass(arg):
                return arg
    return annotation


__all__ = [
    "ConnectomeSection",
    "EncoderSection",
    "EnvSection",
    "ExperimentConfig",
    "TrainingSection",
]
