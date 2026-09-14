"""Reward signals read off the screen.

For a real game there is no reward function, only pixels.  The three terms the
roadmap names -- distance covered, time survived, damage dealt to the boss --
are estimated here:

* **health bars.** Both the player's and the boss's HP are strips of a
  characteristic colour; the fraction of the strip still coloured is the HP
  fraction.  Region and colour test are config, because they differ per game,
  per resolution and per UI mod.
* **progress.** Without game telemetry, mean optic flow magnitude is the
  cheapest proxy for "the world moved past me".  It rewards moving and turning,
  and it is honest about what it is: a proxy, gameable by spinning in place,
  which is why it carries a small weight and is paired with survival.

Everything here is pure numpy on a frame, so it can be unit-tested against
synthetic frames -- see ``tests/test_reward.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BarRegion:
    """A rectangular UI region, in *fractions* of the frame (resolution-free)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x0, x1 = int(self.x0 * w), int(np.ceil(self.x1 * w))
        y0, y1 = int(self.y0 * h), int(np.ceil(self.y1 * h))
        x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
        y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
        return frame[y0:y1, x0:x1]


#: Dark Souls-ish defaults: player bars top-left, boss bar along the bottom.
PLAYER_HP_REGION = BarRegion(0.035, 0.045, 0.245, 0.062)
BOSS_HP_REGION = BarRegion(0.255, 0.885, 0.745, 0.900)


class HealthBarReader:
    """Reads a fraction-filled bar out of a frame.

    ``hue_test`` receives the ``(h, w, 3)`` crop scaled to [0, 1] and returns a
    boolean mask of "bar is still filled here".  The default test is "clearly
    more red than green and blue", which matches the red HP bars of the souls
    games; pass your own for a different UI.
    """

    def __init__(
        self,
        region: BarRegion,
        *,
        hue_test=None,
        min_fill_rows: float = 0.3,
        smoothing: float = 0.0,
    ) -> None:
        self.region = region
        self.hue_test = hue_test or self._default_red_test
        self.min_fill_rows = float(min_fill_rows)
        self.smoothing = float(smoothing)
        self._value: float | None = None
        #: Last *unsmoothed* reading.  Death has to be detected on this: a bar
        #: that drops to zero would otherwise take several frames to cross the
        #: threshold, and the death penalty would land on the wrong step.
        self.raw: float | None = None

    @staticmethod
    def _default_red_test(rgb: np.ndarray) -> np.ndarray:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return (r > 0.25) & (r > g * 1.8) & (r > b * 1.8)

    def read(self, frame_rgb: np.ndarray) -> float:
        """Fraction of the bar still filled, in [0, 1]."""
        arr = np.asarray(frame_rgb, dtype=float)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError("health bars need an RGB frame")
        if arr.max(initial=0.0) > 1.5:
            arr = arr / 255.0
        crop = self.region.crop(arr[..., :3])
        if crop.size == 0:
            return 0.0
        mask = self.hue_test(crop)
        # a column counts as filled if enough of its rows are coloured
        column_filled = mask.mean(axis=0) >= self.min_fill_rows
        value = float(column_filled.mean())
        self.raw = value
        if self.smoothing > 0.0 and self._value is not None:
            value = (1 - self.smoothing) * value + self.smoothing * self._value
        self._value = value
        return value

    def reset(self) -> None:
        self._value = None
        self.raw = None


@dataclass
class RewardConfig:
    """Weights of the reward terms."""

    survival: float = 0.01          # per step alive
    player_damage: float = 2.0      # per unit of HP fraction lost
    boss_damage: float = 10.0       # per unit of boss HP fraction removed
    progress: float = 0.5           # per unit of mean optic flow magnitude
    death: float = 5.0
    boss_kill: float = 20.0


class ScreenRewardEstimator:
    """Turns consecutive frames into a scalar reward."""

    def __init__(
        self,
        config: RewardConfig | None = None,
        *,
        player_bar: HealthBarReader | None = None,
        boss_bar: HealthBarReader | None = None,
    ) -> None:
        self.cfg = config or RewardConfig()
        self.player_bar = player_bar or HealthBarReader(PLAYER_HP_REGION, smoothing=0.3)
        self.boss_bar = boss_bar or HealthBarReader(BOSS_HP_REGION, smoothing=0.3)
        self.reset()

    def reset(self) -> None:
        self.player_bar.reset()
        self.boss_bar.reset()
        self._player_hp: float | None = None
        self._boss_hp: float | None = None

    def __call__(self, frame_rgb: np.ndarray, flow_magnitude: float = 0.0) -> tuple[float, dict]:
        """Returns ``(reward, info)``; ``info`` carries the raw readings."""
        cfg = self.cfg
        player = self.player_bar.read(frame_rgb)
        boss = self.boss_bar.read(frame_rgb)
        raw_player = float(self.player_bar.raw if self.player_bar.raw is not None else player)
        raw_boss = float(self.boss_bar.raw if self.boss_bar.raw is not None else boss)
        prev_player = player if self._player_hp is None else self._player_hp
        prev_boss = boss if self._boss_hp is None else self._boss_hp

        player_loss = max(0.0, prev_player - player)
        boss_loss = max(0.0, prev_boss - boss)
        died = raw_player <= 0.02 and prev_player > 0.02
        boss_dead = raw_boss <= 0.02 and prev_boss > 0.05

        reward = (
            cfg.survival
            + cfg.progress * float(flow_magnitude)
            - cfg.player_damage * player_loss
            + cfg.boss_damage * boss_loss
            - (cfg.death if died else 0.0)
            + (cfg.boss_kill if boss_dead else 0.0)
        )
        self._player_hp, self._boss_hp = player, boss
        info = {
            "player_hp": player,
            "boss_hp": boss,
            "player_hp_raw": raw_player,
            "boss_hp_raw": raw_boss,
            "player_damage": player_loss,
            "boss_damage": boss_loss,
            "died": died,
            "boss_dead": boss_dead,
            "flow": float(flow_magnitude),
        }
        return float(reward), info


def frame_flow_magnitude(previous: np.ndarray, current: np.ndarray) -> float:
    """Crude global motion estimate: mean absolute frame difference.

    Not optic flow in the Horn-Schunck sense; it is the cheapest signal that
    correlates with "something moved" and costs one subtraction per pixel.
    """
    a = np.asarray(previous, dtype=float)
    b = np.asarray(current, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"frame shapes differ: {a.shape} vs {b.shape}")
    return float(np.abs(b - a).mean())


__all__ = [
    "BOSS_HP_REGION",
    "BarRegion",
    "HealthBarReader",
    "PLAYER_HP_REGION",
    "RewardConfig",
    "ScreenRewardEstimator",
    "frame_flow_magnitude",
]
