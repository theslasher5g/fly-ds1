"""Reading the game's state off the screen.

A boss fight has structure a reward number cannot express: it starts when the
fog gate closes, it ends in one of two ways, and between attempts there is a
run-back the agent is not being scored on.  Driving that from pixels needs three
detectors, and all three are heuristics over a configured screen region -- which
means they are also things you can get wrong silently, so each one exposes the
raw measurement it used, and ``flyds1 calibrate`` prints them.

Tuned for the souls games' conventions (a boss health bar along the bottom, a
"YOU DIED" screen that is dark and red), but every threshold is configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyds1.envs.reward import BOSS_HP_REGION, PLAYER_HP_REGION, BarRegion, HealthBarReader

#: The "YOU DIED" screen fills the middle of the frame; sampling a band across
#: the centre avoids the HUD entirely.
DEATH_SCREEN_REGION = BarRegion(0.15, 0.40, 0.85, 0.60)


@dataclass
class DetectorConfig:
    """Regions and thresholds for the three fight-state detectors."""

    player_hp: BarRegion = field(default_factory=lambda: PLAYER_HP_REGION)
    boss_hp: BarRegion = field(default_factory=lambda: BOSS_HP_REGION)
    death_screen: BarRegion = field(default_factory=lambda: DEATH_SCREEN_REGION)

    #: A boss bar counts as present above this fill fraction.  Deliberately
    #: tiny: an almost-drained bar is still on screen, and treating it as gone
    #: loses exactly the frames that distinguish a kill from a retreat.
    boss_bar_present: float = 0.005
    #: Consecutive frames a state has to hold before it is believed.  Loading
    #: screens, cutscenes and hit flashes all produce one-frame lies.
    confirm_frames: int = 3

    #: A boss counts as killed when the *lowest* reading taken while its bar was
    #: still visible is below this.  A killing blow that empties the bar from
    #: well above this in a single frame would read as a retreat instead --
    #: check your own footage with ``flyds1 watch`` and raise it if so.
    boss_dead_below: float = 0.08

    #: "YOU DIED": dark frame, dominated by red.
    death_max_brightness: float = 0.22
    death_min_redness: float = 1.6

    def __post_init__(self) -> None:
        if self.boss_bar_present >= self.boss_dead_below:
            raise ValueError(
                f"boss_bar_present ({self.boss_bar_present}) must be below "
                f"boss_dead_below ({self.boss_dead_below}): the frames that prove a "
                "kill are the ones where the bar is nearly empty, and a presence "
                "threshold above the kill threshold throws exactly those away"
            )
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")


class DeathScreenDetector:
    """Detects the death screen: a dark frame whose red channel dominates."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self.brightness = 0.0
        self.redness = 0.0
        self._streak = 0

    def update(self, frame_rgb: np.ndarray) -> bool:
        arr = np.asarray(frame_rgb, dtype=float)
        if arr.max(initial=0.0) > 1.5:
            arr = arr / 255.0
        crop = self.cfg.death_screen.crop(arr[..., :3])
        if crop.size == 0:
            return False
        self.brightness = float(crop.mean())
        other = float(crop[..., 1:].mean()) + 1e-6
        self.redness = float(crop[..., 0].mean()) / other
        looks_dead = (
            self.brightness < self.cfg.death_max_brightness
            and self.redness > self.cfg.death_min_redness
        )
        self._streak = self._streak + 1 if looks_dead else 0
        return self._streak >= self.cfg.confirm_frames

    def reset(self) -> None:
        self._streak = 0


class FightStateDetector:
    """Is a boss fight running, and how are both sides doing?

    Wraps the two health-bar readers and adds hysteresis: the boss bar has to
    be present (or absent) for several frames before the state flips, because a
    single frame of a hit flash or a loading screen otherwise starts or ends
    fights that never happened.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self.player = HealthBarReader(self.cfg.player_hp, smoothing=0.2)
        self.boss = HealthBarReader(self.cfg.boss_hp, smoothing=0.2)
        self.death = DeathScreenDetector(self.cfg)
        self.reset()

    def reset(self) -> None:
        self.player.reset()
        self.boss.reset()
        self.death.reset()
        self.fight_active = False
        self._present_streak = 0
        self._absent_streak = 0
        self.player_hp = 1.0
        self.boss_hp = 1.0
        self.boss_hp_at_start = 1.0
        #: Lowest boss reading taken from a frame where the bar was actually
        #: visible.  Once the bar disappears the reader returns 0, so judging a
        #: kill by the current value would score "walked back out of the fog
        #: gate" as a victory -- and pay the kill bonus for running away.
        self.boss_hp_min_seen = 1.0

    def update(self, frame_rgb: np.ndarray) -> dict:
        """One frame in; a dict describing the fight state out."""
        self.player_hp = self.player.read(frame_rgb)
        self.boss_hp = self.boss.read(frame_rgb)
        raw_boss = float(self.boss.raw if self.boss.raw is not None else self.boss_hp)
        raw_player = float(self.player.raw if self.player.raw is not None else self.player_hp)
        died = self.death.update(frame_rgb)

        present = raw_boss > self.cfg.boss_bar_present
        self._present_streak = self._present_streak + 1 if present else 0
        self._absent_streak = 0 if present else self._absent_streak + 1
        if present:
            self.boss_hp_min_seen = min(self.boss_hp_min_seen, raw_boss)

        started = False
        ended_reason = None
        if not self.fight_active and self._present_streak >= self.cfg.confirm_frames:
            self.fight_active = True
            self.boss_hp_at_start = self.boss_hp
            self.boss_hp_min_seen = raw_boss
            started = True
        elif self.fight_active:
            if died or raw_player <= 0.02:
                self.fight_active = False
                ended_reason = "player_died"
            elif self._absent_streak >= self.cfg.confirm_frames:
                self.fight_active = False
                # Judge the outcome by the last reading taken while the bar was
                # still on screen: near zero means a kill, anything else means
                # the fight was left.
                ended_reason = (
                    "boss_killed" if self.boss_hp_min_seen <= self.cfg.boss_dead_below else "aborted"
                )

        return {
            "fight_active": self.fight_active,
            "fight_started": started,
            "fight_ended": ended_reason,
            "player_hp": self.player_hp,
            "boss_hp": self.boss_hp,
            "player_hp_raw": raw_player,
            "boss_hp_raw": raw_boss,
            "boss_hp_min_seen": self.boss_hp_min_seen,
            "death_screen": died,
            "death_brightness": self.death.brightness,
            "death_redness": self.death.redness,
        }


__all__ = [
    "DEATH_SCREEN_REGION",
    "DeathScreenDetector",
    "DetectorConfig",
    "FightStateDetector",
]
