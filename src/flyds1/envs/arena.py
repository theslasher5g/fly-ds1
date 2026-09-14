"""Step 4a: a synthetic first-person arena.

Dark Souls cannot run in CI, on a headless box, or 300x faster than real time --
and until the rest of the pipeline is proven, pointing it at a real game only
adds ways to be wrong.  This is a deliberately small stand-in with the
properties the fly model actually cares about:

* a first-person view with **texture**, so turning produces optic flow (a
  flat-shaded world would leave the motion detectors with nothing to correlate);
* a hazard that approaches and does damage, so looming matters;
* a reward built from distance, survival and damage -- the same three terms the
  roadmap proposes for the real game.

Two tasks share that world, selected by ``ArenaConfig.task``:

``"survive"``
    Outrun the chaser. Closest to the game, and the harder benchmark: measured
    here, even a scripted oracle with ground-truth state still loses ~40 hp per
    episode, so most of the return is decided by spawn geometry rather than by
    the policy. Good as a final test, poor as a first one.

``"target"``
    Steer to a bright marker, which relocates each time it is reached. This is
    fly behaviour in the literal sense -- fixation on a visual target is what
    the T4/T5 and lobula-plate machinery in the connectome is *for* -- and the
    return spans a wide, policy-controlled range: a policy that cannot see the
    target reaches roughly none, one that can reaches a dozen.

Rendering is a grid raycaster: one ray per screen column, DDA through a tile
map, plus a billboard for the enemy.  No dependencies beyond numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyds1.motor.actions import ARENA_ACTIONS, ActionSpec

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _GYM_AVAILABLE = False
    gym = None  # type: ignore[assignment]

    class _Stub:
        class Env:
            pass

    gym = _Stub()  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


#: Open arena: border walls only.  The target task uses this, because a
#: fixation task must test the see-it-and-steer loop, not path-finding -- in the
#: pillared map a target across the room is usually behind a wall, and a
#: straight-line policy spends the episode scraping along pillars.
OPEN_MAP = [
    "######################",
    "#....................#",
    "#....................#",
    "#....................#",
    "#....................#",
    "#....................#",
    "#....................#",
    "#....................#",
    "#....................#",
    "#....................#",
    "#....................#",
    "#....................#",
    "######################",
]

DEFAULT_MAP = [
    "################",
    "#..............#",
    "#..##......##..#",
    "#..##......##..#",
    "#..............#",
    "#....######....#",
    "#....#....#....#",
    "#....#....#....#",
    "#....######....#",
    "#..............#",
    "#..##......##..#",
    "#..##......##..#",
    "#..............#",
    "################",
]


@dataclass
class ArenaConfig:
    """Arena, renderer and reward parameters."""

    width: int = 320
    height: int = 180
    fov_h_deg: float = 90.0
    #: ``None`` picks the map that suits the task: the pillared arena for
    #: "survive" (somewhere to break line of sight) and the open one for
    #: "target" (line of sight is the point).
    tile_map: tuple[str, ...] | None = None
    dt: float = 1.0 / 30.0
    max_steps: int = 900

    #: ``"survive"`` (outrun the chaser) or ``"target"`` (steer to the marker).
    task: str = "target"

    move_speed: float = 2.5        # tiles / s
    turn_speed: float = 2.4        # rad / s
    enemy_speed: float = 1.6
    enemy_damage: float = 35.0     # hp / s while in contact
    contact_range: float = 0.9
    start_hp: float = 100.0

    # Reward terms.  The weights matter more than they look: with progress
    # weighted at 1.0 per tile, circling on the spot earns ~26 per episode while
    # dying costs 5, so the best policy is a constant motor command and the task
    # can be solved without looking at anything -- which makes it useless for
    # training a visual model.  Survival and damage therefore dominate, and
    # since the enemy is slower than the player (1.6 vs 2.5 tiles/s), surviving
    # means seeing it coming and moving away.
    reward_progress: float = 0.05      # per tile moved
    reward_survival: float = 0.02      # per step alive
    reward_damage: float = 0.2         # per hp lost
    reward_death: float = 10.0
    reward_wall_bump: float = 0.05

    # target task
    #: Wall brightness multiplier.  The target task dims the walls: a fly
    #: separates a small figure from a textured background with dedicated
    #: small-object detectors (LC11/LC18), which a connectome stand-in without
    #: them does not have, and this benchmark is meant to test see-and-steer,
    #: not figure-ground segregation.  Walls stay textured, so optic flow
    #: survives -- they are just no longer brighter than the thing to steer at.
    wall_brightness: float = 1.0
    target_radius: float = 1.5         # how close counts as reached
    reward_target: float = 5.0         # per target reached
    reward_approach: float = 1.0       # per tile of distance closed
    target_enemy: bool = False         # keep the chaser around in target mode

    seed: int = 0

    def __post_init__(self) -> None:
        if self.task not in ("survive", "target"):
            raise ValueError(f"task must be 'survive' or 'target', got {self.task!r}")
        if self.tile_map is None:
            self.tile_map = tuple(OPEN_MAP if self.task == "target" else DEFAULT_MAP)
            if self.task == "target":
                self.wall_brightness = min(self.wall_brightness, 0.35)
        else:
            self.tile_map = tuple(self.tile_map)


class FlyArenaEnv(gym.Env):
    """First-person arena whose observation is a greyscale frame in [0, 1]."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: ArenaConfig | None = None, action_spec: ActionSpec | None = None):
        if not _GYM_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs gymnasium: pip install -e '.[rl]'")
        super().__init__()
        self.cfg = config or ArenaConfig()
        self.spec_actions = action_spec or ARENA_ACTIONS
        self.grid = np.array([[c != "." for c in row] for row in self.cfg.tile_map], dtype=bool)
        self.map_h, self.map_w = self.grid.shape
        self.focal = (self.cfg.width / 2.0) / np.tan(np.deg2rad(self.cfg.fov_h_deg) / 2.0)
        self._column_angles = np.arctan(
            (np.arange(self.cfg.width) + 0.5 - self.cfg.width / 2.0) / self.focal
        )
        # MultiBinary for held buttons -- see flyds1.motor.actions for why a
        # thresholded Box makes the policy indistinguishable from noise.
        self.action_space = self.spec_actions.gym_space()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.cfg.height, self.cfg.width), dtype=np.float32
        )
        self.reset(seed=self.cfg.seed)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # seeds self.np_random, which Gymnasium requires an env to use
        super().reset(seed=seed)
        self.pos = np.zeros(2)
        self.pos = self._free_cell()
        self.angle = float(self.np_random.uniform(0, 2 * np.pi))
        self.enemy = self._free_cell(min_distance=4.0)
        self.target = self._free_cell(min_distance=3.0)
        self.hp = self.cfg.start_hp
        self.steps = 0
        self.distance_travelled = 0.0
        self.targets_reached = 0
        self._target_distance = float(np.linalg.norm(self.target - self.pos))
        return self._observe(), self._info()

    def _free_cell(self, min_distance: float = 0.0) -> np.ndarray:
        free = np.argwhere(~self.grid)
        for _ in range(500):
            cell = free[self.np_random.integers(len(free))]
            pos = cell[::-1].astype(float) + 0.5  # (x, y)
            if min_distance == 0.0 or np.linalg.norm(pos - self.pos) >= min_distance:
                return pos
        return free[0][::-1].astype(float) + 0.5  # pragma: no cover - tiny maps

    # ------------------------------------------------------------------
    def step(self, action):
        cfg = self.cfg
        held, _ = self.spec_actions.decode(action)
        turn = (1.0 if held.get("right") else 0.0) - (1.0 if held.get("left") else 0.0)
        walk = (1.0 if held.get("forward") else 0.0) - (1.0 if held.get("back") else 0.0)

        self.angle = (self.angle + turn * cfg.turn_speed * cfg.dt) % (2 * np.pi)
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        target = self.pos + direction * walk * cfg.move_speed * cfg.dt
        bumped = False
        moved = 0.0
        for axis in (0, 1):  # slide along walls instead of sticking to them
            candidate = self.pos.copy()
            candidate[axis] = target[axis]
            if self._is_free(candidate):
                moved += abs(candidate[axis] - self.pos[axis])
                self.pos = candidate
            else:
                bumped = True
        self.distance_travelled += moved

        # the chaser: always present in "survive", optional in "target"
        damage = 0.0
        if self._enemy_active():
            to_agent = self.pos - self.enemy
            dist = float(np.linalg.norm(to_agent))
            if dist > 1e-6:
                enemy_step = to_agent / dist * cfg.enemy_speed * cfg.dt
                for axis in (0, 1):
                    candidate = self.enemy.copy()
                    candidate[axis] += enemy_step[axis]
                    if self._is_free(candidate):
                        self.enemy = candidate
            hp_before = self.hp
            if dist < cfg.contact_range:
                self.hp -= cfg.enemy_damage * cfg.dt
            damage = hp_before - self.hp

        self.steps += 1
        terminated = self.hp <= 0.0
        truncated = self.steps >= cfg.max_steps

        reward = (
            cfg.reward_survival
            - cfg.reward_damage * damage
            - (cfg.reward_wall_bump if bumped else 0.0)
            - (cfg.reward_death if terminated else 0.0)
        )
        if cfg.task == "survive":
            reward += cfg.reward_progress * moved
        else:
            reward += self._target_step()
        return self._observe(), float(reward), bool(terminated), bool(truncated), self._info()

    def _enemy_active(self) -> bool:
        return self.cfg.task == "survive" or self.cfg.target_enemy

    def _target_step(self) -> float:
        """Reward for closing on the marker, and for reaching it."""
        cfg = self.cfg
        distance = float(np.linalg.norm(self.target - self.pos))
        # Shaping on the *change* in distance: a potential-based term, so it
        # rewards making progress rather than loitering near the marker.
        reward = cfg.reward_approach * (self._target_distance - distance)
        self._target_distance = distance
        if distance < cfg.target_radius:
            reward += cfg.reward_target
            self.targets_reached += 1
            self.target = self._free_cell(min_distance=4.0)
            self._target_distance = float(np.linalg.norm(self.target - self.pos))
        return reward

    def _is_free(self, pos: np.ndarray) -> bool:
        x, y = pos
        if not (0 <= x < self.map_w and 0 <= y < self.map_h):
            return False
        return not self.grid[int(y), int(x)]

    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        """Raycast the tile map into a greyscale frame."""
        cfg = self.cfg
        frame = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        angles = self.angle + self._column_angles
        dists, shades = self._cast(angles)
        # perpendicular distance keeps walls flat rather than fish-eyed
        perp = np.maximum(dists * np.cos(self._column_angles), 1e-3)
        heights = np.clip(cfg.height / perp, 2, cfg.height * 3)
        horizon = cfg.height / 2.0
        rows = np.arange(cfg.height)[:, None]
        top = horizon - heights / 2.0
        bottom = horizon + heights / 2.0
        wall_mask = (rows >= top) & (rows <= bottom)
        brightness = self.cfg.wall_brightness * shades / (1.0 + 0.25 * perp)
        frame[:] = np.where(wall_mask, brightness, 0.0)
        # floor gets a gradient so looking down is not featureless
        floor = np.clip((rows - horizon) / horizon, 0, 1) * 0.12
        frame[:] = np.where(wall_mask, frame, floor)
        if self._enemy_active():
            self._draw_billboard(frame, perp, self.enemy, size_scale=0.8, brightness=0.95)
        if self.cfg.task == "target":
            self._draw_billboard(frame, perp, self.target, size_scale=1.1, brightness=1.0,
                                 aspect=0.35)
        return np.clip(frame, 0.0, 1.0)

    def _cast(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised DDA: distance and surface shade per ray."""
        n = len(angles)
        dx, dy = np.cos(angles), np.sin(angles)
        map_x = np.full(n, int(self.pos[0]))
        map_y = np.full(n, int(self.pos[1]))
        delta_x = np.where(np.abs(dx) < 1e-9, 1e9, np.abs(1.0 / np.where(dx == 0, 1e-9, dx)))
        delta_y = np.where(np.abs(dy) < 1e-9, 1e9, np.abs(1.0 / np.where(dy == 0, 1e-9, dy)))
        step_x = np.where(dx < 0, -1, 1)
        step_y = np.where(dy < 0, -1, 1)
        side_x = np.where(
            dx < 0, (self.pos[0] - map_x) * delta_x, (map_x + 1.0 - self.pos[0]) * delta_x
        )
        side_y = np.where(
            dy < 0, (self.pos[1] - map_y) * delta_y, (map_y + 1.0 - self.pos[1]) * delta_y
        )
        hit = np.zeros(n, dtype=bool)
        vertical = np.zeros(n, dtype=bool)
        dist = np.full(n, float(max(self.map_w, self.map_h)))
        for _ in range(2 * (self.map_w + self.map_h)):
            if hit.all():
                break
            take_x = (side_x < side_y) & ~hit
            take_y = (~take_x) & ~hit
            map_x = np.where(take_x, map_x + step_x, map_x)
            side_x = np.where(take_x, side_x + delta_x, side_x)
            map_y = np.where(take_y, map_y + step_y, map_y)
            side_y = np.where(take_y, side_y + delta_y, side_y)
            vertical = np.where(take_x, True, np.where(take_y, False, vertical))
            inside = (map_x >= 0) & (map_x < self.map_w) & (map_y >= 0) & (map_y < self.map_h)
            solid = np.zeros(n, dtype=bool)
            solid[inside] = self.grid[map_y[inside], map_x[inside]]
            just_hit = (solid | ~inside) & ~hit
            dist = np.where(
                just_hit,
                np.where(vertical, side_x - delta_x, side_y - delta_y),
                dist,
            )
            hit |= just_hit
        # texture: brick pattern from the hit cell plus darker side faces
        shade = np.where(vertical, 0.95, 0.65)
        stripe = ((map_x + map_y) % 2 == 0).astype(float) * 0.15
        return np.maximum(dist, 1e-3), np.clip(shade - stripe, 0.05, 1.0)

    def _draw_billboard(
        self,
        frame: np.ndarray,
        wall_perp: np.ndarray,
        position: np.ndarray,
        *,
        size_scale: float = 0.8,
        brightness: float = 0.95,
        aspect: float = 1.0,
    ) -> None:
        rel = position - self.pos
        dist = float(np.linalg.norm(rel))
        if dist < 1e-3:
            return
        bearing = np.arctan2(rel[1], rel[0]) - self.angle
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        if abs(bearing) > np.deg2rad(self.cfg.fov_h_deg) / 2 + 0.3:
            return
        cx = self.cfg.width / 2.0 + np.tan(bearing) * self.focal
        size = np.clip(self.cfg.height * size_scale / dist, 3, self.cfg.height)
        width = max(2.0, size * aspect)
        x0, x1 = int(cx - width / 2), int(cx + width / 2)
        y0 = int(self.cfg.height / 2.0 - size / 2)
        y1 = int(self.cfg.height / 2.0 + size / 2)
        xs = np.arange(max(0, x0), min(self.cfg.width, x1 + 1))
        ys = np.arange(max(0, y0), min(self.cfg.height, y1 + 1))
        if len(xs) == 0 or len(ys) == 0:
            return
        visible = xs[dist < wall_perp[xs]]  # occluded by walls in front
        if len(visible) == 0:
            return
        # a bright silhouette: high contrast so looming is unmistakable
        frame[np.ix_(ys, visible)] = brightness

    # ------------------------------------------------------------------
    def _info(self) -> dict:
        return {
            "hp": float(self.hp),
            "position": self.pos.copy(),
            "angle": float(self.angle),
            "enemy_distance": float(np.linalg.norm(self.enemy - self.pos)),
            "target_distance": float(np.linalg.norm(self.target - self.pos)),
            "targets_reached": int(self.targets_reached),
            "distance_travelled": float(self.distance_travelled),
            "steps": int(self.steps),
        }

    def render(self):
        frame = self._observe()
        return (np.stack([frame] * 3, axis=-1) * 255).astype(np.uint8)


__all__ = ["ArenaConfig", "DEFAULT_MAP", "OPEN_MAP", "FlyArenaEnv"]
