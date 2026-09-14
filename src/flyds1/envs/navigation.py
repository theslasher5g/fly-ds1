"""Getting back to the fog gate without pathfinding.

The run-back from a bonfire to a boss is a *fixed* route: it never changes, it
has no branches, and nothing about it needs searching. What it needs is
reliable **execution** -- which is a different and much smaller problem than
navigation, and one that can be solved from pixels alone.

A :class:`Route` is a sequence of timed button holds, each optionally ending at
a visual **checkpoint**: a reference frame captured when the route was recorded.
After every segment the current frame is matched against that reference. If it
does not match, the route tries to unstick itself and replays the segment; if
that fails too, the attempt is abandoned as "lost" rather than blundering on --
a run-back that silently went the wrong way turns every following attempt into
noise.

**Dying along the way is not "lost", it is normal.** A route that runs past
live enemies (skeletons on the way to a boss, say) will occasionally get the
character killed -- Dark Souls' own answer to that is to respawn at the last
bonfire, which for a run-back *is the route's own starting point*. So death is
detected (reusing :class:`~flyds1.envs.detectors.DeathScreenDetector`) and
handled by waiting out the reload and restarting the route from waypoint zero,
up to a configured number of times, rather than being confused for a
navigation failure. The alternative -- teaching the connectome to reliably
dodge or fight those enemies -- is a much larger, unproven claim (see
docs/results.md on how little steering information this pipeline has been
shown to extract even from a plain visual target); tolerating an occasional
death is the honest, buildable way to get a reliable run past them today.

Matching is zero-mean normalised correlation on a heavily downscaled greyscale
patch. Downscaling and mean-removal are what make it survive the things that
change between visits to the same spot: torch flicker, the player's own
animation, HUD numbers, minor camera drift. It is not scene understanding and
does not pretend to be -- it answers one question, "does this look like the
place I recorded?", and answers it robustly enough to gate a retry.

The design is also the one insects actually use: route memory plus reactive
avoidance, not a planner. ``Route.run`` takes an optional ``steer`` callback so
the connectome can contribute turning while the route supplies the direction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flyds1.envs.reward import BarRegion
from flyds1.vision.ommatidia import luminance, resize_nearest

#: Size a frame is reduced to before matching.  Small on purpose: at this scale
#: a checkpoint is the *layout* of a place -- wall, doorway, skyline -- and not
#: its flickering detail.
PATCH_SHAPE = (24, 42)


def normalised_patch(
    frame: np.ndarray,
    region: BarRegion | None = None,
    shape: tuple[int, int] = PATCH_SHAPE,
) -> np.ndarray:
    """Frame -> small zero-mean, unit-variance greyscale patch."""
    arr = np.asarray(frame, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = luminance(arr[..., :3])
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    if region is not None:
        arr = region.crop(arr)
    small = resize_nearest(arr, *shape).astype(float)
    centred = small - small.mean()
    scale = float(np.sqrt((centred ** 2).mean()))
    return centred / scale if scale > 1e-8 else centred


def patch_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of two normalised patches, in [-1, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"patch shapes differ: {a.shape} vs {b.shape}")
    return float((a * b).mean())


@dataclass
class Checkpoint:
    """"Does this look like the place I recorded?"."""

    reference: np.ndarray
    threshold: float = 0.55
    region: BarRegion | None = None
    name: str = ""

    def score(self, frame: np.ndarray) -> float:
        return patch_similarity(self.reference, normalised_patch(frame, self.region))

    def matches(self, frame: np.ndarray) -> tuple[bool, float]:
        value = self.score(frame)
        return value >= self.threshold, value


@dataclass
class Waypoint:
    """Hold these buttons for this long, then expect to be *there*."""

    buttons: tuple[str, ...] = ()
    seconds: float = 1.0
    checkpoint: Checkpoint | None = None
    retries: int = 2
    name: str = ""

    #: Wiggle played before replaying a failed segment: a short turn, which is
    #: what gets a character off a doorframe or a corpse more often than not.
    unstick: tuple[tuple[tuple[str, ...], float], ...] = (
        (("right",), 0.25),
        (("forward",), 0.35),
        (("left",), 0.25),
    )


@dataclass
class RouteResult:
    """What happened on one run-back."""

    completed: bool
    failed_at: int | None = None
    scores: list[float] = field(default_factory=list)
    attempts: list[int] = field(default_factory=list)
    #: Times the route died and restarted from the beginning. Not a failure by
    #: itself -- see the module docstring on why dying past live enemies is
    #: treated as normal and handled by restarting, not by giving up.
    deaths: int = 0
    #: True only when the route was abandoned because it kept dying, past
    #: ``max_respawns`` -- distinct from a genuine "lost" (checkpoint
    #: mismatch), which is ``failed_at`` with ``deaths`` possibly still 0.
    died_out: bool = False

    def describe(self) -> str:
        died = f", died {self.deaths}x en route" if self.deaths else ""
        if self.completed:
            worst = min(self.scores) if self.scores else float("nan")
            return (
                f"route completed, {len(self.scores)} checkpoints, "
                f"worst match {worst:+.2f}{died}"
            )
        if self.died_out:
            return f"route abandoned after dying {self.deaths}x without reaching the end"
        return (
            f"route lost at waypoint {self.failed_at} "
            f"(match {self.scores[-1]:+.2f} if any checkpoint was reached){died}"
        )


@dataclass
class Route:
    """A fixed run-back, optionally verified at every step."""

    waypoints: tuple[Waypoint, ...] = ()
    name: str = "route"

    @property
    def duration_s(self) -> float:
        return float(sum(w.seconds for w in self.waypoints))

    # ------------------------------------------------------------------
    def run(
        self,
        backend,
        bindings: dict[str, str],
        grab,
        *,
        sleep_fn=None,
        fps: float = 30.0,
        steer=None,
        handle_death: bool = True,
        respawn_wait_s: float = 10.0,
        max_respawns: int = 3,
    ) -> RouteResult:
        """Walk the route, verifying each checkpoint.

        ``grab`` returns the current frame. ``steer`` is an optional
        ``callable(frame) -> iterable[str]`` contributing extra buttons each
        frame -- the hook through which the connectome does local avoidance
        while the route supplies the direction.

        ``handle_death`` treats dying along the way as a restart rather than a
        "lost" route: Dark Souls respawns at the last bonfire, which for a
        run-back is the route's own starting point, so the fix is to wait out
        the reload and begin again from waypoint zero. ``max_respawns`` caps
        that so a route that always leads to death does not loop forever.
        """
        import time as _time

        from flyds1.envs.detectors import DeathScreenDetector

        sleep = sleep_fn if sleep_fn is not None else _time.sleep
        detector = DeathScreenDetector() if handle_death else None
        deaths = 0

        while True:
            result = RouteResult(completed=True, deaths=deaths)
            died = False
            for index, waypoint in enumerate(self.waypoints):
                for attempt in range(waypoint.retries + 1):
                    frame, died = self._hold(backend, bindings, grab, waypoint.buttons,
                                             waypoint.seconds, sleep, fps, steer, detector)
                    if died:
                        break
                    if waypoint.checkpoint is None:
                        result.attempts.append(attempt)
                        break
                    ok, score = waypoint.checkpoint.matches(frame)
                    if ok:
                        result.scores.append(score)
                        result.attempts.append(attempt)
                        break
                    if attempt == waypoint.retries:
                        backend.release_all()
                        result.completed = False
                        result.failed_at = index
                        result.scores.append(score)
                        return result
                    for buttons, seconds in waypoint.unstick:
                        _, died = self._hold(backend, bindings, grab, buttons, seconds,
                                             sleep, fps, None, detector)
                        if died:
                            break
                    if died:
                        break
                if died:
                    break

            if not died:
                backend.release_all()
                result.deaths = deaths
                return result

            backend.release_all()
            deaths += 1
            if deaths > max_respawns:
                result.completed = False
                result.died_out = True
                result.deaths = deaths
                return result
            sleep(respawn_wait_s)
            detector.reset()
            # falls through to the top of the while loop: restart at waypoint 0

    def _hold(self, backend, bindings, grab, buttons, seconds, sleep, fps, steer, death_detector=None):
        """Hold buttons for ``seconds``, grabbing frames throughout.

        Returns ``(frame, died)``; stops early the moment ``death_detector``
        (if given) confirms a death screen, since holding movement keys into a
        death/reload screen accomplishes nothing.
        """
        period = 1.0 / max(1e-6, fps)
        frames = max(1, int(round(seconds / period)))
        frame = None
        for _ in range(frames):
            held = {name: True for name in buttons}
            if steer is not None and frame is not None:
                for name in steer(frame):
                    held[name] = True
            backend.apply(held, bindings)
            sleep(period)
            frame = grab()
            if death_detector is not None and death_detector.update(frame):
                backend.release_all()
                return frame, True
        backend.release_all()
        return frame, False

    # ------------------------------------------------------------------
    def record(
        self,
        backend,
        bindings: dict[str, str],
        grab,
        *,
        sleep_fn=None,
        fps: float = 30.0,
        threshold: float = 0.55,
        region: BarRegion | None = None,
    ) -> "Route":
        """Walk the route once and capture what each waypoint looks like.

        Returns a new Route whose waypoints carry checkpoints. Run this from
        the same starting position the run-back will start from, or the
        references describe a place the agent will never be.
        """
        import time as _time

        sleep = sleep_fn if sleep_fn is not None else _time.sleep
        recorded = []
        for waypoint in self.waypoints:
            frame, _died = self._hold(backend, bindings, grab, waypoint.buttons,
                                      waypoint.seconds, sleep, fps, steer=None)
            checkpoint = Checkpoint(
                reference=normalised_patch(frame, region),
                threshold=threshold,
                region=region,
                name=waypoint.name,
            )
            recorded.append(
                Waypoint(
                    buttons=waypoint.buttons,
                    seconds=waypoint.seconds,
                    checkpoint=checkpoint,
                    retries=waypoint.retries,
                    name=waypoint.name,
                    unstick=waypoint.unstick,
                )
            )
        backend.release_all()
        return Route(waypoints=tuple(recorded), name=self.name)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Store the route and its reference patches in one ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "waypoints": [
                {
                    "buttons": list(w.buttons),
                    "seconds": w.seconds,
                    "retries": w.retries,
                    "name": w.name,
                    "unstick": [[list(b), s] for b, s in w.unstick],
                    "has_checkpoint": w.checkpoint is not None,
                    "threshold": w.checkpoint.threshold if w.checkpoint else None,
                    "region": (
                        [w.checkpoint.region.x0, w.checkpoint.region.y0,
                         w.checkpoint.region.x1, w.checkpoint.region.y1]
                        if w.checkpoint is not None and w.checkpoint.region is not None
                        else None
                    ),
                }
                for w in self.waypoints
            ],
        }
        references = [
            w.checkpoint.reference for w in self.waypoints if w.checkpoint is not None
        ]
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),
            references=np.stack(references) if references else np.zeros((0, *PATCH_SHAPE)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Route":
        data = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        references = list(data["references"])
        waypoints = []
        for entry in meta["waypoints"]:
            checkpoint = None
            if entry["has_checkpoint"]:
                region = BarRegion(*entry["region"]) if entry["region"] else None
                checkpoint = Checkpoint(
                    reference=references.pop(0),
                    threshold=entry["threshold"],
                    region=region,
                    name=entry["name"],
                )
            waypoints.append(
                Waypoint(
                    buttons=tuple(entry["buttons"]),
                    seconds=entry["seconds"],
                    checkpoint=checkpoint,
                    retries=entry["retries"],
                    name=entry["name"],
                    unstick=tuple((tuple(b), s) for b, s in entry["unstick"]),
                )
            )
        return cls(waypoints=tuple(waypoints), name=meta["name"])

    def describe(self) -> str:
        verified = sum(1 for w in self.waypoints if w.checkpoint is not None)
        return (
            f"Route({self.name!r}): {len(self.waypoints)} waypoints, "
            f"{self.duration_s:.1f}s, {verified} verified by a checkpoint"
        )


#: A rough Undead Asylum run-back to start from -- walk it once with
#: ``flyds1 route --record`` and the checkpoints get filled in from your game.
DEFAULT_ASYLUM_ROUTE = Route(
    waypoints=(
        Waypoint(buttons=("forward",), seconds=3.0, name="out of the bonfire room"),
        Waypoint(buttons=("forward", "right"), seconds=1.0, name="turn to the stairs"),
        Waypoint(buttons=("forward",), seconds=3.5, name="up the stairs"),
        Waypoint(buttons=("forward",), seconds=2.0, name="through the fog"),
    ),
    name="undead asylum -> asylum demon",
)


__all__ = [
    "DEFAULT_ASYLUM_ROUTE",
    "PATCH_SHAPE",
    "Checkpoint",
    "Route",
    "RouteResult",
    "Waypoint",
    "normalised_patch",
    "patch_similarity",
]
