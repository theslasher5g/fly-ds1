"""Route execution: checkpoints, retries, and giving up rather than blundering on."""

import numpy as np
import pytest

from flyds1.envs.input_backends import DryRunBackend
from flyds1.envs.navigation import (
    Checkpoint,
    Route,
    Waypoint,
    normalised_patch,
    patch_similarity,
)
from flyds1.envs.reward import BarRegion
from flyds1.motor.actions import DARKSOULS_ACTIONS

BINDINGS = {b.name: b.key for b in DARKSOULS_ACTIONS.buttons}


def place(seed: int, h: int = 90, w: int = 160) -> np.ndarray:
    """A distinct-looking 'place' that is stable but not identical each visit."""
    rng = np.random.default_rng(seed)
    base = rng.random((9, 16))
    frame = np.kron(base, np.ones((h // 9, w // 16)))
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


def jitter(frame: np.ndarray, rng: np.random.Generator, amount: float = 0.08) -> np.ndarray:
    """Same place, different visit: torch flicker, animation, slight drift."""
    noisy = frame.astype(float) / 255.0 + rng.normal(scale=amount, size=frame.shape)
    return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)


def test_patch_matching_separates_places():
    a, b = place(1), place(2)
    rng = np.random.default_rng(0)
    same = patch_similarity(normalised_patch(a), normalised_patch(jitter(a, rng)))
    different = patch_similarity(normalised_patch(a), normalised_patch(b))
    assert same > 0.8, "the same place after a bit of flicker must still match"
    assert different < 0.5, "two different places must not"
    assert same > different


def test_patch_is_invariant_to_overall_brightness():
    frame = place(3)
    dimmed = (frame.astype(float) * 0.4).astype(np.uint8)
    assert patch_similarity(normalised_patch(frame), normalised_patch(dimmed)) > 0.95


def test_checkpoint_region_restricts_matching():
    frame = place(4)
    top = Checkpoint(reference=normalised_patch(frame, BarRegion(0, 0, 1, 0.5)),
                     region=BarRegion(0, 0, 1, 0.5))
    changed = frame.copy()
    changed[45:, :] = 0  # wreck the bottom half only
    ok, score = top.matches(changed)
    assert ok and score > 0.9


class Walk:
    """A fake game: each segment of the route lands you in a known place."""

    def __init__(self, places: list[int], derail_after: int | None = None):
        self.places = places
        self.derail_after = derail_after
        self.frames = 0
        self.rng = np.random.default_rng(0)

    def grab(self) -> np.ndarray:
        self.frames += 1
        index = min((self.frames - 1) // 3, len(self.places) - 1)
        if self.derail_after is not None and index > self.derail_after:
            return jitter(place(999), self.rng)  # somewhere else entirely
        return jitter(place(self.places[index]), self.rng)


def build_route(n: int = 3) -> Route:
    return Route(
        waypoints=tuple(
            Waypoint(buttons=("forward",), seconds=0.1, name=f"leg {i}", retries=1)
            for i in range(n)
        ),
        name="test route",
    )


def test_record_then_replay_completes():
    backend = DryRunBackend()
    walk = Walk([10, 11, 12])
    recorded = build_route().record(
        backend, BINDINGS, walk.grab, sleep_fn=lambda _s: None, fps=30.0
    )
    assert all(w.checkpoint is not None for w in recorded.waypoints)

    replay = Walk([10, 11, 12])
    result = recorded.run(
        DryRunBackend(), BINDINGS, replay.grab, sleep_fn=lambda _s: None, fps=30.0
    )
    assert result.completed
    assert len(result.scores) == 3
    assert min(result.scores) > 0.5


def test_a_derailed_run_is_reported_not_continued():
    backend = DryRunBackend()
    walk = Walk([20, 21, 22])
    recorded = build_route().record(
        backend, BINDINGS, walk.grab, sleep_fn=lambda _s: None, fps=30.0
    )

    lost = Walk([20, 21, 22], derail_after=0)  # goes wrong after the first leg
    result = recorded.run(
        DryRunBackend(), BINDINGS, lost.grab, sleep_fn=lambda _s: None, fps=30.0
    )
    assert not result.completed
    assert result.failed_at == 1
    assert "lost" in result.describe()


def test_steer_callback_can_add_buttons():
    backend = DryRunBackend()
    route = Route(waypoints=(Waypoint(buttons=("forward",), seconds=0.1),))
    walk = Walk([30])
    route.run(backend, BINDINGS, walk.grab, sleep_fn=lambda _s: None, fps=30.0,
              steer=lambda _frame: ["left"])
    pressed = {key for kind, key in backend.log if kind == "press"}
    assert "w" in pressed and "a" in pressed


def test_route_survives_a_save_load_round_trip(tmp_path):
    backend = DryRunBackend()
    walk = Walk([40, 41])
    recorded = build_route(2).record(
        backend, BINDINGS, walk.grab, sleep_fn=lambda _s: None, fps=30.0
    )
    path = recorded.save(tmp_path / "route.npz")
    back = Route.load(path)

    assert back.name == recorded.name
    assert len(back.waypoints) == len(recorded.waypoints)
    for original, restored in zip(recorded.waypoints, back.waypoints):
        assert restored.buttons == original.buttons
        assert restored.seconds == pytest.approx(original.seconds)
        assert np.allclose(restored.checkpoint.reference, original.checkpoint.reference)

    replay = Walk([40, 41])
    assert back.run(
        DryRunBackend(), BINDINGS, replay.grab, sleep_fn=lambda _s: None, fps=30.0
    ).completed


def test_route_without_checkpoints_still_runs():
    route = build_route(2)  # never recorded, so no checkpoints
    walk = Walk([50, 51])
    result = route.run(
        DryRunBackend(), BINDINGS, walk.grab, sleep_fn=lambda _s: None, fps=30.0
    )
    assert result.completed and result.scores == []


def test_describe_reports_verification_coverage():
    backend = DryRunBackend()
    walk = Walk([60, 61])
    recorded = build_route(2).record(
        backend, BINDINGS, walk.grab, sleep_fn=lambda _s: None, fps=30.0
    )
    assert "2 verified" in recorded.describe()
    assert "0 verified" in build_route(2).describe()
