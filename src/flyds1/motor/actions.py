"""Step 5a: what "an action" is.

Descending neurons are the only way the brain reaches the body: ~1300 of them
in Drosophila, carrying steering, walking, grooming and escape commands to the
ventral nerve cord.  Here they reach a keyboard instead, so an action is a set
of held keys plus a mouse delta.

The mapping is data, not code, so re-binding for a different game (or a
different key layout) is a config change.  ``ActionSpec`` deliberately models
*held* buttons: a fly's descending activity is continuous, and in a souls game
holding block or run matters as much as tapping attack.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ButtonAction:
    """One binary control the agent can hold."""

    name: str
    key: str          # keyboard key name, or "mouse:left" / "mouse:right"
    description: str = ""


@dataclass
class ActionSpec:
    """A set of buttons plus optional continuous camera axes."""

    buttons: tuple[ButtonAction, ...]
    camera_axes: tuple[str, ...] = ("camera_x", "camera_y")
    camera_scale_px: float = 40.0

    @property
    def n_buttons(self) -> int:
        return len(self.buttons)

    @property
    def n_camera(self) -> int:
        return len(self.camera_axes)

    @property
    def size(self) -> int:
        """Total action dimension (buttons then camera axes)."""
        return self.n_buttons + self.n_camera

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.buttons) + self.camera_axes

    def button_index(self, name: str) -> int:
        for i, b in enumerate(self.buttons):
            if b.name == name:
                return i
        raise KeyError(f"no button {name!r}; have {[b.name for b in self.buttons]}")

    def decode(self, action, *, threshold: float = 0.0) -> tuple[dict[str, bool], tuple[float, float]]:
        """Raw action vector -> ``({button: held}, (dx, dy))`` in pixels."""
        import numpy as np

        arr = np.asarray(action, dtype=float).ravel()
        if arr.size != self.size:
            raise ValueError(f"action has size {arr.size}, expected {self.size}")
        held = {b.name: bool(arr[i] > threshold) for i, b in enumerate(self.buttons)}
        cam = arr[self.n_buttons :]
        dx = float(cam[0]) * self.camera_scale_px if self.n_camera > 0 else 0.0
        dy = float(cam[1]) * self.camera_scale_px if self.n_camera > 1 else 0.0
        return held, (dx, dy)


#: Default bindings for Dark Souls on a keyboard+mouse (PC defaults).
DARKSOULS_ACTIONS = ActionSpec(
    buttons=(
        ButtonAction("forward", "w", "walk forward"),
        ButtonAction("back", "s", "walk back"),
        ButtonAction("left", "a", "strafe/turn left"),
        ButtonAction("right", "d", "strafe/turn right"),
        ButtonAction("sprint_roll", "space", "hold to sprint, tap to roll"),
        ButtonAction("attack", "mouse:left", "light attack"),
        ButtonAction("block", "mouse:right", "raise shield"),
        ButtonAction("lock_on", "q", "toggle lock-on"),
        ButtonAction("heal", "r", "use estus"),
    ),
    camera_axes=("camera_x", "camera_y"),
    camera_scale_px=40.0,
)

#: A minimal spec for the synthetic arena: move and turn only.
ARENA_ACTIONS = ActionSpec(
    buttons=(
        ButtonAction("forward", "w", "move forward"),
        ButtonAction("back", "s", "move back"),
        ButtonAction("left", "a", "turn left"),
        ButtonAction("right", "d", "turn right"),
    ),
    camera_axes=(),
)

__all__ = ["ARENA_ACTIONS", "DARKSOULS_ACTIONS", "ActionSpec", "ButtonAction"]
