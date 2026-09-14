"""Step 5a: what "an action" is.

Descending neurons are the only way the brain reaches the body: ~1300 of them
in Drosophila, carrying steering, walking, grooming and escape commands to the
ventral nerve cord.  Here they reach a keyboard instead, so an action is a set
of held keys plus a mouse delta.

The mapping is data, not code, so re-binding for a different game (or a
different key layout) is a config change.  ``ActionSpec`` deliberately models
*held* buttons: a fly's descending activity is continuous, and in a souls game
holding block or run matters as much as tapping attack.

**Buttons are binary, and the action space should say so.**  Representing them
as a continuous vector thresholded at zero looks harmless and is not: with a
Gaussian policy of unit standard deviation, every button flips with probability
~0.5 no matter what the policy has learnt, and a deterministic evaluation then
decides behaviour by the sign of means three orders of magnitude smaller than
that noise.  Measured on this arena: mean magnitude 0.004 against a policy
standard deviation of 1.0.  A Bernoulli policy over ``MultiBinary`` actions
maps a logit shift straight onto a change in behaviour, so that is the default;
``camera_mode`` decides whether mouse movement joins them as four more buttons
or stays continuous.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ButtonAction:
    """One binary control the agent can hold."""

    name: str
    key: str          # keyboard key name, or "mouse:left" / "mouse:right"
    description: str = ""


#: Names of the four camera buttons, in the order they appear in an action.
CAMERA_BUTTON_NAMES = ("camera_left", "camera_right", "camera_up", "camera_down")


@dataclass
class ActionSpec:
    """A set of buttons, plus camera control as axes, buttons, or nothing.

    ``camera_mode``:

    * ``"buttons"`` -- four more binary controls; the whole action is
      ``MultiBinary`` and a Bernoulli policy fits it exactly.
    * ``"continuous"`` -- two axes in [-1, 1]; needs a ``Box`` action space.
    * ``"none"`` -- no camera control at all (the arena).
    """

    buttons: tuple[ButtonAction, ...]
    camera_mode: str = "buttons"
    camera_scale_px: float = 40.0

    def __post_init__(self) -> None:
        if self.camera_mode not in ("buttons", "continuous", "none"):
            raise ValueError(
                f"camera_mode must be 'buttons', 'continuous' or 'none', got {self.camera_mode!r}"
            )

    @property
    def n_buttons(self) -> int:
        return len(self.buttons)

    @property
    def n_camera(self) -> int:
        return {"buttons": 4, "continuous": 2, "none": 0}[self.camera_mode]

    @property
    def is_binary(self) -> bool:
        """True when every entry of an action is a button, i.e. MultiBinary."""
        return self.camera_mode != "continuous"

    @property
    def size(self) -> int:
        """Total action dimension (buttons first, camera last)."""
        return self.n_buttons + self.n_camera

    @property
    def camera_axes(self) -> tuple[str, ...]:
        if self.camera_mode == "buttons":
            return CAMERA_BUTTON_NAMES
        if self.camera_mode == "continuous":
            return ("camera_x", "camera_y")
        return ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.buttons) + self.camera_axes

    def with_camera(self, mode: str) -> "ActionSpec":
        """A copy using a different camera representation."""
        return ActionSpec(self.buttons, camera_mode=mode, camera_scale_px=self.camera_scale_px)

    def button_index(self, name: str) -> int:
        for i, b in enumerate(self.buttons):
            if b.name == name:
                return i
        raise KeyError(f"no button {name!r}; have {[b.name for b in self.buttons]}")

    def decode(self, action, *, threshold: float = 0.0) -> tuple[dict[str, bool], tuple[float, float]]:
        """Raw action vector -> ``({button: held}, (dx, dy))`` in pixels.

        Works for both encodings: ``MultiBinary`` gives 0/1 and ``Box`` gives
        values in [-1, 1], and "held" means "above ``threshold``" either way.
        """
        import numpy as np

        arr = np.asarray(action, dtype=float).ravel()
        if arr.size != self.size:
            raise ValueError(f"action has size {arr.size}, expected {self.size}")
        held = {b.name: bool(arr[i] > threshold) for i, b in enumerate(self.buttons)}
        cam = arr[self.n_buttons :]
        if self.camera_mode == "buttons":
            left, right, up, down = (float(v) > threshold for v in cam)
            dx = (float(right) - float(left)) * self.camera_scale_px
            dy = (float(down) - float(up)) * self.camera_scale_px
        elif self.camera_mode == "continuous":
            dx = float(cam[0]) * self.camera_scale_px
            dy = float(cam[1]) * self.camera_scale_px
        else:
            dx = dy = 0.0
        return held, (dx, dy)

    def gym_space(self):
        """The Gymnasium action space this spec should be exposed as."""
        from gymnasium import spaces

        if self.is_binary:
            return spaces.MultiBinary(self.size)
        return spaces.Box(low=-1.0, high=1.0, shape=(self.size,), dtype=__import__("numpy").float32)


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
    camera_mode="buttons",
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
    camera_mode="none",
)

__all__ = [
    "ARENA_ACTIONS",
    "CAMERA_BUTTON_NAMES",
    "DARKSOULS_ACTIONS",
    "ActionSpec",
    "ButtonAction",
]
