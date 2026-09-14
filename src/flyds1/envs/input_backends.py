"""Sending keys and mouse movement to a running game.

Three backends, chosen by availability and platform:

* ``dry`` -- records what *would* be pressed and sends nothing.  The default,
  because an agent that starts pressing keys the moment you import a module is
  not a good neighbour.
* ``pydirectinput`` -- Windows.  Uses scancode-level ``SendInput``, which is
  what DirectX titles read; ``pyautogui``-style messages are ignored by them.
* ``xdotool`` -- Linux/X11, via subprocess.  Works for windowed OpenGL/Vulkan
  games (native or Proton).  Wayland has no equivalent, so it will refuse.

Only use these against a single-player, offline session you control.  Injecting
input into an online session is a matter between you and the game's terms of
service, and anti-cheat systems are entitled to treat it as tampering -- that is
a reason to keep the game offline, not a reason to hide the input.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from abc import ABC, abstractmethod


class InputBackend(ABC):
    """Holds a set of keys down and moves the mouse."""

    def __init__(self) -> None:
        self._held: set[str] = set()

    @abstractmethod
    def _press(self, key: str) -> None: ...

    @abstractmethod
    def _release(self, key: str) -> None: ...

    @abstractmethod
    def move_mouse(self, dx: float, dy: float) -> None: ...

    def apply(self, held: dict[str, bool], bindings: dict[str, str]) -> None:
        """Make the physical state match ``held`` (name -> pressed)."""
        want = {bindings[name] for name, pressed in held.items() if pressed and name in bindings}
        for key in sorted(self._held - want):
            self._release(key)
            self._held.discard(key)
        for key in sorted(want - self._held):
            self._press(key)
            self._held.add(key)

    def release_all(self) -> None:
        for key in sorted(self._held):
            self._release(key)
        self._held.clear()

    @property
    def held_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._held))


class DryRunBackend(InputBackend):
    """Records intended input; sends nothing.  Used in tests and dry runs."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[tuple[str, str | tuple[float, float]]] = []

    def _press(self, key: str) -> None:
        self.log.append(("press", key))

    def _release(self, key: str) -> None:
        self.log.append(("release", key))

    def move_mouse(self, dx: float, dy: float) -> None:
        if dx or dy:
            self.log.append(("move", (float(dx), float(dy))))


class PyDirectInputBackend(InputBackend):  # pragma: no cover - Windows only
    """Windows scancode injection via ``pydirectinput``."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import pydirectinput
        except ImportError as exc:
            raise ImportError(
                "pydirectinput is Windows-only: pip install -e '.[game]' on Windows"
            ) from exc
        pydirectinput.PAUSE = 0.0  # we pace the loop ourselves
        pydirectinput.FAILSAFE = True  # mouse to a corner aborts: keep it on
        self._pdi = pydirectinput

    def _press(self, key: str) -> None:
        if key.startswith("mouse:"):
            self._pdi.mouseDown(button=key.split(":", 1)[1])
        else:
            self._pdi.keyDown(key)

    def _release(self, key: str) -> None:
        if key.startswith("mouse:"):
            self._pdi.mouseUp(button=key.split(":", 1)[1])
        else:
            self._pdi.keyUp(key)

    def move_mouse(self, dx: float, dy: float) -> None:
        if dx or dy:
            self._pdi.moveRel(int(dx), int(dy), relative=True)


class XdotoolBackend(InputBackend):  # pragma: no cover - needs an X server
    """Linux/X11 injection via the ``xdotool`` binary."""

    _BUTTONS = {"left": "1", "middle": "2", "right": "3"}

    def __init__(self, window: str | None = None) -> None:
        super().__init__()
        if shutil.which("xdotool") is None:
            raise RuntimeError("xdotool not found; install it or use backend='dry'")
        self.window = window

    def _run(self, *args: str) -> None:
        cmd = ["xdotool", *args]
        if self.window:
            cmd = ["xdotool", "search", "--name", self.window, *args]
        subprocess.run(cmd, check=False, capture_output=True)

    def _press(self, key: str) -> None:
        if key.startswith("mouse:"):
            self._run("mousedown", self._BUTTONS.get(key.split(":", 1)[1], "1"))
        else:
            self._run("keydown", key)

    def _release(self, key: str) -> None:
        if key.startswith("mouse:"):
            self._run("mouseup", self._BUTTONS.get(key.split(":", 1)[1], "1"))
        else:
            self._run("keyup", key)

    def move_mouse(self, dx: float, dy: float) -> None:
        if dx or dy:
            self._run("mousemove_relative", "--", str(int(dx)), str(int(dy)))


def make_input_backend(kind: str = "dry", *, window: str | None = None) -> InputBackend:
    """Create a backend; ``"auto"`` picks per platform, ``"dry"`` is the default."""
    kind = kind.lower()
    if kind == "auto":
        kind = "pydirectinput" if sys.platform == "win32" else "xdotool"
    if kind in ("dry", "none", "off"):
        return DryRunBackend()
    if kind in ("pydirectinput", "windows"):
        return PyDirectInputBackend()
    if kind in ("xdotool", "x11", "linux"):
        return XdotoolBackend(window=window)
    raise ValueError(f"unknown input backend {kind!r}")


__all__ = [
    "DryRunBackend",
    "InputBackend",
    "PyDirectInputBackend",
    "XdotoolBackend",
    "make_input_backend",
]
