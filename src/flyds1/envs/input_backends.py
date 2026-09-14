"""Sending keys and mouse movement to a running game.

Three backends, chosen by availability and platform:

* ``dry`` -- records what *would* be pressed and sends nothing.  The default,
  because an agent that starts pressing keys the moment you import a module is
  not a good neighbour.
* ``pydirectinput`` -- Windows.  Uses scancode-level ``SendInput``, which is
  what DirectX titles read; ``pyautogui``-style messages are ignored by them.
* ``xdotool`` -- Linux/X11, via subprocess.  Works for windowed OpenGL/Vulkan
  games (native or Proton).  Wayland has no equivalent, so it will refuse.

**Both real backends deliver to whatever window currently has OS focus, not to
a window you name.** ``SendInput`` on Windows and, in practice, most X11
synthetic-event delivery both work that way. That means the moment the game
loses focus -- alt-tab, clicking the terminal, the live-view browser stealing
focus -- every key press and every mouse-move call keeps firing at whatever
*is* now focused: the desktop, a text field, another window. Observed directly:
tabbing out of Dark Souls left the agent clicking and moving the mouse across
the desktop. :class:`_FocusGuardedBackend` is the fix -- both real backends
check the foreground window's title against a configured substring before
acting, release whatever they were holding the moment focus is lost (so
alt-tabbing away never leaves a key latched down in the game once you return),
and resume once the title matches again. ``DryRunBackend`` needs no guard: it
never touches the OS.

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


def _title_contains(active_title: str | None, wanted: str | None) -> bool:
    """Case-insensitive substring match, with ``wanted=None`` meaning "no
    window configured to check, so do not block" -- pure and platform-free so
    the gating policy below is testable without a real window manager."""
    if not wanted:
        return True
    return wanted.lower() in (active_title or "").lower()


class _FocusGuardedBackend(InputBackend):
    """An :class:`InputBackend` that only acts while a named window has focus.

    Subclasses implement :meth:`has_focus` (the platform-specific part, e.g. a
    ``ctypes``/``user32`` call on Windows) and :meth:`_move_mouse_unguarded`
    (the real mouse move); this class holds the policy, which is what actually
    matters and is what the tests exercise, independent of any platform:

    * while unfocused, no new key or button is pressed and the mouse does not
      move -- this is the fix for input leaking to the desktop, a terminal, or
      whatever else the user tabbed to;
    * the instant focus is lost, everything currently held is released *once*
      (not on every subsequent unfocused call), so a key never reads as
      latched down when the game regains focus later;
    * ``window=None`` disables the guard entirely (treated as "always
      focused") -- only meaningful if you have your own reason to trust
      whatever else might be focused, which normal use never does.
    """

    def __init__(self, window: str | None) -> None:
        super().__init__()
        self.window = window
        self._was_focused = True

    def has_focus(self) -> bool:  # pragma: no cover - overridden per platform
        return True

    def apply(self, held: dict[str, bool], bindings: dict[str, str]) -> None:
        if not self.has_focus():
            if self._was_focused:
                super().release_all()
                self._was_focused = False
            return
        self._was_focused = True
        super().apply(held, bindings)

    def move_mouse(self, dx: float, dy: float) -> None:
        if not self.has_focus():
            return
        self._move_mouse_unguarded(dx, dy)

    def _move_mouse_unguarded(self, dx: float, dy: float) -> None:  # pragma: no cover
        raise NotImplementedError


class PyDirectInputBackend(_FocusGuardedBackend):  # pragma: no cover - Windows only
    """Windows scancode injection via ``pydirectinput``."""

    def __init__(self, window: str | None = None) -> None:
        super().__init__(window)
        try:
            import pydirectinput
        except ImportError as exc:
            raise ImportError(
                "pydirectinput is Windows-only: pip install -e '.[game]' on Windows"
            ) from exc
        pydirectinput.PAUSE = 0.0  # we pace the loop ourselves
        pydirectinput.FAILSAFE = True  # mouse to a corner aborts: keep it on
        self._pdi = pydirectinput

    def has_focus(self) -> bool:
        if not self.window:
            return True
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return _title_contains(buf.value, self.window)
        except Exception:
            # Cannot check -- fail open rather than lock control out entirely
            # over an unrelated ctypes failure.
            return True

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

    def _move_mouse_unguarded(self, dx: float, dy: float) -> None:
        if dx or dy:
            self._pdi.moveRel(int(dx), int(dy), relative=True)


class XdotoolBackend(_FocusGuardedBackend):  # pragma: no cover - needs an X server
    """Linux/X11 injection via the ``xdotool`` binary.

    ``search --name`` chaining targets the named window for the action itself,
    which helps, but is not the same guarantee as real focus -- a search that
    matches nothing (title changed, window closed, wrong string) silently falls
    back to whatever xdotool's default target is. The focus check is the
    actual safety net.
    """

    _BUTTONS = {"left": "1", "middle": "2", "right": "3"}

    def __init__(self, window: str | None = None) -> None:
        super().__init__(window)
        if shutil.which("xdotool") is None:
            raise RuntimeError("xdotool not found; install it or use backend='dry'")

    def has_focus(self) -> bool:
        if not self.window:
            return True
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                check=False, capture_output=True, text=True, timeout=1.0,
            )
            return _title_contains(result.stdout, self.window)
        except Exception:
            return True

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

    def _move_mouse_unguarded(self, dx: float, dy: float) -> None:
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
        return PyDirectInputBackend(window=window)
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
