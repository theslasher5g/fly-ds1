"""The canvas instrument panel: publish state, let the browser draw it.

Three files, three update rates, chosen by how often each actually changes:

``layout.json``
    Once. Facet coordinates, population sizes, cell-type groups, descending
    labels, decoder weights. ~100 KB for a full-size eye, sent one time.
``state.json``
    Every step. A few thousand floats: per-stage statistics, hemisphere
    balance, cell-type means, both retinae, descending rates, held keys.
``frame.png``
    Every ``frame_every``-th step. The game frame is the one thing that has to
    be a bitmap, and it is also the only expensive thing to encode, so it runs
    slower than the rest of the panel on purpose.

Measured against the viewer this replaces: the old path PNG-encoded a ~960 px
filmstrip *every* step, which is both more bytes and more CPU inside the step
budget than the JSON is. The richer panel is the cheaper one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flyds1.viz.server import FilePublisher
from flyds1.viz.state import LiveStateBuilder

#: Files copied out of the package into the served directory.
ASSETS: tuple[str, ...] = ("index.html", "fly.css", "fly.js")


def _asset_text(name: str) -> str:
    return (Path(__file__).parent / "assets" / name).read_text(encoding="utf-8")


@dataclass
class PanelView:
    """Publishes ``layout.json`` / ``state.json`` / ``frame.png`` for the page.

    Parameters
    ----------
    directory:
        Where the files go; also what the HTTP server serves.
    refresh_ms:
        How often the page re-fetches ``state.json``.
    frame_every:
        Publish the game frame on every n-th update. 1 means every step.
    frame_max_width:
        Downscale the game frame to at most this width before encoding. The
        panel shows it in a box a few hundred pixels wide, so encoding a 1920 px
        capture is pure cost.
    """

    directory: Path | str
    refresh_ms: int = 100
    frame_every: int = 3
    frame_max_width: int = 480
    _pub: FilePublisher = field(init=False)

    def __post_init__(self) -> None:
        self._pub = FilePublisher(self.directory)
        self.directory = self._pub.directory
        for name in ASSETS:
            text = _asset_text(name)
            if name == "index.html":
                text = text.replace("__REFRESH_MS__", str(int(self.refresh_ms)))
            (self.directory / name).write_text(text, encoding="utf-8")
        self._n = 0
        # Until the first state is published the page would 404 on state.json
        # and log a console error on every poll; an explicit "not started yet"
        # is friendlier and lets the page show a waiting message.
        self._pub.publish_text("state.json", '{"waiting":true}')

    # ------------------------------------------------------------------
    def serve(self, port: int = 8000) -> str:
        return self._pub.serve(port)

    def stop(self) -> None:
        self._pub.stop()

    @property
    def url(self) -> str | None:
        return self._pub.url

    # ------------------------------------------------------------------
    def publish_layout(self, layout: dict) -> None:
        self._pub.publish_text("layout.json", LiveStateBuilder.to_json(layout))

    def update(self, state: dict, *, frame: np.ndarray | None = None) -> None:
        """Publish one step's state, and the game frame every n-th step."""
        if frame is not None and self._n % max(1, self.frame_every) == 0:
            self._publish_frame(frame)
        self._n += 1
        # State goes last: the page reads state.json and then draws, so a frame
        # that is one step stale is invisible, while a state that refers to a
        # frame not yet written is a flicker.
        self._pub.publish_text("state.json", LiveStateBuilder.to_json(state))

    def _publish_frame(self, frame: np.ndarray) -> None:
        from flyds1.plotting import encode_png, to_uint8

        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(arr)) > 1.5 else 1.0
            arr = to_uint8(np.asarray(arr, dtype=float) / scale)
        width = arr.shape[1]
        if width > self.frame_max_width:
            stride = int(np.ceil(width / self.frame_max_width))
            arr = arr[::stride, ::stride]
        self._pub.publish_bytes("frame.png", encode_png(arr))


__all__ = ["ASSETS", "PanelView"]
