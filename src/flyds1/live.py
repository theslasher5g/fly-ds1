"""Watching the fly play, live.

Renders one panel per decision -- the game frame, what the ommatidia see, the
motion channels, and the descending-neuron activity that produced the key
presses -- and serves it on a local web page that refreshes itself.  A browser
window is the least intrusive viewer: it does not steal focus from the game,
which an OpenCV window on the same screen happily does mid-fight.

Deliberately read-only: the viewer never touches the environment, so watching
cannot change what the agent does.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flyds1.plotting import bar_chart, facet_image, filmstrip, save_png, to_uint8

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>flyds1 - live</title>
<style>
  body {{ margin: 0; background: #14161a; color: #d8dde5;
         font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  header {{ padding: 10px 14px; border-bottom: 1px solid #262b33; }}
  img {{ display: block; width: 100%; image-rendering: pixelated; }}
  #stats {{ padding: 10px 14px; white-space: pre; }}
  .legend {{ color: #8a93a2; }}
</style>
<header><b>flyds1</b> <span class="legend">game &middot; what the eye sees &middot;
motion &middot; descending neurons</span></header>
<img id="frame" src="panel.png?0">
<div id="stats">waiting for the first decision...</div>
<script>
let n = 0;
async function tick() {{
  n += 1;
  document.getElementById('frame').src = 'panel.png?' + n;
  try {{
    const r = await fetch('stats.txt?' + n);
    document.getElementById('stats').textContent = await r.text();
  }} catch (e) {{}}
  setTimeout(tick, {interval});
}}
setTimeout(tick, {interval});
</script>
"""


@dataclass
class LiveView:
    """Writes the panel and stats a browser polls.

    ``directory`` holds ``index.html``, ``panel.png`` and ``stats.txt``; the
    page rewrites itself from those, so the render loop only ever writes files
    and never blocks on a client.
    """

    directory: Path
    refresh_ms: int = 200
    panel_width: int = 960

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "index.html").write_text(PAGE.format(interval=self.refresh_ms))
        self._server: socketserver.TCPServer | None = None
        self.url: str | None = None

    # ------------------------------------------------------------------
    def serve(self, port: int = 8000) -> str:
        """Start a background HTTP server; returns the URL to open."""
        directory = str(self.directory)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def log_message(self, *args):  # keep the console for the run itself
                pass

        socketserver.TCPServer.allow_reuse_address = True
        self._server = socketserver.TCPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/"
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # ------------------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        *,
        eye_coords: np.ndarray | None = None,
        photoreceptors: np.ndarray | None = None,
        motion: np.ndarray | None = None,
        descending: np.ndarray | None = None,
        stats: str = "",
    ) -> None:
        """Render one decision's worth of state."""
        panels = [to_uint8(np.asarray(frame, dtype=float) / (255.0 if np.max(frame) > 1.5 else 1.0))]
        if eye_coords is not None and photoreceptors is not None:
            panels.append(facet_image(eye_coords, photoreceptors, signed=True))
        if eye_coords is not None and motion is not None:
            panels.append(facet_image(eye_coords, motion, signed=True))
        if descending is not None:
            panels.append(bar_chart(descending, width=260, height=180))
        strip = filmstrip(panels)
        # write to a temporary name first: a browser polling every 200 ms will
        # otherwise fetch a half-written PNG and show a broken image
        tmp = self.directory / ".panel.tmp.png"
        save_png(strip, tmp)
        tmp.replace(self.directory / "panel.png")
        tmp_stats = self.directory / ".stats.tmp"
        tmp_stats.write_text(stats)
        tmp_stats.replace(self.directory / "stats.txt")


def format_stats(info: dict, extra: dict | None = None) -> str:
    """One block of plain text for the viewer."""
    rows = []
    for key in ("state", "attempts", "kills", "player_hp", "boss_hp", "steps"):
        if key in info:
            value = info[key]
            rows.append(f"{key:14s} {value:.3f}" if isinstance(value, float) else f"{key:14s} {value}")
    for key, value in (extra or {}).items():
        rows.append(f"{key:14s} {value:.3f}" if isinstance(value, float) else f"{key:14s} {value}")
    return "\n".join(rows)


__all__ = ["LiveView", "format_stats"]
