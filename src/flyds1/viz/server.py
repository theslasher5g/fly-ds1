"""Publishing viewer files, and serving them.

Split out of :mod:`flyds1.live` because both viewers -- the old PNG filmstrip
and the new canvas panel -- need exactly the same two mechanics, and only one of
them is obvious:

* **Writes must be atomic.** A browser polling every 100 ms will otherwise fetch
  a half-written file and show a broken image or throw on a truncated JSON
  parse.
* **The replace must tolerate a locked target.** On POSIX, renaming over a file
  someone has open just works. On Windows it raises ``PermissionError`` if the
  HTTP server's GET handler happens to hold the file open at that instant, which
  a fast poll loop makes routine rather than exceptional. A dropped viewer frame
  must never crash the run that produces it.

Both viewers also share the "serve a directory on localhost" part, which stays
deliberately dumb: a background ``SimpleHTTPRequestHandler`` over a folder. The
render loop only ever writes files, so it never blocks on a client, and a
browser that falls behind or disconnects cannot affect the agent.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
import time
from pathlib import Path


class FilePublisher:
    """A directory of viewer files, served on localhost."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._server: socketserver.TCPServer | None = None
        self.url: str | None = None

    # ------------------------------------------------------------------
    def serve(self, port: int = 8000) -> str:
        """Start a background HTTP server; returns the URL to open."""
        directory = str(self.directory)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def end_headers(self):
                # The page re-fetches the same three names forever. Without
                # this, a browser serves state.json from cache and the panel
                # freezes while the run continues -- which looks exactly like
                # a hung agent.
                self.send_header("Cache-Control", "no-store, must-revalidate")
                super().end_headers()

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
    def publish_text(self, name: str, text: str) -> None:
        tmp = self.directory / f".{name}.tmp"
        tmp.write_text(text, encoding="utf-8")
        atomic_replace(tmp, self.directory / name)

    def publish_bytes(self, name: str, data: bytes) -> None:
        tmp = self.directory / f".{name}.tmp"
        tmp.write_bytes(data)
        atomic_replace(tmp, self.directory / name)


def atomic_replace(tmp: Path, dest: Path, attempts: int = 5, delay: float = 0.02) -> None:
    """Move ``tmp`` onto ``dest``, tolerating a concurrent reader.

    Retries briefly, then gives up *silently*, leaving ``tmp`` in place to be
    overwritten and retried on the next call. Giving up is the right behaviour:
    the cost is one missed viewer frame, and the alternative is killing a boss
    attempt over a file lock.
    """
    for attempt in range(attempts):
        try:
            tmp.replace(dest)
            return
        except OSError:
            if attempt == attempts - 1:
                return
            time.sleep(delay)


__all__ = ["FilePublisher", "atomic_replace"]
