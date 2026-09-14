"""The live viewer: files on disk, a server, and no effect on the run."""

import json
import urllib.request

import numpy as np
import pytest

from flyds1.live import LiveView, format_stats


def test_view_writes_a_panel_and_stats(tmp_path):
    view = LiveView(tmp_path)
    view.update(
        np.zeros((32, 48, 3), dtype=np.uint8),
        eye_coords=np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.3]]),
        photoreceptors=np.array([0.2, -0.4, 0.9]),
        motion=np.array([0.1, -0.2, 0.3]),
        descending=np.array([1.0, -0.5, 0.25]),
        stats="hello",
    )
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "panel.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "stats.txt").read_text() == "hello"
    # no half-written temporaries left behind
    assert not list(tmp_path.glob(".*tmp*"))


def test_view_survives_missing_optional_panels(tmp_path):
    view = LiveView(tmp_path)
    view.update(np.zeros((16, 16)), stats="")
    assert (tmp_path / "panel.png").exists()


def test_server_serves_the_page_and_the_panel(tmp_path):
    view = LiveView(tmp_path)
    view.update(np.zeros((16, 16)), descending=np.array([0.5, 0.2]), stats="alive")
    url = view.serve(port=0)  # 0 = let the OS pick a free port
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            page = response.read().decode()
        assert "flyds1" in page
        with urllib.request.urlopen(url + "panel.png", timeout=5) as response:
            assert response.headers.get_content_type() == "image/png"
        with urllib.request.urlopen(url + "stats.txt", timeout=5) as response:
            assert response.read().decode() == "alive"
    finally:
        view.stop()


def test_stats_formatting():
    text = format_stats(
        {"state": "fighting", "attempts": 3, "kills": 0, "player_hp": 0.5},
        {"return": -1.25},
    )
    assert "fighting" in text and "0.500" in text and "-1.250" in text
    assert "kills" in text


def test_page_refresh_interval_is_embedded(tmp_path):
    view = LiveView(tmp_path, refresh_ms=123)
    assert "123" in (tmp_path / "index.html").read_text()
