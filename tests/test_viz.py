"""Population indexing and the live state payload."""

import json

import numpy as np
import pytest

from flyds1.connectome.schema import NeuronTable
from flyds1.viz.groups import PopulationIndex, TypeGroup
from flyds1.viz.state import LiveStateBuilder


def _table(rows):
    """rows: (cell_type, super_class, side) -> a NeuronTable with no retinotopy."""
    n = len(rows)
    return NeuronTable(
        ids=np.arange(1, n + 1, dtype=np.int64),
        cell_type=np.array([r[0] for r in rows], dtype=object),
        super_class=np.array([r[1] for r in rows], dtype=object),
        nt_type=np.array(["acetylcholine"] * n, dtype=object),
        side=np.array([r[2] for r in rows], dtype=object),
        eye_coord=np.full((n, 2), np.nan),
    )


def _tiny_index():
    table = _table([
        ("r1-6", "sensory", "left"),
        ("r1-6", "sensory", "right"),
        ("t4a", "optic", "left"),
        ("t4a", "optic", "right"),
        ("t5b", "optic", "right"),
        ("hsn", "visual_projection", "left"),
        ("vs3", "visual_projection", "right"),
        ("lc4", "visual_projection", "right"),
        ("unnamed", "central", "center"),
        ("dn01", "descending", "left"),
        ("dn02", "descending", "right"),
    ])
    return PopulationIndex.from_neurons(table)


def test_groups_cover_sides_stages_and_types():
    idx = _tiny_index()
    assert set(idx.sides) == {"left", "center", "right"}
    assert len(idx.sides["left"]) == 4 and len(idx.sides["right"]) == 6
    # stages come out in signal-flow order, not dict-insertion accident
    assert list(idx.stages) == ["sensory", "optic", "visual_projection", "central", "descending"]
    assert set(idx.types) >= {"R1-6", "T4a", "T5b", "HS", "VS", "LC4"}
    assert len(idx.types["T4a"]) == 2


def test_prefix_matching_catches_the_tangential_cell_families():
    """HSN/HSE/HSS and VS1..VS10 are one functional group spelled many ways."""
    table = _table([("hsn", "visual_projection", "left"),
                    ("hse", "visual_projection", "left"),
                    ("vs1", "visual_projection", "right"),
                    ("vs10", "visual_projection", "right")])
    idx = PopulationIndex.from_neurons(table)
    assert len(idx.types["HS"]) == 2
    assert len(idx.types["VS"]) == 2


def test_exact_matching_does_not_bleed_between_subtypes():
    """Substring matching would make T4a swallow nothing, but T5 swallow T5b --
    the subtypes are the whole point of the motion panel, so they must not
    merge."""
    table = _table([("t4a", "optic", "left"), ("t4ab", "optic", "left")])
    idx = PopulationIndex.from_neurons(table)
    assert len(idx.types["T4a"]) == 1


def test_unknown_super_class_is_kept_not_dropped():
    table = _table([("x", "sensory", "left"), ("y", "mystery", "right")])
    idx = PopulationIndex.from_neurons(table)
    assert "mystery" in idx.stages
    assert sum(len(v) for v in idx.stages.values()) == 2


def test_missing_types_are_reported_rather_than_invented():
    table = _table([("t4a", "optic", "left")])
    idx = PopulationIndex.from_neurons(table)
    assert "LC4" in idx.missing
    assert "LC4" not in idx.types
    assert "not in this dataset" in idx.describe()


def test_custom_type_group_rejects_an_unknown_mode():
    group = TypeGroup("bad", ("x",), mode="regex")
    with pytest.raises(ValueError, match="unknown match mode"):
        group.matches(np.array(["x"], dtype=object))


# ----------------------------------------------------------------------
def test_balance_is_signed_and_zero_when_silent():
    idx = _tiny_index()
    rates = np.zeros(idx.n_neurons)
    assert idx.balance(rates) == 0.0, "a silent brain has no lateral preference"

    rates[idx.sides["right"]] = 1.0
    assert idx.balance(rates) == pytest.approx(1.0)
    rates[:] = 0.0
    rates[idx.sides["left"]] = 1.0
    assert idx.balance(rates) == pytest.approx(-1.0)

    rates[idx.sides["right"]] = 1.0
    assert idx.balance(rates) == pytest.approx(0.0)


def test_balance_is_available_per_stage():
    """The whole-brain average hides the case the project actually cares about:
    a left/right difference that exists in the optic lobe and is gone by the
    descending neurons."""
    idx = _tiny_index()
    rates = np.zeros(idx.n_neurons)
    rates[idx.stage_sides[("optic", "right")]] = 2.0
    assert idx.balance(rates, "optic") == pytest.approx(1.0)
    assert idx.balance(rates, "descending") == 0.0


def test_stage_stats_reports_mean_peak_and_active_fraction():
    idx = _tiny_index()
    rates = np.zeros(idx.n_neurons)
    optic = idx.stages["optic"]
    rates[optic[0]] = 3.0
    mean, peak, active = idx.stage_stats(rates, "optic")
    assert peak == 3.0
    assert mean == pytest.approx(1.0)
    assert active == pytest.approx(1 / 3)
    assert idx.stage_stats(rates, "nonexistent") == (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
def _builder(idx=None):
    idx = idx or _tiny_index()
    return LiveStateBuilder(
        idx,
        eye_coords={"left": np.zeros((2, 2)), "right": np.ones((2, 2))},
        action_names=("forward", "attack"),
        dn_labels=("dn01", "dn02"),
        decoder_weight=np.array([[0.5, -0.25], [1.0, 0.0]]),
    )


def test_layout_is_static_and_json_serialisable():
    builder = _builder()
    layout = builder.layout()
    text = LiveStateBuilder.to_json(layout)
    back = json.loads(text)
    assert back["n_neurons"] == 11
    assert [s["name"] for s in back["stages"]][0] == "sensory"
    assert back["decoder"]["shape"] == [2, 2]
    assert set(back["eyes"]) == {"left", "right"}
    assert back["actions"] == ["forward", "attack"]
    # the per-stage hemisphere counts the balance gauge needs
    optic = next(s for s in back["stages"] if s["name"] == "optic")
    assert optic["left"] == 1 and optic["right"] == 2


def test_state_carries_every_panel():
    builder = _builder()
    rates = np.linspace(0.0, 1.0, builder.index.n_neurons)
    state = builder.update(
        rates,
        step=7,
        reward=0.5,
        eyes={"left": {"photo": np.array([0.1, -0.2]), "motion": None}},
        dn_rates=np.array([0.3, 0.9]),
        action=np.array([1, 0]),
        held_keys=("w",),
        info={"state": "fight", "player_hp": 0.8, "boss_hp": 0.5, "dry_run": True},
    )
    text = LiveStateBuilder.to_json(state)
    back = json.loads(text)
    assert back["step"] == 7
    assert back["player_hp"] == 0.8 and back["state"] == "fight"
    assert back["dry_run"] is True, "a bool must survive as a bool, not become 1.0"
    assert back["dn"] == [0.3, 0.9]
    assert back["action"] == [1, 0]
    assert back["held_keys"] == ["w"]
    assert "motion" not in back["eyes"]["left"]  # None is dropped, not sent as null
    assert {s["name"] for s in back["stages"]} == set(builder.index.stages)
    assert "T4a" in back["types"]


def test_state_rejects_a_wrong_sized_rate_vector():
    builder = _builder()
    with pytest.raises(ValueError, match="expected 11"):
        builder.update(np.zeros(5))


def test_non_finite_values_never_reach_the_browser():
    """Regression guard: NaN is not valid JSON. Python's json emits a bare NaN
    anyway, JSON.parse then throws, and the whole panel goes blank -- so one
    diverged neuron reads as "the fly is dead" instead of "the fly diverged"."""
    builder = _builder()
    rates = np.zeros(builder.index.n_neurons)
    rates[0] = np.nan
    rates[1] = np.inf
    state = builder.update(
        rates,
        reward=float("nan"),
        eyes={"left": {"photo": np.array([np.nan, -np.inf])}},
        dn_rates=np.array([np.inf, 0.0]),
        info={"player_hp": float("nan")},
    )
    text = LiveStateBuilder.to_json(state)  # allow_nan=False: raises if any slipped through
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["eyes"]["left"]["photo"] == [0.0, 0.0]


def test_modulation_depth_separates_a_constant_from_a_varying_stage():
    """The headline finding in docs/results.md is a stage with activity but no
    variation. A panel that only showed the mean could not tell the two apart."""
    builder = _builder()
    rng = np.random.default_rng(0)
    optic = builder.index.stages["optic"]
    descending = builder.index.stages["descending"]
    for _ in range(50):
        rates = np.zeros(builder.index.n_neurons)
        rates[optic] = rng.normal(5.0, 2.0, size=len(optic))
        rates[descending] = 5.0  # active, but carrying nothing
        state = builder.update(rates)
    rows = {s["name"]: s for s in state["stages"]}
    assert rows["descending"]["mean"] == pytest.approx(5.0, abs=0.1)
    assert rows["descending"]["modulation"] < 1e-6
    assert rows["optic"]["modulation"] > 0.1


def test_threat_gauge_normalises_against_a_decaying_peak():
    builder = _builder()
    rates = np.zeros(builder.index.n_neurons)
    rates[builder.index.types["LC4"]] = 10.0
    spike = builder.update(rates)
    assert spike["threat_norm"] == pytest.approx(1.0)

    rates[builder.index.types["LC4"]] = 1.0
    quiet = builder.update(rates)
    assert 0.0 < quiet["threat_norm"] < 0.5, "the needle must fall back after a spike"


def test_index_is_built_once_and_indexing_is_cheap():
    """The whole reason this module exists: a per-frame mask rebuild over an
    object array costs milliseconds, an index array costs microseconds."""
    import time

    from flyds1.connectome.loader import load_connectome

    connectome = load_connectome("synthetic:rings=3")
    idx = PopulationIndex.from_neurons(connectome.neurons)
    rates = np.random.default_rng(0).random(idx.n_neurons)

    start = time.perf_counter()
    for _ in range(20):
        for name in idx.stages:
            idx.stage_stats(rates, name)
        for name in idx.types:
            idx.mean_of(rates, idx.types[name])
        idx.balance(rates)
    per_frame = (time.perf_counter() - start) / 20

    assert per_frame < 0.005, f"{per_frame * 1e3:.2f} ms per frame is too much of a 33 ms budget"


def test_a_real_synthetic_network_populates_the_panels():
    """The groups have to match what the synthetic generator actually names its
    cells, or every panel is empty against the default config."""
    from flyds1.connectome.loader import load_connectome

    connectome = load_connectome("synthetic:rings=3")
    idx = PopulationIndex.from_neurons(connectome.neurons)

    assert {"left", "right"} <= set(idx.sides)
    assert {"sensory", "optic", "descending"} <= set(idx.stages)
    for name in ("T4a", "T4b", "T4c", "T4d", "T5a", "LC4", "LPLC2", "HS", "VS", "Mi1", "Mi9"):
        assert name in idx.types, f"{name} missing: {idx.describe()}"
    by_panel = {idx.type_meta[n].panel for n in idx.types}
    assert {"motion_on", "motion_off", "threat", "flow"} <= by_panel


# ----------------------------------------------------------------------
def test_panel_publishes_layout_state_and_assets(tmp_path):
    from flyds1.viz.panel import ASSETS, PanelView

    view = PanelView(tmp_path, refresh_ms=77)
    builder = _builder()
    view.publish_layout(builder.layout())
    view.update(builder.update(np.zeros(builder.index.n_neurons)),
                frame=np.zeros((20, 30, 3), dtype=np.uint8))

    for name in ASSETS:
        assert (tmp_path / name).exists()
    assert "77" in (tmp_path / "index.html").read_text()
    assert json.loads((tmp_path / "layout.json").read_text())["n_neurons"] == 11
    assert json.loads((tmp_path / "state.json").read_text())["step"] == 0
    assert (tmp_path / "frame.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not list(tmp_path.glob(".*tmp*")), "no half-written temporaries left behind"


def test_panel_state_is_valid_before_the_first_update(tmp_path):
    """The page polls immediately; a missing state.json is a console error on
    every tick and nothing to show for it."""
    from flyds1.viz.panel import PanelView

    PanelView(tmp_path)
    assert json.loads((tmp_path / "state.json").read_text()) == {"waiting": True}


def test_panel_publishes_the_frame_less_often_than_the_state(tmp_path):
    """The PNG encode is the only expensive part of an update, so it runs at a
    fraction of the state rate -- deliberately, not by accident."""
    from flyds1.viz.panel import PanelView

    view = PanelView(tmp_path, frame_every=3)
    builder = _builder()
    state = builder.update(np.zeros(builder.index.n_neurons))

    written = []
    frames = [np.full((8, 8, 3), v, dtype=np.uint8) for v in (10, 20, 30, 40)]
    for frame in frames:
        view.update(state, frame=frame)
        written.append((tmp_path / "frame.png").stat().st_mtime_ns)
    # steps 0 and 3 publish; 1 and 2 do not
    assert written[0] == written[1] == written[2]
    assert written[3] != written[0]


def test_panel_downscales_a_large_frame(tmp_path):
    from flyds1.viz.panel import PanelView

    small = PanelView(tmp_path / "a", frame_max_width=64)
    large = PanelView(tmp_path / "b", frame_max_width=10_000)
    frame = np.random.default_rng(0).integers(0, 255, (540, 960, 3), dtype=np.uint8)
    builder = _builder()
    state = builder.update(np.zeros(builder.index.n_neurons))
    small.update(state, frame=frame)
    large.update(state, frame=frame)
    assert (tmp_path / "a" / "frame.png").stat().st_size < \
           (tmp_path / "b" / "frame.png").stat().st_size


def test_panel_serves_every_file_the_page_needs(tmp_path):
    import urllib.request

    from flyds1.viz.panel import PanelView

    view = PanelView(tmp_path)
    builder = _builder()
    view.publish_layout(builder.layout())
    view.update(builder.update(np.zeros(builder.index.n_neurons)),
                frame=np.zeros((8, 8), dtype=np.uint8))
    url = view.serve(port=0)
    try:
        for name in ("", "fly.css", "fly.js", "layout.json", "state.json", "frame.png"):
            with urllib.request.urlopen(url + name, timeout=5) as response:
                assert response.status == 200
                # a cached state.json freezes the panel while the run continues,
                # which reads as a hung agent
                assert "no-store" in response.headers.get("Cache-Control", "")
                assert response.read()
    finally:
        view.stop()


def test_decoder_weight_is_none_without_a_policy():
    from flyds1.cli import _decoder_weight

    assert _decoder_weight(None, 4, 8) is None


def test_json_payload_is_smaller_than_the_png_it_replaces(tmp_path):
    """The claim in the panel's docstring, checked: a richer viewer that costs
    more per step would not be worth having."""
    from flyds1.connectome.loader import load_connectome
    from flyds1.plotting import bar_chart, encode_png, facet_image, filmstrip, to_uint8

    connectome = load_connectome("synthetic:rings=6")
    idx = PopulationIndex.from_neurons(connectome.neurons)
    photo = connectome.neurons.photoreceptor_mask()
    coords = connectome.neurons.eye_coord[photo]
    rng = np.random.default_rng(0)
    rates = rng.random(idx.n_neurons)

    builder = LiveStateBuilder(idx, eye_coords={"left": coords[: len(coords) // 2],
                                               "right": coords[len(coords) // 2:]})
    state = builder.update(
        rates,
        eyes={"left": {"photo": rng.normal(size=len(coords) // 2)},
              "right": {"photo": rng.normal(size=len(coords) - len(coords) // 2)}},
        dn_rates=rates[idx.stages["descending"]],
    )
    json_bytes = len(LiveStateBuilder.to_json(state).encode())

    classic = encode_png(filmstrip([
        to_uint8(rng.random((180, 320))),
        facet_image(coords, rng.normal(size=len(coords)), signed=True),
        facet_image(coords, rng.normal(size=len(coords)), signed=True),
        bar_chart(rates[idx.stages["descending"]], width=260, height=180),
    ]))
    assert json_bytes < len(classic), f"json {json_bytes} vs png {len(classic)}"
