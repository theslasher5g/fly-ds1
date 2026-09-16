"""One frame of brain state, as numbers a browser can draw.

The old viewer rasterised panels server-side into a PNG filmstrip inside the
33 ms step budget.  This module does the opposite: it reduces the rate vector to
a few hundred floats and lets the browser draw them.  Two consequences worth
being explicit about, because they are why the richer panel is also the cheaper
one:

* **Geometry is static, so it is sent once.**  Facet coordinates, cell-type
  groups, descending-neuron labels and the decoder's weight matrix do not change
  while a policy plays.  :meth:`LiveStateBuilder.layout` emits them at startup;
  :meth:`LiveStateBuilder.update` then only ever sends values.
* **History lives in the browser.**  The page polls every frame anyway, so it
  can accumulate its own sparklines.  Re-sending a 300-sample window 30 times a
  second would cost more than everything else here combined.  The one exception
  is modulation depth, which *is* a statistic over a window and therefore has to
  be computed where the window is kept -- here.

Nothing in this module touches the environment or the network.  It reads arrays
it is handed, which keeps the guarantee the live viewer has always had: watching
cannot change what the agent does.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from flyds1.viz.groups import PopulationIndex

#: Sent to the browser as -1..1 or 0..1 floats; three decimals is well past
#: what a canvas can show and keeps the payload small.
_ND = 3


def _f(value: float) -> float:
    """A JSON-safe float.

    ``NaN`` and ``Infinity`` are not valid JSON.  Python's ``json`` emits them
    anyway (as bare ``NaN``), and ``JSON.parse`` in a browser then throws --
    which blanks the whole panel, so one bad neuron reads as "the fly is dead".
    Clamp them to 0.0 at the boundary instead, and test for it.
    """
    v = float(value)
    if not math.isfinite(v):
        return 0.0
    return round(v, _ND)


def _fl(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return [round(float(v), _ND) for v in arr]


@dataclass
class _Window:
    """Rolling window of a scalar, for standard deviation only."""

    size: int
    values: deque = field(init=False)

    def __post_init__(self) -> None:
        self.values = deque(maxlen=max(2, int(self.size)))

    def push(self, value: float) -> None:
        self.values.append(float(value) if math.isfinite(float(value)) else 0.0)

    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return float(np.std(np.fromiter(self.values, dtype=float)))


@dataclass
class LiveStateBuilder:
    """Turns rates plus env info into the two payloads the page reads.

    Parameters
    ----------
    index:
        Precomputed population groups (see :mod:`flyds1.viz.groups`).
    eye_coords:
        ``side -> (n_facets, 2)`` facet coordinates in degrees, straight from
        ``RetinaFrontEnd.eyes[side]["coords"]``.  Sent once.
    action_names:
        The action-spec button names, in action order.
    dn_labels:
        Cell-type (or id) label per descending neuron, in output order.
    decoder_weight:
        ``(n_actions, n_dn)`` decoder weights, sent once.  The DN-to-key panel
        multiplies these by the live rates *in the browser*.  Static during a
        live run because the policy is not training; a training-time viewer
        would need to re-emit the layout, which is deliberately not supported
        here rather than silently showing stale wiring.
    window:
        Samples used for the modulation-depth statistic.  The default is ~10 s
        at 30 fps, long enough to span a boss's attack cycle.
    """

    index: PopulationIndex
    eye_coords: dict[str, np.ndarray] = field(default_factory=dict)
    action_names: tuple[str, ...] = ()
    dn_labels: tuple[str, ...] = ()
    decoder_weight: np.ndarray | None = None
    window: int = 300

    def __post_init__(self) -> None:
        self._stage_windows = {name: _Window(self.window) for name in self.index.stages}
        self._threat_peak = 1e-6

    # ------------------------------------------------------------------
    def layout(self) -> dict:
        """Static description of what the panels contain.  Sent once."""
        stages = [
            {"name": name, "n": int(len(idx)),
             "left": int(len(self.index.stage_sides.get((name, "left"), ()))),
             "right": int(len(self.index.stage_sides.get((name, "right"), ())))}
            for name, idx in self.index.stages.items()
        ]
        types = [
            {"name": name, "n": int(len(idx)),
             "panel": self.index.type_meta[name].panel,
             "about": self.index.type_meta[name].about}
            for name, idx in self.index.types.items()
        ]
        out = {
            "n_neurons": int(self.index.n_neurons),
            "stages": stages,
            "types": types,
            "missing_types": list(self.index.missing),
            "sides": {k: int(len(v)) for k, v in self.index.sides.items()},
            "eyes": {
                side: {"coords": [[round(float(x), 2), round(float(y), 2)] for x, y in coords]}
                for side, coords in self.eye_coords.items()
            },
            "actions": list(self.action_names),
            "dn_labels": list(self.dn_labels),
        }
        if self.decoder_weight is not None:
            w = np.asarray(self.decoder_weight, dtype=float)
            out["decoder"] = {
                "shape": [int(w.shape[0]), int(w.shape[1])],
                "weight": [_fl(row) for row in w],
            }
        return out

    # ------------------------------------------------------------------
    def update(
        self,
        rates: np.ndarray,
        *,
        step: int = 0,
        episode: int = 0,
        reward: float = 0.0,
        total_reward: float = 0.0,
        eyes: dict[str, dict[str, np.ndarray]] | None = None,
        dn_rates: np.ndarray | None = None,
        action: np.ndarray | None = None,
        held_keys: tuple[str, ...] = (),
        info: dict | None = None,
    ) -> dict:
        """One frame's state.  ``rates`` is the full ``(n_neurons,)`` vector."""
        rates = np.asarray(rates, dtype=float).ravel()
        if rates.shape[0] != self.index.n_neurons:
            raise ValueError(
                f"rates has {rates.shape[0]} entries, expected {self.index.n_neurons}"
            )
        info = info or {}

        stage_rows = []
        for name in self.index.stages:
            mean, peak, active = self.index.stage_stats(rates, name)
            self._stage_windows[name].push(mean)
            stage_rows.append({
                "name": name,
                "mean": _f(mean),
                "peak": _f(peak),
                "active": _f(active),
                # Modulation depth: how much this stage's activity *varies*.
                # A stage with a high mean and near-zero modulation is passing a
                # constant, i.e. carrying no information -- which is exactly the
                # decay documented in docs/results.md, and the reason this
                # number is on the panel at all.
                "modulation": _f(self._stage_windows[name].std()),
                "balance": _f(self.index.balance(rates, name)),
            })

        type_rows = {
            name: _f(self.index.mean_of(rates, idx)) for name, idx in self.index.types.items()
        }

        threat_raw = 0.0
        threat_groups = [n for n, g in self.index.type_meta.items() if g.panel == "threat"]
        if threat_groups:
            threat_raw = float(np.mean([type_rows[n] for n in threat_groups]))
        # Normalise against a slowly decaying peak so the needle is readable
        # without knowing the network's absolute rate scale, which depends on
        # gain and changes between connectomes.  The decay lets it re-scale down
        # after a spike instead of being pinned by one outlier forever.
        self._threat_peak = max(threat_raw, self._threat_peak * 0.999)

        state = {
            "step": int(step),
            "episode": int(episode),
            "reward": _f(reward),
            "total_reward": _f(total_reward),
            "stages": stage_rows,
            "types": type_rows,
            "balance": _f(self.index.balance(rates)),
            "threat": _f(threat_raw),
            "threat_norm": _f(threat_raw / self._threat_peak if self._threat_peak > 0 else 0.0),
            "held_keys": list(held_keys),
        }

        if eyes:
            state["eyes"] = {
                side: {
                    key: _fl(value)
                    for key, value in payload.items()
                    if value is not None
                }
                for side, payload in eyes.items()
            }
        if dn_rates is not None:
            state["dn"] = _fl(dn_rates)
        if action is not None:
            state["action"] = [int(bool(a)) for a in np.atleast_1d(action)]

        for key in ("state", "player_hp", "boss_hp", "attempts", "kills", "deaths",
                    "waypoint", "focused", "dry_run", "steps_per_second"):
            if key in info:
                value = info[key]
                state[key] = _f(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        return state

    @staticmethod
    def to_json(payload: dict) -> str:
        """Serialise, refusing to emit the non-finite values JSON cannot hold."""
        return json.dumps(payload, allow_nan=False, separators=(",", ":"))


__all__ = ["LiveStateBuilder"]
