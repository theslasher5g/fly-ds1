"""Which neurons belong to which population -- resolved once, not per frame.

The viewer wants to show things like "how active is the left optic lobe", "what
are the T4 subtypes doing", "how much signal survives to the descending
neurons".  Every one of those questions is a mean over a subset of the rate
vector, and the subset never changes: it follows from the neuron table's
``side``, ``super_class`` and ``cell_type`` columns, which are fixed once the
network is wired.

So resolve every subset to an index array at startup and keep it.  With ~27 000
neurons, ``rates[index].mean()`` on a precomputed index is tens of
microseconds; rebuilding the mask each frame (``[ct == "t4a" for ct in ...]``
over an object array) is milliseconds, and there are dozens of groups.  Inside
a 33 ms step budget that difference is the whole feasibility of the panel.

Nothing here knows about rendering or JSON -- it is the index layer only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyds1.connectome.schema import NeuronTable

#: Super-classes in signal-flow order: eye -> optic lobe -> projection
#: neurons -> central brain -> descending -> motor.  The order is the point:
#: read the population-flow panel left to right and you are reading the path a
#: stimulus has to survive to reach a key press.  ``ascending`` and ``motor``
#: sit at the end because they are feedback/output side-channels, not part of
#: that chain.
STAGE_ORDER: tuple[str, ...] = (
    "sensory",
    "optic",
    "visual_projection",
    "central",
    "descending",
    "ascending",
    "motor",
)

#: Sides in display order, left to right, as the neuron table spells them.
SIDE_ORDER: tuple[str, ...] = ("left", "center", "right")


@dataclass(frozen=True)
class TypeGroup:
    """A named set of cell types, and how to match them.

    ``mode="exact"`` compares the whole ``cell_type`` string; ``mode="prefix"``
    matches any type starting with the pattern.  Both are needed because real
    annotation is inconsistent in a specific way: ``T4a`` is one cell type, but
    the horizontal-system tangential cells are ``HSN``/``HSE``/``HSS`` and the
    vertical ones ``VS1``..``VS10`` -- one functional group spelled as several
    types.  Substring matching (what :meth:`NeuronTable.mask_cell_type` does)
    would be the wrong tool here: ``"vs"`` as a substring also matches anything
    that merely contains those letters.
    """

    name: str
    patterns: tuple[str, ...]
    mode: str = "exact"
    panel: str = "other"
    #: Free-text note the viewer can show as a tooltip -- what the cell does,
    #: so a panel is readable by someone who does not know fly anatomy.
    about: str = ""

    def matches(self, cell_type: np.ndarray) -> np.ndarray:
        pats = tuple(p.lower() for p in self.patterns)
        if self.mode == "exact":
            return np.array([str(ct) in pats for ct in cell_type], dtype=bool)
        if self.mode == "prefix":
            return np.array([str(ct).startswith(pats) for ct in cell_type], dtype=bool)
        raise ValueError(f"unknown match mode {self.mode!r}")


def _dir(sub: str) -> str:
    return {"a": "front-to-back", "b": "back-to-front", "c": "upward", "d": "downward"}[sub]


#: The groups the viewer draws.  This is the fly's visual motion pathway, in the
#: order a signal traverses it, and it is data rather than code so that adding a
#: panel is a list entry.
DEFAULT_TYPE_GROUPS: tuple[TypeGroup, ...] = (
    # Photoreceptors and the first two lamina relays: the input and the
    # ON/OFF split that everything downstream is built on.
    TypeGroup("R1-6", ("r1-6", "r16", "r1", "r2", "r3", "r4", "r5", "r6"), panel="input",
              about="outer photoreceptors, motion vision's input"),
    TypeGroup("L1", ("l1",), panel="input", about="lamina relay feeding the ON pathway"),
    TypeGroup("L2", ("l2",), panel="input", about="lamina relay feeding the OFF pathway"),
    # The elementary motion detectors.  Four preferred directions, twice:
    # T4 for ON (brightening) edges, T5 for OFF (darkening) edges.
    *[
        TypeGroup(f"T4{s}", (f"t4{s}",), panel="motion_on",
                  about=f"ON-edge motion, prefers {_dir(s)}")
        for s in "abcd"
    ],
    *[
        TypeGroup(f"T5{s}", (f"t5{s}",), panel="motion_off",
                  about=f"OFF-edge motion, prefers {_dir(s)}")
        for s in "abcd"
    ],
    # The offset motif that makes T4/T5 directional in the first place.
    TypeGroup("Mi1", ("mi1",), panel="medulla", about="ON, centre, excitatory"),
    TypeGroup("Mi4", ("mi4",), panel="medulla", about="ON, delayed inhibition, trailing side"),
    TypeGroup("Mi9", ("mi9",), panel="medulla", about="ON, inhibition, leading side"),
    TypeGroup("Tm1", ("tm1",), panel="medulla", about="OFF pathway relay"),
    TypeGroup("CT1", ("ct1",), panel="medulla", about="wide-field inhibition"),
    # Looming detectors.  In a real fly these drive escape; if this agent ever
    # learns to dodge, these move first.
    TypeGroup("LC4", ("lc4",), panel="threat", about="looming: approaching object, escape"),
    TypeGroup("LPLC2", ("lplc2",), panel="threat", about="looming: expanding edges, escape"),
    # Wide-field optic flow: am I turning, am I rising.
    TypeGroup("HS", ("hs",), mode="prefix", panel="flow",
              about="horizontal-system tangential cells: yaw optic flow"),
    TypeGroup("VS", ("vs",), mode="prefix", panel="flow",
              about="vertical-system tangential cells: pitch/roll optic flow"),
)


@dataclass
class PopulationIndex:
    """Index arrays for every population the viewer summarises.

    Built from a :class:`~flyds1.connectome.schema.NeuronTable` in matrix-index
    order -- i.e. the same order as the rate vector the simulation produces, so
    every array here indexes straight into it.
    """

    sides: dict[str, np.ndarray] = field(default_factory=dict)
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    types: dict[str, np.ndarray] = field(default_factory=dict)
    #: ``(stage, side) -> index``.  This is what makes "which hemisphere" a
    #: per-stage question instead of a whole-brain one, which matters: a left/
    #: right difference that exists in the optic lobe and is gone by the
    #: descending neurons is the project's known failure, and averaging over
    #: the whole brain hides exactly that.
    stage_sides: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    type_meta: dict[str, TypeGroup] = field(default_factory=dict)
    #: Requested type groups with no members in this connectome.  Not an error
    #: (the synthetic generator does not produce every cell type, and neither
    #: does a partially annotated real dataset) but the viewer should know, so
    #: an empty panel reads as "not in this dataset" and not as "silent".
    missing: tuple[str, ...] = ()
    n_neurons: int = 0

    @classmethod
    def from_neurons(
        cls,
        neurons: NeuronTable,
        *,
        type_groups: tuple[TypeGroup, ...] = DEFAULT_TYPE_GROUPS,
    ) -> "PopulationIndex":
        n = len(neurons)
        side = np.asarray(neurons.side, dtype=object)
        super_class = np.asarray(neurons.super_class, dtype=object)
        cell_type = np.asarray(neurons.cell_type, dtype=object)

        sides: dict[str, np.ndarray] = {}
        for name in SIDE_ORDER:
            idx = np.flatnonzero(np.array([str(s) == name for s in side], dtype=bool))
            if len(idx):
                sides[name] = idx.astype(np.int64)

        stages: dict[str, np.ndarray] = {}
        # Known stages first, in flow order; then anything else the dataset
        # happens to carry, so an unexpected super_class is visible rather than
        # silently dropped from every total.
        seen = set()
        for name in STAGE_ORDER:
            idx = np.flatnonzero(np.array([str(sc) == name for sc in super_class], dtype=bool))
            seen.add(name)
            if len(idx):
                stages[name] = idx.astype(np.int64)
        for name in sorted({str(sc) for sc in super_class} - seen):
            idx = np.flatnonzero(np.array([str(sc) == name for sc in super_class], dtype=bool))
            stages[name] = idx.astype(np.int64)

        stage_sides: dict[tuple[str, str], np.ndarray] = {}
        for stage, stage_idx in stages.items():
            for side_name, side_idx in sides.items():
                both = np.intersect1d(stage_idx, side_idx, assume_unique=True)
                if len(both):
                    stage_sides[(stage, side_name)] = both.astype(np.int64)

        types: dict[str, np.ndarray] = {}
        meta: dict[str, TypeGroup] = {}
        missing: list[str] = []
        for group in type_groups:
            idx = np.flatnonzero(group.matches(cell_type))
            if len(idx) == 0:
                missing.append(group.name)
                continue
            types[group.name] = idx.astype(np.int64)
            meta[group.name] = group

        return cls(
            sides=sides,
            stages=stages,
            types=types,
            stage_sides=stage_sides,
            type_meta=meta,
            missing=tuple(missing),
            n_neurons=n,
        )

    # ------------------------------------------------------------------
    def mean_of(self, rates: np.ndarray, index: np.ndarray) -> float:
        """Mean rate over ``index``; 0.0 for an empty group."""
        if len(index) == 0:
            return 0.0
        return float(np.mean(rates[index]))

    def balance(self, rates: np.ndarray, stage: str | None = None) -> float:
        """Right-minus-left activity, normalised to ``-1 .. +1``.

        ``stage=None`` uses the whole brain; a stage name restricts it to that
        super-class.  Returns 0.0 when both sides are silent, which is the
        honest answer -- a silent network has no lateral preference, and a bare
        difference of zeros must not be reported as "balanced" by accident of
        arithmetic.

        The ``eps`` in the denominator is deliberately small enough not to
        distort real values and large enough that two near-zero means cannot
        produce a swing to +-1 out of numerical noise.
        """
        if stage is None:
            left = self.sides.get("left", _EMPTY)
            right = self.sides.get("right", _EMPTY)
        else:
            left = self.stage_sides.get((stage, "left"), _EMPTY)
            right = self.stage_sides.get((stage, "right"), _EMPTY)
        lm = self.mean_of(rates, left)
        rm = self.mean_of(rates, right)
        total = lm + rm
        if total <= 1e-9:
            return 0.0
        return float((rm - lm) / total)

    def stage_stats(self, rates: np.ndarray, stage: str) -> tuple[float, float, float]:
        """``(mean, peak, active_fraction)`` for one stage."""
        idx = self.stages.get(stage, _EMPTY)
        if len(idx) == 0:
            return 0.0, 0.0, 0.0
        r = rates[idx]
        return float(np.mean(r)), float(np.max(r)), float(np.mean(r > 1e-6))

    def describe(self) -> str:
        """One block of text for ``flyds1 info`` -- what the viewer can show."""
        rows = [f"PopulationIndex over {self.n_neurons} neurons"]
        rows.append("  sides:  " + ", ".join(f"{k}={len(v)}" for k, v in self.sides.items()))
        rows.append("  stages: " + ", ".join(f"{k}={len(v)}" for k, v in self.stages.items()))
        by_panel: dict[str, list[str]] = {}
        for name, idx in self.types.items():
            by_panel.setdefault(self.type_meta[name].panel, []).append(f"{name}={len(idx)}")
        for panel, entries in sorted(by_panel.items()):
            rows.append(f"  {panel:10s} " + ", ".join(entries))
        if self.missing:
            rows.append("  not in this dataset: " + ", ".join(self.missing))
        return "\n".join(rows)


_EMPTY = np.zeros(0, dtype=np.int64)


__all__ = [
    "DEFAULT_TYPE_GROUPS",
    "SIDE_ORDER",
    "STAGE_ORDER",
    "PopulationIndex",
    "TypeGroup",
]
