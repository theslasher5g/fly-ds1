"""A synthetic stand-in for the real connectome.

Downloading FlyWire needs network access, an account, and a few GB; CI has
none of those, and neither does a laptop on a train.  This module generates a
connectome with the *structure* the rest of the pipeline relies on:

* retinotopic columns on a hex lattice, so step 3 (screen -> ommatidia) has
  real eye coordinates to map onto;
* the ON/OFF pathways and the T4/T5 input motif (Mi9- / Mi1+ / Mi4- offset by
  one column), so direction selectivity is a property of the *wiring* rather
  than something the encoder has to be taught;
* lobula plate tangential cells (HS/VS) and looming detectors (LC4/LPLC2)
  pooling those motion signals;
* a recurrent central brain;
* descending neurons as the single motor output bottleneck (step 5).

The numbers are plausible, not measured.  Anything quantitative you conclude
from a synthetic connectome is a statement about this generator, not about
Drosophila -- use it for shape, speed and tests, then swap in
``flyds1.connectome.codex``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyds1.connectome.schema import Connectome, NeuronTable, SynapseTable
from flyds1.hexlattice import HEX_AXES, hex_lattice, neighbour_pairs

#: Per-column cell types with their neurotransmitter.  Signs come from the
#: transmitter alone (Dale's law), exactly as in the real pipeline.
COLUMNAR_TYPES: dict[str, str] = {
    "r1-6": "histamine",   # outer photoreceptors, one unit per ommatidium
    "l1": "glutamate",     # ON pathway input
    "l2": "acetylcholine",  # OFF pathway input
    "l3": "glutamate",
    "mi1": "acetylcholine",  # ON, centre of the T4 motif
    "mi4": "gaba",          # ON, delayed inhibition on one side
    "mi9": "glutamate",     # ON, inhibition on the other side
    "tm1": "acetylcholine",  # OFF
    "tm2": "acetylcholine",  # OFF
    "tm9": "acetylcholine",  # OFF, sustained
    "ct1": "gaba",          # wide-field inhibition
}

#: T4 (ON) and T5 (OFF) subtypes, each tuned to one direction on the lattice.
#: ``(axis, sign)`` picks the direction: axis indexes :data:`HEX_AXES`.
DIRECTION_SUBTYPES: dict[str, tuple[int, int]] = {
    "a": (0, +1),
    "b": (0, -1),
    "c": (1, +1),
    "d": (1, -1),
}

#: Lobula plate tangential cells: wide-field integrators of one direction each.
TANGENTIAL_CELLS: dict[str, tuple[int, int]] = {
    "hsn": (0, +1),
    "hse": (0, +1),
    "hss": (0, -1),
    "vs1": (1, +1),
    "vs2": (1, -1),
    "vs3": (1, +1),
}


@dataclass
class SyntheticConfig:
    """Knobs for the generator.  ``rings`` dominates the size."""

    rings: int = 6
    spacing_deg: float = 5.0
    n_central: int = 200
    n_descending: int = 24
    n_lc4: int = 8
    n_lplc2: int = 8
    central_in_degree: int = 16
    dn_in_degree: int = 24
    feedback_edges: int = 200
    seed: int = 0

    #: Mean synapse counts per motif, roughly matching the spread seen in
    #: Codex (a handful to a few hundred per connected pair).
    w_photoreceptor: float = 45.0
    w_lamina: float = 30.0
    w_motion_centre: float = 25.0
    w_motion_flank: float = 18.0
    w_pooling: float = 8.0
    w_central: float = 12.0
    w_descending: float = 15.0


class _Builder:
    """Accumulates neurons and edges, then aggregates duplicate pairs."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.ids: list[int] = []
        self.cell_type: list[str] = []
        self.super_class: list[str] = []
        self.nt_type: list[str] = []
        self.side: list[str] = []
        self.eye_coord: list[tuple[float, float]] = []
        self._next_id = 720_575_940_600_000_000  # FlyWire-shaped root ids
        self.pre: list[int] = []
        self.post: list[int] = []
        self.counts: list[float] = []

    def add(
        self,
        cell_type: str,
        super_class: str,
        nt: str,
        side: str,
        eye_coord: tuple[float, float] | None = None,
    ) -> int:
        self._next_id += 1
        nid = self._next_id
        self.ids.append(nid)
        self.cell_type.append(cell_type)
        self.super_class.append(super_class)
        self.nt_type.append(nt)
        self.side.append(side)
        self.eye_coord.append(eye_coord if eye_coord is not None else (np.nan, np.nan))
        return nid

    def connect(self, pre: int, post: int, mean_count: float) -> None:
        """Add an edge with a lognormally jittered synapse count (>= 1)."""
        if pre == post:
            return
        count = float(np.maximum(1.0, np.round(self.rng.lognormal(np.log(mean_count), 0.5))))
        self.pre.append(pre)
        self.post.append(post)
        self.counts.append(count)

    def synapse_table(self) -> SynapseTable:
        if not self.pre:
            return SynapseTable(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
        pre = np.array(self.pre, dtype=np.int64)
        post = np.array(self.post, dtype=np.int64)
        counts = np.array(self.counts, dtype=float)
        # aggregate duplicates: one row per (pre, post)
        keys, inverse = np.unique(np.stack([pre, post], axis=1), axis=0, return_inverse=True)
        summed = np.zeros(len(keys), dtype=float)
        np.add.at(summed, inverse, counts)
        return SynapseTable(keys[:, 0], keys[:, 1], summed)

    def neuron_table(self) -> NeuronTable:
        return NeuronTable(
            ids=np.array(self.ids, dtype=np.int64),
            cell_type=np.array(self.cell_type, dtype=object),
            super_class=np.array(self.super_class, dtype=object),
            nt_type=np.array(self.nt_type, dtype=object),
            side=np.array(self.side, dtype=object),
            eye_coord=np.array(self.eye_coord, dtype=float),
        )


def _step(axial: np.ndarray, axis: int, sign: int) -> np.ndarray:
    dq, dr = HEX_AXES[axis]
    return axial + sign * np.array([dq, dr], dtype=np.int64)


def build_synthetic_connectome(cfg: SyntheticConfig | None = None) -> Connectome:
    """Generate a connectome for offline development and tests."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    b = _Builder(rng)

    axial, coords = hex_lattice(cfg.rings, cfg.spacing_deg)
    column_key = {(int(q), int(r)): k for k, (q, r) in enumerate(axial)}
    n_col = len(axial)

    # ---- optic lobes, one per eye -------------------------------------
    # columnar[side][cell_type] -> array of neuron ids, indexed by column
    columnar: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        per_type: dict[str, np.ndarray] = {}
        for ctype, nt in COLUMNAR_TYPES.items():
            is_photoreceptor = ctype == "r1-6"
            # Columnar cells are retinotopic: every one of them belongs to a
            # column and therefore looks in a definite direction.  Carrying that
            # coordinate lets step 3 inject motion column-by-column instead of
            # as a population average.
            ids = np.array(
                [
                    b.add(
                        ctype,
                        "sensory" if is_photoreceptor else "optic",
                        nt,
                        side,
                        tuple(coords[k]),
                    )
                    for k in range(n_col)
                ],
                dtype=np.int64,
            )
            per_type[ctype] = ids
        for base in ("t4", "t5"):
            for sub in DIRECTION_SUBTYPES:
                per_type[f"{base}{sub}"] = np.array(
                    [
                        b.add(f"{base}{sub}", "optic", "acetylcholine", side, tuple(coords[k]))
                        for k in range(n_col)
                    ],
                    dtype=np.int64,
                )
        columnar[side] = per_type

    # ---- feedforward retinotopic wiring ------------------------------
    for side, per_type in columnar.items():
        for k in range(n_col):
            r16 = per_type["r1-6"][k]
            for target in ("l1", "l2", "l3"):
                b.connect(r16, per_type[target][k], cfg.w_photoreceptor)
            # ON pathway
            for target in ("mi1", "mi4", "mi9"):
                b.connect(per_type["l1"][k], per_type[target][k], cfg.w_lamina)
            # OFF pathway
            for target in ("tm1", "tm2", "tm9"):
                b.connect(per_type["l2"][k], per_type[target][k], cfg.w_lamina)
            b.connect(per_type["l3"][k], per_type["tm9"][k], cfg.w_lamina)
            # wide-field inhibition sampled from the column
            b.connect(per_type["mi1"][k], per_type["ct1"][k], cfg.w_pooling)
            b.connect(per_type["tm1"][k], per_type["ct1"][k], cfg.w_pooling)

    # ---- T4/T5 direction selectivity: the offset three-input motif ----
    # Mi9 (inhibitory) on the leading side, Mi1 (excitatory) in the centre,
    # Mi4 (inhibitory) on the trailing side.  A T4 subtype tuned to motion
    # along +axis therefore reads Mi9 from the column one step *against* its
    # preferred direction and Mi4 from one step *along* it.
    for side, per_type in columnar.items():
        for sub, (axis, sign) in DIRECTION_SUBTYPES.items():
            t4 = per_type[f"t4{sub}"]
            t5 = per_type[f"t5{sub}"]
            for k, qr in enumerate(axial):
                ahead = column_key.get(tuple(int(v) for v in _step(qr, axis, sign)))
                behind = column_key.get(tuple(int(v) for v in _step(qr, axis, -sign)))
                b.connect(per_type["mi1"][k], t4[k], cfg.w_motion_centre)
                b.connect(per_type["tm1"][k], t5[k], cfg.w_motion_centre)
                if behind is not None:
                    b.connect(per_type["mi9"][behind], t4[k], cfg.w_motion_flank)
                    b.connect(per_type["tm9"][behind], t5[k], cfg.w_motion_flank)
                if ahead is not None:
                    b.connect(per_type["mi4"][ahead], t4[k], cfg.w_motion_flank)
                    b.connect(per_type["ct1"][ahead], t5[k], cfg.w_motion_flank)

    # ---- lobula plate tangential cells + looming detectors ------------
    tangential: dict[str, dict[str, int]] = {}
    projection: dict[str, dict[str, np.ndarray]] = {}
    for side, per_type in columnar.items():
        tangential[side] = {}
        for name, (axis, sign) in TANGENTIAL_CELLS.items():
            tc = b.add(name, "visual_projection", "acetylcholine", side, None)
            tangential[side][name] = tc
            sub = next(s for s, v in DIRECTION_SUBTYPES.items() if v == (axis, sign))
            for k in range(n_col):
                b.connect(per_type[f"t4{sub}"][k], tc, cfg.w_pooling)
                b.connect(per_type[f"t5{sub}"][k], tc, cfg.w_pooling)

        # LC4 / LPLC2: looming-sensitive, pool all directions over a patch.
        proj: dict[str, np.ndarray] = {}
        for name, n_cells in (("lc4", cfg.n_lc4), ("lplc2", cfg.n_lplc2)):
            ids = np.array(
                [b.add(name, "visual_projection", "acetylcholine", side, None) for _ in range(n_cells)],
                dtype=np.int64,
            )
            patches = np.array_split(rng.permutation(n_col), n_cells)
            for cell, patch in zip(ids, patches):
                for k in patch:
                    for sub in DIRECTION_SUBTYPES:
                        b.connect(per_type[f"t4{sub}"][k], cell, cfg.w_pooling)
                        b.connect(per_type[f"t5{sub}"][k], cell, cfg.w_pooling)
            proj[name] = ids
        projection[side] = proj

    # ---- recurrent central brain --------------------------------------
    nt_choices = np.array(["acetylcholine", "gaba", "glutamate"])
    nt_probs = np.array([0.6, 0.25, 0.15])
    central = np.array(
        [
            b.add(
                f"cb{i:04d}",
                "central",
                str(rng.choice(nt_choices, p=nt_probs)),
                str(rng.choice(["left", "right"])),
                None,
            )
            for i in range(cfg.n_central)
        ],
        dtype=np.int64,
    )
    for post in central:
        for pre in rng.choice(central, size=cfg.central_in_degree, replace=False):
            b.connect(int(pre), int(post), cfg.w_central)

    # visual projection neurons feed the central brain
    vpn_ids = np.concatenate(
        [
            np.array(list(tangential[side].values()), dtype=np.int64)
            for side in ("left", "right")
        ]
        + [projection[side][name] for side in ("left", "right") for name in ("lc4", "lplc2")]
    )
    for pre in vpn_ids:
        for post in rng.choice(central, size=max(4, cfg.central_in_degree // 2), replace=False):
            b.connect(int(pre), int(post), cfg.w_central)

    # ---- descending neurons: the motor bottleneck ---------------------
    descending = np.array(
        [
            b.add(f"dn{i:02d}", "descending", "acetylcholine", str(rng.choice(["left", "right"])), None)
            for i in range(cfg.n_descending)
        ],
        dtype=np.int64,
    )
    for dn in descending:
        for pre in rng.choice(central, size=cfg.dn_in_degree, replace=False):
            b.connect(int(pre), int(dn), cfg.w_descending)
        # direct looming -> escape and tangential -> steering shortcuts
        for pre in rng.choice(vpn_ids, size=4, replace=False):
            b.connect(int(pre), int(dn), cfg.w_descending)

    # ---- central feedback into the optic lobe -------------------------
    optic_targets = np.concatenate(
        [columnar[side][ct] for side in ("left", "right") for ct in ("mi1", "tm1", "ct1")]
    )
    for _ in range(cfg.feedback_edges):
        b.connect(int(rng.choice(central)), int(rng.choice(optic_targets)), cfg.w_pooling)

    neurons = b.neuron_table()
    synapses = b.synapse_table()
    return Connectome(
        neurons=neurons,
        synapses=synapses,
        source="synthetic",
        meta={
            "config": cfg.__dict__.copy(),
            "n_columns": int(n_col),
            "spacing_deg": cfg.spacing_deg,
            "warning": "synthetic wiring -- plausible structure, invented numbers",
        },
    )


def synthetic_neighbour_pairs(cfg: SyntheticConfig | None = None) -> dict[int, np.ndarray]:
    """Neighbour pairs of the lattice used by :func:`build_synthetic_connectome`."""
    cfg = cfg or SyntheticConfig()
    axial, _ = hex_lattice(cfg.rings, cfg.spacing_deg)
    return neighbour_pairs(axial)


__all__ = [
    "COLUMNAR_TYPES",
    "DIRECTION_SUBTYPES",
    "TANGENTIAL_CELLS",
    "SyntheticConfig",
    "build_synthetic_connectome",
    "synthetic_neighbour_pairs",
]
