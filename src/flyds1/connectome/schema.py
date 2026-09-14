"""Data model for a connectome, independent of where it came from.

Every ingestion path (FlyWire/Codex CSV dumps, ``navis``/``fafbseg`` queries,
the synthetic generator) produces the same two tables:

* :class:`NeuronTable` -- one row per neuron: id, annotations, neurotransmitter,
  soma position, and (for photoreceptors) the eye coordinate of its ommatidium.
* :class:`SynapseTable` -- one row per connected pair: pre id, post id, synapse
  count.

Downstream code (``flyds1.connectome.graph``) only ever sees these two tables,
so swapping FAFB for the male CNS volume is a change of loader, not of model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Neurotransmitter -> sign of every outgoing synapse (Dale's law).
#
# Signs follow the convention used by connectome-constrained models of the fly
# visual system (Lappalainen et al., "flyvis"): cholinergic excitatory, GABA
# and glutamate inhibitory.  Glutamate is the contentious one -- in Drosophila
# it is frequently inhibitory via the GluCl-alpha chloride channel, which is
# why it is negative here -- so the mapping is data, not hard-coded logic, and
# can be overridden per run (see ``configs/default.yaml: connectome.nt_signs``).
# Modulatory transmitters default to 0.0: they are in the tables but a rate
# model has no mechanism for them, and giving them a sign would be invention.
# ---------------------------------------------------------------------------
NT_SIGNS: Mapping[str, float] = {
    "acetylcholine": +1.0,
    "ach": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "glut": -1.0,
    "dopamine": 0.0,
    "da": 0.0,
    "serotonin": 0.0,
    "5ht": 0.0,
    "octopamine": 0.0,
    "oa": 0.0,
    "tyramine": 0.0,
    "histamine": -1.0,  # photoreceptor transmitter, inhibits L1/L2
    "unknown": 0.0,
    "": 0.0,
}

#: Super-classes we care about, spelled as FlyWire/Codex spells them.
SUPERCLASS_SENSORY = "sensory"
SUPERCLASS_DESCENDING = "descending"
SUPERCLASS_ASCENDING = "ascending"
SUPERCLASS_CENTRAL = "central"
SUPERCLASS_MOTOR = "motor"
SUPERCLASS_VISUAL_PROJECTION = "visual_projection"
SUPERCLASS_OPTIC = "optic"


@dataclass
class NeuronTable:
    """One row per neuron.

    Attributes
    ----------
    ids:
        Root ids (int64).  Opaque and possibly huge -- never used as an index,
        always mapped through :meth:`index_of`.
    cell_type:
        Free-form type label, e.g. ``"R1-6"``, ``"L1"``, ``"T4a"``, ``"DNa02"``.
    super_class:
        Coarse class, e.g. ``"sensory"``, ``"optic"``, ``"central"``,
        ``"descending"``.
    nt_type:
        Predicted neurotransmitter, lower-case, keys of :data:`NT_SIGNS`.
    side:
        ``"left"`` / ``"right"`` / ``"center"``.
    eye_coord:
        ``(n, 2)`` hex lattice coordinate of the ommatidium a photoreceptor
        belongs to, ``NaN`` for every non-photoreceptor.  This is what step 3
        needs to map screen pixels onto the retina.
    """

    ids: np.ndarray
    cell_type: np.ndarray
    super_class: np.ndarray
    nt_type: np.ndarray
    side: np.ndarray
    eye_coord: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids, dtype=np.int64)
        n = len(self.ids)
        for name in ("cell_type", "super_class", "nt_type", "side"):
            arr = np.asarray(getattr(self, name), dtype=object)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
            setattr(self, name, np.array([str(v).strip().lower() for v in arr], dtype=object))
        self.eye_coord = np.asarray(self.eye_coord, dtype=float)
        if self.eye_coord.shape != (n, 2):
            raise ValueError(f"eye_coord has shape {self.eye_coord.shape}, expected ({n}, 2)")
        if len(np.unique(self.ids)) != n:
            raise ValueError("neuron ids are not unique")
        self._index = {int(i): k for k, i in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def index_of(self, ids: Iterable[int]) -> np.ndarray:
        """Map root ids to row indices, raising on unknown ids."""
        out = np.empty(len(list(ids)) if not hasattr(ids, "__len__") else len(ids), dtype=np.int64)
        for k, i in enumerate(ids):
            try:
                out[k] = self._index[int(i)]
            except KeyError as exc:  # pragma: no cover - defensive
                raise KeyError(f"unknown neuron id {i}") from exc
        return out

    def signs(self, nt_signs: Mapping[str, float] | None = None) -> np.ndarray:
        """Per-neuron output sign from its predicted neurotransmitter."""
        table = dict(NT_SIGNS if nt_signs is None else nt_signs)
        return np.array([float(table.get(str(nt), 0.0)) for nt in self.nt_type], dtype=float)

    def mask_super_class(self, *names: str) -> np.ndarray:
        wanted = {n.lower() for n in names}
        return np.array([sc in wanted for sc in self.super_class], dtype=bool)

    def mask_cell_type(self, *patterns: str) -> np.ndarray:
        """Substring match on ``cell_type`` (case-insensitive)."""
        pats = [p.lower() for p in patterns]
        return np.array([any(p in ct for p in pats) for ct in self.cell_type], dtype=bool)

    def photoreceptor_mask(self) -> np.ndarray:
        """Photoreceptors: sensory neurons that carry an eye coordinate."""
        return self.mask_super_class(SUPERCLASS_SENSORY) & np.isfinite(self.eye_coord).all(axis=1)

    def descending_mask(self) -> np.ndarray:
        """Descending neurons -- the brain's only motor output channel.

        Annotated as a super-class in FlyWire/Codex; the ``dn`` cell-type
        fallback catches datasets where only the type label is filled in.
        """
        return self.mask_super_class(SUPERCLASS_DESCENDING) | self.mask_cell_type("dn")

    def select(self, mask: np.ndarray) -> "NeuronTable":
        mask = np.asarray(mask, dtype=bool)
        return NeuronTable(
            ids=self.ids[mask],
            cell_type=self.cell_type[mask],
            super_class=self.super_class[mask],
            nt_type=self.nt_type[mask],
            side=self.side[mask],
            eye_coord=self.eye_coord[mask],
            extra={k: v[mask] for k, v in self.extra.items()},
        )


@dataclass
class SynapseTable:
    """Aggregated connectivity: ``pre_ids[k] --(count[k])--> post_ids[k]``."""

    pre_ids: np.ndarray
    post_ids: np.ndarray
    counts: np.ndarray

    def __post_init__(self) -> None:
        self.pre_ids = np.asarray(self.pre_ids, dtype=np.int64)
        self.post_ids = np.asarray(self.post_ids, dtype=np.int64)
        self.counts = np.asarray(self.counts, dtype=float)
        if not (len(self.pre_ids) == len(self.post_ids) == len(self.counts)):
            raise ValueError("pre_ids, post_ids and counts must have equal length")
        if np.any(self.counts < 0):
            raise ValueError("synapse counts must be non-negative")

    def __len__(self) -> int:
        return len(self.counts)

    def filter_min_count(self, min_count: float) -> "SynapseTable":
        """Drop weak edges -- the standard first defence against reconstruction
        false positives (Codex itself defaults to a threshold of 5)."""
        keep = self.counts >= min_count
        return SynapseTable(self.pre_ids[keep], self.post_ids[keep], self.counts[keep])

    def restrict_to(self, ids: Sequence[int]) -> "SynapseTable":
        known = set(int(i) for i in ids)
        keep = np.array(
            [(int(a) in known and int(b) in known) for a, b in zip(self.pre_ids, self.post_ids)],
            dtype=bool,
        )
        if len(keep) == 0:
            keep = np.zeros(0, dtype=bool)
        return SynapseTable(self.pre_ids[keep], self.post_ids[keep], self.counts[keep])


@dataclass
class Connectome:
    """A neuron table plus a synapse table, with provenance attached."""

    neurons: NeuronTable
    synapses: SynapseTable
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        known = set(int(i) for i in self.neurons.ids)
        dangling = {
            int(i)
            for i in np.concatenate([self.synapses.pre_ids, self.synapses.post_ids])
            if int(i) not in known
        } if len(self.synapses) else set()
        if dangling:
            raise ValueError(
                f"{len(dangling)} synapse endpoints are missing from the neuron table, "
                f"e.g. {sorted(dangling)[:3]}"
            )

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return len(self.synapses)

    def summary(self) -> str:
        n = self.neurons
        counts = {
            "photoreceptors": int(n.photoreceptor_mask().sum()),
            "descending": int(n.descending_mask().sum()),
        }
        by_class: dict[str, int] = {}
        for sc in n.super_class:
            by_class[sc] = by_class.get(sc, 0) + 1
        top = ", ".join(f"{k}={v}" for k, v in sorted(by_class.items(), key=lambda kv: -kv[1])[:6])
        return (
            f"Connectome(source={self.source!r}, neurons={self.n_neurons}, "
            f"edges={self.n_edges}, synapses={self.synapses.counts.sum():.0f})\n"
            f"  super_class: {top}\n"
            f"  photoreceptors={counts['photoreceptors']}, descending={counts['descending']}"
        )
