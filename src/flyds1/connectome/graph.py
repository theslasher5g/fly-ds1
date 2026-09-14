"""Step 2a: connectome tables -> a signed, weighted sparse adjacency matrix.

The convention used everywhere downstream is

    W[post, pre] = sign(nt_type[pre]) * synapse_count(pre -> post) / norm

so a matrix-vector product ``W @ r`` is "total synaptic input each neuron
receives".  The sign depends on the **presynaptic** neuron only, which is
Dale's law: a neuron is excitatory or inhibitory, not both.  That makes the
sign pattern a column property of ``W`` and keeps it fixed during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import scipy.sparse as sp

from flyds1.connectome.schema import Connectome, NeuronTable, SynapseTable

Normalisation = Literal["none", "in_degree", "sqrt_in_degree", "global_max", "spectral"]


@dataclass
class WiredNetwork:
    """A connectome ready to be simulated.

    Attributes
    ----------
    W:
        ``(n, n)`` CSR matrix, ``W[post, pre]``, signed and scaled.
    neurons:
        The neuron table in matrix-index order.
    input_index:
        Indices of photoreceptors -- where step 3 injects the screen.
    output_index:
        Indices of descending neurons -- what step 5 decodes into keys.
    tau:
        ``(n,)`` membrane time constants in seconds.
    """

    W: sp.csr_matrix
    neurons: NeuronTable
    input_index: np.ndarray
    output_index: np.ndarray
    tau: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.neurons)
        if self.W.shape != (n, n):
            raise ValueError(f"W has shape {self.W.shape}, expected ({n}, {n})")
        self.input_index = np.asarray(self.input_index, dtype=np.int64)
        self.output_index = np.asarray(self.output_index, dtype=np.int64)
        self.tau = np.asarray(self.tau, dtype=float)
        if self.tau.shape != (n,):
            raise ValueError(f"tau has shape {self.tau.shape}, expected ({n},)")
        if np.any(self.tau <= 0):
            raise ValueError("time constants must be positive")

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_inputs(self) -> int:
        return len(self.input_index)

    @property
    def n_outputs(self) -> int:
        return len(self.output_index)

    def eye_coords(self) -> np.ndarray:
        """``(n_inputs, 2)`` eye coordinates of the photoreceptors, in degrees."""
        return self.neurons.eye_coord[self.input_index]

    def spectral_radius(self, k: int = 1) -> float:
        """Largest ``|eigenvalue|`` of ``W`` -- the stability knob of the
        linearised dynamics.  Above 1 the network can run away."""
        n = self.n_neurons
        if n < 3:
            return float(np.abs(self.W.toarray()).max() if n else 0.0)
        try:
            vals = sp.linalg.eigs(
                self.W.astype(float), k=min(k, n - 2), return_eigenvectors=False, maxiter=5000
            )
            return float(np.max(np.abs(vals)))
        except Exception:  # pragma: no cover - ARPACK convergence
            dense = self.W.toarray()
            return float(np.max(np.abs(np.linalg.eigvals(dense))))

    def describe(self) -> str:
        nnz = self.W.nnz
        w = np.abs(self.W.data)
        return (
            f"WiredNetwork(n={self.n_neurons}, edges={nnz}, "
            f"density={nnz / max(1, self.n_neurons ** 2):.2e})\n"
            f"  |w|: mean={w.mean():.4f} max={w.max():.4f}\n"
            f"  excitatory edges={int((self.W.data > 0).sum())}, "
            f"inhibitory={int((self.W.data < 0).sum())}\n"
            f"  inputs (photoreceptors)={self.n_inputs}, outputs (DNs)={self.n_outputs}"
        )

    def save(self, path: str | Path) -> Path:
        """Save to a single ``.npz`` (sparse matrix + tables + metadata)."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        coo = self.W.tocoo()
        np.savez_compressed(
            path,
            row=coo.row,
            col=coo.col,
            data=coo.data,
            shape=np.array(self.W.shape),
            ids=self.neurons.ids,
            cell_type=self.neurons.cell_type.astype(str),
            super_class=self.neurons.super_class.astype(str),
            nt_type=self.neurons.nt_type.astype(str),
            side=self.neurons.side.astype(str),
            eye_coord=self.neurons.eye_coord,
            input_index=self.input_index,
            output_index=self.output_index,
            tau=self.tau,
            meta=np.array(json.dumps(self.meta, default=str)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WiredNetwork":
        import json

        z = np.load(Path(path), allow_pickle=False)
        W = sp.coo_matrix(
            (z["data"], (z["row"], z["col"])), shape=tuple(int(v) for v in z["shape"])
        ).tocsr()
        neurons = NeuronTable(
            ids=z["ids"],
            cell_type=z["cell_type"].astype(object),
            super_class=z["super_class"].astype(object),
            nt_type=z["nt_type"].astype(object),
            side=z["side"].astype(object),
            eye_coord=z["eye_coord"],
        )
        return cls(
            W=W,
            neurons=neurons,
            input_index=z["input_index"],
            output_index=z["output_index"],
            tau=z["tau"],
            meta=json.loads(str(z["meta"])),
        )


#: Membrane time constants per super-class, in seconds.  Photoreceptors and the
#: lamina are fast (they have to track a flickering screen), central neurons
#: slower.  Order-of-magnitude values from fly electrophysiology; override per
#: run in the config.
DEFAULT_TAU: Mapping[str, float] = {
    "sensory": 0.010,
    "optic": 0.020,
    "visual_projection": 0.030,
    "central": 0.050,
    "descending": 0.030,
    "ascending": 0.030,
    "motor": 0.030,
}
DEFAULT_TAU_FALLBACK = 0.030


def taus_from_super_class(
    neurons: NeuronTable, table: Mapping[str, float] | None = None, fallback: float | None = None
) -> np.ndarray:
    table = DEFAULT_TAU if table is None else table
    fb = DEFAULT_TAU_FALLBACK if fallback is None else fallback
    return np.array([float(table.get(sc, fb)) for sc in neurons.super_class], dtype=float)


def _aggregate_to_matrix(
    neurons: NeuronTable, synapses: SynapseTable, signs: np.ndarray
) -> sp.csr_matrix:
    n = len(neurons)
    if len(synapses) == 0:
        return sp.csr_matrix((n, n), dtype=float)
    pre = neurons.index_of(synapses.pre_ids)
    post = neurons.index_of(synapses.post_ids)
    data = synapses.counts * signs[pre]
    W = sp.coo_matrix((data, (post, pre)), shape=(n, n), dtype=float).tocsr()
    W.sum_duplicates()
    W.eliminate_zeros()
    return W


def _normalise(W: sp.csr_matrix, mode: Normalisation, gain: float) -> tuple[sp.csr_matrix, dict]:
    """Scale raw synapse counts into usable weights.

    Raw counts reach the hundreds; feeding them into a rate network straight
    would saturate or explode it.  ``in_degree`` divides each row by the total
    number of synapses that neuron receives, which makes a weight "fraction of
    this neuron's input budget" -- the scaling used by connectome-constrained
    visual models.
    """
    info: dict = {"mode": mode, "gain": gain}
    if mode == "none":
        scaled = W
    elif mode in ("in_degree", "sqrt_in_degree"):
        totals = np.asarray(np.abs(W).sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        if mode == "sqrt_in_degree":
            totals = np.sqrt(totals)
        scaled = sp.diags(1.0 / totals) @ W
        info["mean_in_degree_synapses"] = float(totals.mean())
    elif mode == "global_max":
        peak = float(np.abs(W.data).max()) if W.nnz else 1.0
        scaled = W / peak
        info["peak"] = peak
    elif mode == "spectral":
        tmp = W / (float(np.abs(W.data).max()) if W.nnz else 1.0)
        net = WiredNetwork(
            W=sp.csr_matrix(tmp),
            neurons=NeuronTable(
                ids=np.arange(tmp.shape[0]),
                cell_type=np.array(["x"] * tmp.shape[0], dtype=object),
                super_class=np.array(["central"] * tmp.shape[0], dtype=object),
                nt_type=np.array(["acetylcholine"] * tmp.shape[0], dtype=object),
                side=np.array(["center"] * tmp.shape[0], dtype=object),
                eye_coord=np.full((tmp.shape[0], 2), np.nan),
            ),
            input_index=np.zeros(0, np.int64),
            output_index=np.zeros(0, np.int64),
            tau=np.full(tmp.shape[0], 0.02),
        )
        radius = net.spectral_radius()
        info["spectral_radius_before_gain"] = radius
        scaled = tmp / max(radius, 1e-9)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown normalisation {mode!r}")
    return sp.csr_matrix(scaled * gain), info


def build_wired_network(
    connectome: Connectome,
    *,
    nt_signs: Mapping[str, float] | None = None,
    min_synapse_count: float = 5.0,
    normalise: Normalisation = "in_degree",
    gain: float = 1.0,
    drop_modulatory: bool = True,
    keep_largest_component: bool = False,
    tau_table: Mapping[str, float] | None = None,
) -> WiredNetwork:
    """Turn a :class:`Connectome` into a simulable :class:`WiredNetwork`.

    Parameters
    ----------
    min_synapse_count:
        Edges with fewer synapses are dropped as likely reconstruction noise.
        5 is the Codex default.
    normalise:
        See :func:`_normalise`.  ``"in_degree"`` is the sane default.
    drop_modulatory:
        Neurons whose transmitter maps to sign 0 (dopamine, serotonin,
        octopamine, unknown) have no effect in a rate model.  With this set
        their *edges* are removed, so ``W`` contains only edges that do
        something; the neurons stay in the table.
    keep_largest_component:
        Restrict to the largest weakly connected component, discarding
        neurons the reconstruction left unconnected.
    """
    neurons = connectome.neurons
    synapses = connectome.synapses.filter_min_count(min_synapse_count)
    signs = neurons.signs(nt_signs)

    if drop_modulatory and len(synapses):
        pre_idx = neurons.index_of(synapses.pre_ids)
        keep = signs[pre_idx] != 0.0
        synapses = SynapseTable(
            synapses.pre_ids[keep], synapses.post_ids[keep], synapses.counts[keep]
        )

    if keep_largest_component and len(synapses):
        keep_ids = _largest_component_ids(neurons, synapses)
        mask = np.isin(neurons.ids, keep_ids)
        neurons = neurons.select(mask)
        synapses = synapses.restrict_to(neurons.ids)
        signs = neurons.signs(nt_signs)

    W_raw = _aggregate_to_matrix(neurons, synapses, signs)
    W, scale_info = _normalise(W_raw, normalise, gain)

    meta = {
        "source": connectome.source,
        "min_synapse_count": min_synapse_count,
        "normalisation": scale_info,
        "drop_modulatory": drop_modulatory,
        "kept_largest_component": keep_largest_component,
        "raw_synapses": float(connectome.synapses.counts.sum()),
        "used_synapses": float(synapses.counts.sum()),
        "connectome_meta": connectome.meta,
    }
    return WiredNetwork(
        W=W,
        neurons=neurons,
        input_index=np.flatnonzero(neurons.photoreceptor_mask()),
        output_index=np.flatnonzero(neurons.descending_mask()),
        tau=taus_from_super_class(neurons, tau_table),
        meta=meta,
    )


def _largest_component_ids(neurons: NeuronTable, synapses: SynapseTable) -> np.ndarray:
    n = len(neurons)
    pre = neurons.index_of(synapses.pre_ids)
    post = neurons.index_of(synapses.post_ids)
    adj = sp.coo_matrix((np.ones(len(pre)), (pre, post)), shape=(n, n))
    n_comp, labels = sp.csgraph.connected_components(adj, directed=True, connection="weak")
    if n_comp <= 1:
        return neurons.ids
    sizes = np.bincount(labels, minlength=n_comp)
    return neurons.ids[labels == int(np.argmax(sizes))]


__all__ = [
    "DEFAULT_TAU",
    "Normalisation",
    "WiredNetwork",
    "build_wired_network",
    "taus_from_super_class",
]
