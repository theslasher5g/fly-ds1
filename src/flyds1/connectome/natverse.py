"""Optional step-1 route: the natverse Python stack (``fafbseg`` / ``navis``).

Use this when you want to query FlyWire live (with a CAVE token) instead of
working from a downloaded snapshot.  The dependencies are heavy and need
credentials, so everything is imported lazily and the module raises a message
that tells you what to install.

Set up once:

    pip install -e ".[connectome]"
    python -c "import fafbseg; fafbseg.flywire.set_chunkedgraph_secret('<CAVE token>')"
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from flyds1.connectome.codex import _normalise_nt
from flyds1.connectome.retinotopy import eye_coords_from_soma
from flyds1.connectome.schema import Connectome, NeuronTable, SynapseTable

_INSTALL_HINT = (
    "this route needs the natverse stack: pip install -e '.[connectome]' "
    "and a CAVE token (fafbseg.flywire.set_chunkedgraph_secret)"
)


def _require_fafbseg():
    import importlib.util

    missing = [m for m in ("fafbseg", "pandas") if importlib.util.find_spec(m) is None]
    if missing:  # pragma: no cover - optional dependency
        raise ImportError(f"{_INSTALL_HINT} (missing: {', '.join(missing)})")
    import fafbseg

    return fafbseg


def connectome_from_fafbseg(
    root_ids: Sequence[int] | None = None,
    *,
    dataset: str = "public",
    min_count: float = 5.0,
    spacing_deg: float = 5.0,
) -> Connectome:
    """Build a :class:`Connectome` from a live FlyWire query.

    Parameters
    ----------
    root_ids:
        Neurons to include.  ``None`` pulls the full annotated set, which is
        large (~140k neurons) -- start with a region's ids while prototyping.
    dataset:
        ``"public"`` for the released snapshot, ``"production"`` if your token
        has access.
    """
    fafbseg = _require_fafbseg()
    import pandas as pd

    annotations: pd.DataFrame = fafbseg.flywire.get_annotations(root_ids, dataset=dataset)
    if root_ids is None:
        root_ids = annotations["root_id"].to_numpy()

    edges: pd.DataFrame = fafbseg.flywire.get_connectivity(
        root_ids, dataset=dataset, min_size=min_count, upstream=True, downstream=True
    )
    return connectome_from_frames(annotations, edges, spacing_deg=spacing_deg)


def connectome_from_frames(annotations, edges, *, spacing_deg: float = 5.0) -> Connectome:
    """Convert two pandas frames (natverse shape) into a :class:`Connectome`.

    Split out from :func:`connectome_from_fafbseg` so it can be used on frames
    you obtained any other way -- a ``navis`` NeuronList, an R ``natverse``
    export written to feather, a cached parquet file.
    """
    def pick(frame, *names, default=None):
        for name in names:
            if name in frame.columns:
                return frame[name]
        if default is None:
            raise KeyError(f"none of {names} in columns {list(frame.columns)}")
        return default

    ids = np.asarray(pick(annotations, "root_id", "pt_root_id", "bodyId"), dtype=np.int64)
    n = len(ids)
    zeros = np.array([""] * n, dtype=object)
    table = NeuronTable(
        ids=ids,
        cell_type=np.asarray(pick(annotations, "cell_type", "type", default=zeros), dtype=object),
        super_class=np.asarray(
            pick(annotations, "super_class", "class", default=zeros), dtype=object
        ),
        nt_type=np.array(
            [_normalise_nt(str(v)) for v in pick(annotations, "nt_type", "nt", default=zeros)],
            dtype=object,
        ),
        side=np.asarray(pick(annotations, "side", "hemisphere", default=zeros), dtype=object),
        eye_coord=np.full((n, 2), np.nan),
    )

    soma_cols = [c for c in ("soma_x", "soma_y", "soma_z") if c in annotations.columns]
    if len(soma_cols) == 3:
        soma = annotations[soma_cols].to_numpy(dtype=float)
        photo = table.photoreceptor_mask() | table.mask_cell_type("r1-6", "r7", "r8")
        for side in np.unique(table.side[photo]):
            sel = photo & (table.side == side) & np.isfinite(soma).all(axis=1)
            if sel.sum() >= 3:
                table.eye_coord[sel] = eye_coords_from_soma(
                    soma[sel], spacing_deg, flip_x=(str(side).lower() == "left")
                )

    pre = np.asarray(pick(edges, "pre", "pre_root_id", "bodyId_pre", "source"), dtype=np.int64)
    post = np.asarray(pick(edges, "post", "post_root_id", "bodyId_post", "target"), dtype=np.int64)
    counts = np.asarray(pick(edges, "weight", "syn_count", "count"), dtype=float)
    synapses = SynapseTable(pre, post, counts).restrict_to(ids)

    return Connectome(
        neurons=table,
        synapses=synapses,
        source="fafbseg",
        meta={"spacing_deg": spacing_deg, "note": "live natverse query"},
    )


def as_navis_graph(connectome: Connectome):
    """Hand the graph to ``navis``/``networkx`` for plotting and graph metrics."""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("needs networkx: pip install -e '.[connectome]'") from exc
    g = nx.DiGraph()
    for i, ct, sc in zip(
        connectome.neurons.ids, connectome.neurons.cell_type, connectome.neurons.super_class
    ):
        g.add_node(int(i), cell_type=ct, super_class=sc)
    for a, b, c in zip(
        connectome.synapses.pre_ids, connectome.synapses.post_ids, connectome.synapses.counts
    ):
        g.add_edge(int(a), int(b), weight=float(c))
    return g


__all__ = ["as_navis_graph", "connectome_from_fafbseg", "connectome_from_frames"]
