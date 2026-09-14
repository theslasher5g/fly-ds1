"""Hexagonal lattice helpers.

The fly's retina is a hexagonal array of ommatidia, so the same lattice math is
needed in two places: the synthetic connectome generator (to hand out eye
coordinates to photoreceptors) and the vision front end (to sample the screen
and to wire up Reichardt detectors between *neighbouring* facets).  It lives
here so neither of those two has to import the other.

Coordinates are axial hex coordinates ``(q, r)`` converted to a Cartesian
"eye plane" in **degrees of visual angle**, with the spacing equal to the
interommatidial angle (~5 deg in Drosophila).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

#: The three lattice axes, in axial (q, r) steps.  A hex lattice has three
#: directions, not four -- these are the axes T4/T5 motion detectors are
#: organised along.
HEX_AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))

_SQRT3_2 = np.sqrt(3.0) / 2.0


def axial_ring_indices(rings: int) -> np.ndarray:
    """All axial coordinates within ``rings`` steps of the origin.

    Returns an ``(n, 2)`` int array; ``n = 3*rings*(rings+1) + 1``.
    """
    if rings < 0:
        raise ValueError("rings must be >= 0")
    out = [
        (q, r)
        for q in range(-rings, rings + 1)
        for r in range(-rings, rings + 1)
        if abs(q + r) <= rings
    ]
    return np.array(sorted(out, key=lambda qr: (qr[1], qr[0])), dtype=np.int64)


def axial_to_cartesian(axial: np.ndarray, spacing: float = 5.0) -> np.ndarray:
    """Axial hex coordinates -> Cartesian degrees ``(azimuth, elevation)``."""
    axial = np.asarray(axial, dtype=float)
    q, r = axial[:, 0], axial[:, 1]
    x = spacing * (q + 0.5 * r)
    y = spacing * (_SQRT3_2 * r)
    return np.stack([x, y], axis=1)


def hex_lattice(rings: int = 12, spacing: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a hex lattice.

    Parameters
    ----------
    rings:
        Number of rings around the centre facet.  ``rings=15`` gives 721
        facets, close to the ~750 ommatidia of one Drosophila eye.
    spacing:
        Interommatidial angle in degrees (Drosophila: ~4.5-5.5 deg).

    Returns
    -------
    axial, coords:
        ``(n, 2)`` integer axial coordinates and ``(n, 2)`` Cartesian degrees.
    """
    axial = axial_ring_indices(rings)
    return axial, axial_to_cartesian(axial, spacing)


def neighbour_pairs(axial: np.ndarray) -> dict[int, np.ndarray]:
    """Neighbouring facet pairs along each of the three lattice axes.

    Returns ``{axis_index: (m, 2) array of (i, j)}`` where ``j`` sits one step
    along :data:`HEX_AXES` from ``i``.  Motion from ``i`` to ``j`` is the
    positive direction of that axis.
    """
    axial = np.asarray(axial, dtype=np.int64)
    lookup = {(int(q), int(r)): k for k, (q, r) in enumerate(axial)}
    pairs: dict[int, np.ndarray] = {}
    for a, (dq, dr) in enumerate(HEX_AXES):
        found = [
            (k, lookup[(int(q) + dq, int(r) + dr)])
            for k, (q, r) in enumerate(axial)
            if (int(q) + dq, int(r) + dr) in lookup
        ]
        pairs[a] = np.array(found, dtype=np.int64).reshape(-1, 2)
    return pairs


def axis_directions(spacing: float = 5.0) -> np.ndarray:
    """Unit vectors of the three lattice axes in the Cartesian eye plane."""
    dirs = axial_to_cartesian(np.array(HEX_AXES, dtype=float), spacing)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def iter_rings(rings: int) -> Iterator[np.ndarray]:
    """Yield the axial coordinates of ring 0, 1, ... ``rings`` separately."""
    for k in range(rings + 1):
        inner = axial_ring_indices(k - 1) if k else np.zeros((0, 2), dtype=np.int64)
        outer = axial_ring_indices(k)
        inner_set = {(int(q), int(r)) for q, r in inner}
        yield np.array(
            [qr for qr in outer if (int(qr[0]), int(qr[1])) not in inner_set], dtype=np.int64
        )
