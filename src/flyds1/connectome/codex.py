"""Step 1: read real connectome dumps (FlyWire/Codex, MANC, male CNS).

Codex (https://codex.flywire.ai/) publishes, per data snapshot, a neuron
annotation table and a connection table as CSV.  The male CNS and MANC
exports from Janelia/Google Research have the same shape with different column
names.  Rather than hard-coding one spelling, this module resolves columns
through alias lists and tells you exactly what it found when a required column
is missing.

Downloading is deliberately *not* automated behind your back: Codex downloads
sit behind a terms-of-use click-through, and the data carry a citation
requirement (FlyWire: Dorkenwald et al. 2024 / Schlegel et al. 2024).  Fetch
the files once, point ``--connectome codex:<dir>`` at the directory, and the
loader takes it from there; :func:`download_instructions` prints the exact
steps, and :func:`fetch_csv` will pull a URL you supply yourself.
"""

from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from flyds1.connectome.retinotopy import eye_coords_from_hex_columns, eye_coords_from_soma
from flyds1.connectome.schema import Connectome, NeuronTable, SynapseTable

CODEX_URL = "https://codex.flywire.ai/"
CODEX_DOWNLOAD_URL = "https://codex.flywire.ai/api/download"
NEUROGLANCER_MALE_CNS = "https://neuroglancer-demo.appspot.com/"

#: Column aliases, first match wins.  Extend these when a new export shows up.
NEURON_COLUMNS: Mapping[str, Sequence[str]] = {
    "id": ("root_id", "pt_root_id", "bodyid", "body_id", "id", "neuron_id"),
    "cell_type": ("cell_type", "type", "hemibrain_type", "cell_subclass", "celltype"),
    "super_class": ("super_class", "superclass", "class", "cell_class", "coarse_class"),
    "nt_type": ("nt_type", "neurotransmitter", "nt", "predicted_nt", "consensus_nt"),
    "side": ("side", "hemisphere", "soma_side"),
}
NEURON_OPTIONAL: Mapping[str, Sequence[str]] = {
    "hex1": ("hex1", "column_p", "col_p", "p"),
    "hex2": ("hex2", "column_q", "col_q", "q"),
    "soma_x": ("soma_x", "pos_x", "x", "soma_position_x"),
    "soma_y": ("soma_y", "pos_y", "y", "soma_position_y"),
    "soma_z": ("soma_z", "pos_z", "z", "soma_position_z"),
}
CONNECTION_COLUMNS: Mapping[str, Sequence[str]] = {
    "pre": ("pre_root_id", "pre_pt_root_id", "pre_id", "bodyid_pre", "pre", "source"),
    "post": ("post_root_id", "post_pt_root_id", "post_id", "bodyid_post", "post", "target"),
    "count": ("syn_count", "synapse_count", "weight", "count", "n_syn"),
}

#: Filenames the loader looks for inside a Codex directory.
NEURON_FILE_CANDIDATES = (
    "neurons.csv",
    "neurons.csv.gz",
    "cell_annotations.csv",
    "annotations.csv",
    "classification.csv",
)
CONNECTION_FILE_CANDIDATES = (
    "connections.csv",
    "connections.csv.gz",
    "connectivity.csv",
    "connections_princeton_no_threshold.csv.gz",
    "connections_no_threshold.csv.gz",
)

#: FlyWire/Codex spellings of the photoreceptor and descending annotations.
PHOTORECEPTOR_TYPES = ("r1-6", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r7/r8")


@dataclass
class CodexDump:
    """Paths to one snapshot's two tables."""

    neurons: Path
    connections: Path

    @classmethod
    def discover(cls, directory: str | Path) -> "CodexDump":
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"{d} is not a directory")
        neurons = _first_existing(d, NEURON_FILE_CANDIDATES)
        connections = _first_existing(d, CONNECTION_FILE_CANDIDATES)
        if neurons is None or connections is None:
            present = sorted(p.name for p in d.iterdir())
            raise FileNotFoundError(
                f"could not find a neuron and a connection table in {d}.\n"
                f"  looked for neurons in: {NEURON_FILE_CANDIDATES}\n"
                f"  looked for connections in: {CONNECTION_FILE_CANDIDATES}\n"
                f"  found: {present}\n\n{download_instructions()}"
            )
        return cls(neurons=neurons, connections=connections)


def _first_existing(directory: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def _resolve(header: Sequence[str], aliases: Mapping[str, Sequence[str]], *, required: bool) -> dict[str, int]:
    lowered = {h.strip().lower(): i for i, h in enumerate(header)}
    out: dict[str, int] = {}
    for field, options in aliases.items():
        for option in options:
            if option in lowered:
                out[field] = lowered[option]
                break
        else:
            if required:
                raise KeyError(
                    f"no column for {field!r} (tried {list(options)}); "
                    f"columns present: {sorted(lowered)}"
                )
    return out


def read_neuron_csv(path: str | Path, *, spacing_deg: float = 5.0) -> NeuronTable:
    """Read a neuron annotation CSV into a :class:`NeuronTable`."""
    path = Path(path)
    with _open_text(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = _resolve(header, NEURON_COLUMNS, required=True)
        opt = _resolve(header, NEURON_OPTIONAL, required=False)
        ids: list[int] = []
        cell_type: list[str] = []
        super_class: list[str] = []
        nt_type: list[str] = []
        side: list[str] = []
        hex_coords: list[tuple[float, float]] = []
        soma: list[tuple[float, float, float]] = []
        seen: set[int] = set()
        for row in reader:
            if not row:
                continue
            try:
                nid = _parse_id(row[cols["id"]])
            except (ValueError, IndexError):
                continue
            if nid in seen:  # annotation tables can carry one row per label
                continue
            seen.add(nid)
            ids.append(nid)
            cell_type.append(_get(row, cols, "cell_type"))
            super_class.append(_get(row, cols, "super_class"))
            nt_type.append(_normalise_nt(_get(row, cols, "nt_type")))
            side.append(_get(row, cols, "side"))
            hex_coords.append((_getf(row, opt, "hex1"), _getf(row, opt, "hex2")))
            soma.append(
                (_getf(row, opt, "soma_x"), _getf(row, opt, "soma_y"), _getf(row, opt, "soma_z"))
            )

    n = len(ids)
    table = NeuronTable(
        ids=np.array(ids, dtype=np.int64),
        cell_type=np.array(cell_type, dtype=object),
        super_class=np.array(super_class, dtype=object),
        nt_type=np.array(nt_type, dtype=object),
        side=np.array(side, dtype=object),
        eye_coord=np.full((n, 2), np.nan),
        extra={"soma_xyz": np.array(soma, dtype=float)},
    )
    _fill_eye_coords(table, np.array(hex_coords, dtype=float), spacing_deg)
    return table


def _parse_id(raw: str) -> int:
    """Parse a root/body id exactly.

    FlyWire root ids are ~7.2e17, which does **not** fit in a float64 mantissa:
    going through ``float`` silently merges distinct neurons into one id.  So
    parse as ``int`` and fall back to ``Decimal`` (not ``float``) for exports
    that write ids as ``7.2e17`` or ``123.0``.
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty id")
    try:
        return int(s)
    except ValueError:
        try:
            return int(Decimal(s))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"cannot parse id {raw!r}") from exc


def _get(row: Sequence[str], cols: Mapping[str, int], field: str) -> str:
    idx = cols.get(field)
    if idx is None or idx >= len(row):
        return ""
    return row[idx].strip()


def _getf(row: Sequence[str], cols: Mapping[str, int], field: str) -> float:
    raw = _get(row, cols, field)
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _normalise_nt(raw: str) -> str:
    nt = raw.strip().lower()
    return {
        "ach": "acetylcholine",
        "acetylcholine": "acetylcholine",
        "gaba": "gaba",
        "glut": "glutamate",
        "glutamate": "glutamate",
        "da": "dopamine",
        "dopamine": "dopamine",
        "ser": "serotonin",
        "5ht": "serotonin",
        "serotonin": "serotonin",
        "oct": "octopamine",
        "octopamine": "octopamine",
        "his": "histamine",
        "histamine": "histamine",
        "unk": "unknown",
        "": "unknown",
    }.get(nt, nt)


def _fill_eye_coords(table: NeuronTable, hex_coords: np.ndarray, spacing_deg: float) -> None:
    """Attach eye coordinates to photoreceptors, hex annotations first."""
    photo = table.mask_cell_type(*PHOTORECEPTOR_TYPES) & table.mask_super_class("sensory")
    if not photo.any():
        # Some exports label photoreceptors only by type, not super_class.
        photo = table.mask_cell_type(*PHOTORECEPTOR_TYPES)
        table.super_class[photo] = "sensory"
    if not photo.any():
        return

    have_hex = photo & np.isfinite(hex_coords).all(axis=1)
    if have_hex.any():
        table.eye_coord[have_hex] = eye_coords_from_hex_columns(
            hex_coords[have_hex, 0], hex_coords[have_hex, 1], spacing_deg
        )

    soma = table.extra.get("soma_xyz")
    remaining = photo & ~np.isfinite(table.eye_coord).all(axis=1)
    if soma is None or not remaining.any():
        return
    for side in np.unique(table.side[remaining]):
        sel = remaining & (table.side == side)
        usable = sel & np.isfinite(soma).all(axis=1)
        if usable.sum() >= 3:
            table.eye_coord[usable] = eye_coords_from_soma(
                soma[usable], spacing_deg, flip_x=(str(side) == "left")
            )


def read_connection_csv(path: str | Path, *, min_count: float = 0.0) -> SynapseTable:
    """Read a connection CSV, aggregating duplicate rows (e.g. per neuropil)."""
    path = Path(path)
    totals: dict[tuple[int, int], float] = {}
    with _open_text(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = _resolve(header, CONNECTION_COLUMNS, required=True)
        for row in reader:
            if not row:
                continue
            try:
                pre = _parse_id(row[cols["pre"]])
                post = _parse_id(row[cols["post"]])
                count = float(row[cols["count"]])
            except (ValueError, IndexError):
                continue
            if count < min_count:
                continue
            key = (pre, post)
            totals[key] = totals.get(key, 0.0) + count
    if not totals:
        return SynapseTable(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    keys = np.array(list(totals.keys()), dtype=np.int64)
    return SynapseTable(keys[:, 0], keys[:, 1], np.array(list(totals.values()), dtype=float))


def load_codex_dump(
    directory: str | Path, *, spacing_deg: float = 5.0, min_count: float = 0.0
) -> Connectome:
    """Load a Codex/MANC style directory into a :class:`Connectome`.

    Synapse endpoints missing from the neuron table are dropped (partial
    annotation exports are common) and reported in ``meta``.
    """
    dump = CodexDump.discover(directory)
    neurons = read_neuron_csv(dump.neurons, spacing_deg=spacing_deg)
    synapses = read_connection_csv(dump.connections, min_count=min_count)

    known = set(int(i) for i in neurons.ids)
    keep = np.array(
        [(int(a) in known and int(b) in known) for a, b in zip(synapses.pre_ids, synapses.post_ids)],
        dtype=bool,
    ) if len(synapses) else np.zeros(0, dtype=bool)
    dropped = int((~keep).sum())
    synapses = SynapseTable(synapses.pre_ids[keep], synapses.post_ids[keep], synapses.counts[keep])

    return Connectome(
        neurons=neurons,
        synapses=synapses,
        source=f"codex:{Path(directory).name}",
        meta={
            "neuron_file": str(dump.neurons),
            "connection_file": str(dump.connections),
            "dropped_unannotated_edges": dropped,
            "spacing_deg": spacing_deg,
            "citation": (
                "FlyWire: Dorkenwald et al. 2024 (Nature) and Schlegel et al. 2024 (Nature); "
                "check the snapshot's own license and citation terms before publishing."
            ),
        },
    )


def fetch_csv(url: str, dest: str | Path, *, timeout: float = 60.0) -> Path:
    """Download one CSV you have the URL for.  Requires ``requests``.

    Kept separate from :func:`load_codex_dump` on purpose: you accept Codex's
    terms, you pick the snapshot, this only moves bytes.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fetch_csv needs 'requests': pip install -e '.[connectome]'"
        ) from exc
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def download_instructions() -> str:
    """Printable instructions for getting real data onto disk."""
    return (
        "How to get a real connectome:\n"
        f"  1. Open {CODEX_URL} and sign in; accept the data terms.\n"
        "  2. Downloads -> pick a snapshot (FlyWire FAFB 783 is the usual choice) and\n"
        "     grab the neuron annotation CSV and the connection CSV.\n"
        "  3. Put both in one directory, e.g. data/flywire_783/.\n"
        "  4. flyds1 build --connectome codex:data/flywire_783 --out data/net.npz\n"
        "\n"
        "Male CNS / MANC: the Janelia+Google exports have the same two-table shape;\n"
        f"browse the volume via Neuroglancer ({NEUROGLANCER_MALE_CNS}) and use the\n"
        "published connection CSV -- the column aliases in this module already cover\n"
        "'bodyId'/'type'/'class'/'weight'.\n"
        "\n"
        "Alternatively use the natverse route (pip install -e '.[connectome]'):\n"
        "  from flyds1.connectome.natverse import connectome_from_fafbseg\n"
    )


__all__ = [
    "CODEX_URL",
    "CodexDump",
    "download_instructions",
    "fetch_csv",
    "load_codex_dump",
    "read_connection_csv",
    "read_neuron_csv",
]
