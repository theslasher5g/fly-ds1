"""One entry point for "where does the connectome come from?".

Specs are strings so they can live in a config file or on the command line:

* ``synthetic``                -- default synthetic connectome
* ``synthetic:rings=8,seed=3`` -- synthetic with overrides
* ``codex:data/flywire_783``   -- a downloaded Codex/MANC dump directory
* ``npz:data/net.npz``         -- an already built :class:`WiredNetwork`
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from flyds1.connectome.schema import Connectome


def load_connectome(spec: str = "synthetic", *, spacing_deg: float | None = None) -> Connectome:
    """Resolve a connectome spec into a :class:`Connectome`."""
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()

    if kind in ("", "synthetic"):
        from flyds1.connectome.synthetic import SyntheticConfig, build_synthetic_connectome

        cfg = SyntheticConfig(**_parse_overrides(rest, SyntheticConfig))
        if spacing_deg is not None:
            cfg.spacing_deg = spacing_deg
        return build_synthetic_connectome(cfg)

    if kind in ("codex", "flywire", "manc", "malecns", "csv"):
        from flyds1.connectome.codex import load_codex_dump

        return load_codex_dump(Path(rest), spacing_deg=5.0 if spacing_deg is None else spacing_deg)

    if kind == "npz":
        raise ValueError(
            "npz holds an already-built WiredNetwork; load it with "
            "WiredNetwork.load(path) instead of load_connectome()"
        )

    raise ValueError(
        f"unknown connectome spec {spec!r}; expected 'synthetic[:k=v,...]', "
        "'codex:<dir>' or 'npz:<file>'"
    )


def _parse_overrides(rest: str, cls) -> dict:
    if not rest.strip():
        return {}
    types = {f.name: f.type for f in fields(cls)}
    out: dict = {}
    for chunk in rest.split(","):
        if not chunk.strip():
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        if key not in types:
            raise ValueError(f"{cls.__name__} has no field {key!r}")
        out[key] = float(value) if "." in value else int(value)
    return out


__all__ = ["load_connectome"]
