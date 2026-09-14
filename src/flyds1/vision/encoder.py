"""Step 3d / 6: the trainable bridge from observation to synaptic drive.

Two injection sites:

* **Photoreceptors.** Per-facet contrast becomes external current on the R1-6
  units.  One gain and one offset per photoreceptor -- the retina's own
  sensitivity -- and nothing else, so the network still has to do the work.
* **T4/T5 (optional).** Reichardt motion is injected into the direction-
  selective cells whose preferred direction matches the channel.  This is the
  "Reichardt detectors as a pre-stage" idea from the roadmap: the connectome
  already contains the T4/T5 motif, so this is a short-cut that can be switched
  off (``inject_motion=False``) to ask whether the wiring manages on its own.

Subtype directions: FlyWire labels T4/T5 subtypes ``a``-``d``, biologically the
four cardinal directions of the eye's coordinate system.  We map them onto the
hex lattice as ``a/b`` = +/- along axis 0 and ``c/d`` = +/- along axis 1, so
``c/d`` are lattice diagonals rather than true vertical -- an approximation, not
a claim about the animal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyds1.connectome.graph import WiredNetwork
from flyds1.vision.frontend import ObsLayout

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

    class _NNStub:
        class Module:
            pass

    nn = _NNStub()  # type: ignore[assignment]

#: subtype letter -> (lattice axis, direction sign)
SUBTYPE_DIRECTIONS: dict[str, tuple[int, int]] = {
    "a": (0, +1),
    "b": (0, -1),
    "c": (1, +1),
    "d": (1, -1),
}
#: cell-type prefix -> which ON/OFF pathway it reads
PATHWAY_OF_PREFIX: dict[str, int] = {"t4": 0, "t5": 1}


@dataclass
class EncoderWiring:
    """Precomputed, fixed index maps.  Built once, reused every frame."""

    n_neurons: int
    photoreceptor_index: np.ndarray            # (n_photo,) global neuron indices
    motion_target: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    motion_source: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    motion_sign: np.ndarray = field(default_factory=lambda: np.zeros(0))
    motion_group: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    group_names: tuple[str, ...] = ()
    motion_flat_size: int = 0

    @property
    def n_photoreceptors(self) -> int:
        return len(self.photoreceptor_index)

    @property
    def n_motion_targets(self) -> int:
        return len(self.motion_target)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    def describe(self) -> str:
        if self.n_motion_targets == 0:
            motion = "motion injection: off"
        else:
            per_group = {
                name: int((self.motion_group == g).sum())
                for g, name in enumerate(self.group_names)
            }
            motion = f"motion injection: {self.n_motion_targets} targets {per_group}"
        return f"EncoderWiring: {self.n_photoreceptors} photoreceptors, {motion}"


def build_encoder_wiring(
    network: WiredNetwork,
    layout: ObsLayout,
    *,
    inject_motion: bool = True,
) -> EncoderWiring:
    """Work out where each observation number has to go."""
    photo_index = np.asarray(network.input_index, dtype=np.int64)
    wiring = EncoderWiring(
        n_neurons=network.n_neurons,
        photoreceptor_index=photo_index,
        motion_flat_size=int(np.prod(layout.motion_shape)) if layout.motion_shape else 0,
    )
    if not inject_motion or layout.motion_mode == "off":
        return wiring

    neurons = network.neurons
    n_channels = layout.motion_shape[-1]
    n_axes = n_channels // 2 if n_channels % 2 == 0 else n_channels
    photo_coords = network.eye_coords()
    photo_sides = neurons.side[photo_index]
    sides = sorted({str(s) for s in photo_sides})

    targets: list[int] = []
    sources: list[int] = []
    signs: list[float] = []
    groups: list[int] = []
    group_names: list[str] = []

    for prefix, pathway in PATHWAY_OF_PREFIX.items():
        for sub, (axis, sign) in SUBTYPE_DIRECTIONS.items():
            cell_type = f"{prefix}{sub}"
            mask = neurons.mask_cell_type(cell_type)
            if not mask.any():
                continue
            channel = pathway * n_axes + axis if n_channels % 2 == 0 else axis
            if channel >= n_channels:
                continue
            group = len(group_names)
            group_names.append(cell_type)
            for idx in np.flatnonzero(mask):
                row = _source_row(
                    layout, neurons, idx, photo_coords, photo_sides, sides
                )
                if row is None:
                    continue
                targets.append(int(idx))
                sources.append(int(row * n_channels + channel))
                signs.append(float(sign))
                groups.append(group)

    wiring.motion_target = np.array(targets, dtype=np.int64)
    wiring.motion_source = np.array(sources, dtype=np.int64)
    wiring.motion_sign = np.array(signs, dtype=float)
    wiring.motion_group = np.array(groups, dtype=np.int64)
    wiring.group_names = tuple(group_names)
    return wiring


def _source_row(
    layout: ObsLayout,
    neurons,
    neuron_idx: int,
    photo_coords: np.ndarray,
    photo_sides: np.ndarray,
    sides: list[str],
) -> int | None:
    """Which row of the motion array feeds this neuron."""
    side = str(neurons.side[neuron_idx])
    if layout.motion_mode == "pooled":
        return sides.index(side) if side in sides else None
    coord = neurons.eye_coord[neuron_idx]
    if not np.isfinite(coord).all():
        return None  # no retinotopy for this cell -> cannot place it
    same_side = np.flatnonzero(photo_sides == side)
    if len(same_side) == 0:
        return None
    d = np.linalg.norm(photo_coords[same_side] - coord, axis=1)
    return int(same_side[int(np.argmin(d))])


# ---------------------------------------------------------------------------
# numpy path (no gradients): used by the reference rollout and diagnostics
# ---------------------------------------------------------------------------
def encode_numpy(
    obs: np.ndarray,
    layout: ObsLayout,
    wiring: EncoderWiring,
    *,
    photo_gain: float | np.ndarray = 1.0,
    photo_bias: float | np.ndarray = 0.0,
    motion_gain: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Observation -> drive vector ``(batch, n_neurons)``."""
    arr = np.atleast_2d(np.asarray(obs, dtype=float))
    photo, motion = layout.split(arr)
    drive = np.zeros((arr.shape[0], wiring.n_neurons), dtype=float)
    drive[:, wiring.photoreceptor_index] = photo * photo_gain + photo_bias
    if wiring.n_motion_targets:
        flat = motion.reshape(arr.shape[0], -1)
        vals = np.maximum(flat[:, wiring.motion_source] * wiring.motion_sign, 0.0)
        gains = motion_gain
        if np.ndim(motion_gain) == 1:
            gains = np.asarray(motion_gain)[wiring.motion_group]
        np.add.at(drive.T, wiring.motion_target, (vals * gains).T)
    return drive


# ---------------------------------------------------------------------------
# torch path (trainable)
# ---------------------------------------------------------------------------
class DriveEncoder(nn.Module):
    """Trainable observation -> drive map.

    Parameters are intentionally few: a gain and offset per photoreceptor and
    one gain per T4/T5 subtype.  That is the "encoder" of roadmap step 6 --
    enough to calibrate the eye to a monitor, not enough to replace the brain.
    """

    def __init__(
        self,
        wiring: EncoderWiring,
        layout: ObsLayout,
        *,
        per_facet_gain: bool = True,
        init_photo_gain: float = 1.0,
        init_motion_gain: float = 1.0,
        learn_motion_gain: bool = True,
    ) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError("needs torch: pip install -e '.[torch]'")
        super().__init__()
        self.layout = layout
        self.n_neurons = wiring.n_neurons
        self.register_buffer(
            "photo_index", torch.as_tensor(wiring.photoreceptor_index, dtype=torch.long)
        )
        n_photo = wiring.n_photoreceptors
        shape = (n_photo,) if per_facet_gain else (1,)
        self.photo_gain = nn.Parameter(torch.full(shape, float(init_photo_gain)))
        self.photo_bias = nn.Parameter(torch.zeros(shape))

        self.has_motion = wiring.n_motion_targets > 0
        if self.has_motion:
            self.register_buffer("m_target", torch.as_tensor(wiring.motion_target, dtype=torch.long))
            self.register_buffer("m_source", torch.as_tensor(wiring.motion_source, dtype=torch.long))
            self.register_buffer("m_sign", torch.as_tensor(wiring.motion_sign, dtype=torch.float32))
            self.register_buffer("m_group", torch.as_tensor(wiring.motion_group, dtype=torch.long))
            gain = torch.full((max(1, wiring.n_groups),), float(init_motion_gain))
            self.motion_gain = nn.Parameter(gain) if learn_motion_gain else None
            if self.motion_gain is None:
                self.register_buffer("motion_gain_fixed", gain)
        self.group_names = wiring.group_names

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        photo = obs[..., self.layout.photoreceptor_slice]
        drive = torch.zeros(obs.shape[:-1] + (self.n_neurons,), dtype=obs.dtype, device=obs.device)
        drive = drive.index_add(
            -1, self.photo_index, photo * self.photo_gain + self.photo_bias
        )
        if self.has_motion:
            motion = obs[..., self.layout.motion_slice]
            vals = torch.relu(motion.index_select(-1, self.m_source) * self.m_sign)
            gain = self.motion_gain if self.motion_gain is not None else self.motion_gain_fixed
            drive = drive.index_add(-1, self.m_target, vals * gain.index_select(0, self.m_group))
        return drive

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"photoreceptors={self.photo_index.numel()}, motion_groups={len(self.group_names)}"


__all__ = [
    "DriveEncoder",
    "EncoderWiring",
    "SUBTYPE_DIRECTIONS",
    "build_encoder_wiring",
    "encode_numpy",
]
