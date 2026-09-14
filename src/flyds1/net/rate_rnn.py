"""Step 2b: the connectome as a differentiable recurrent network (PyTorch).

Same equation as :mod:`flyds1.net.dynamics`, written so gradients flow through
it.  What is trainable is a deliberate choice (roadmap step 6):

* ``W`` from the connectome -- **frozen**.  It is the biological prior; training
  it away defeats the purpose.
* ``bias`` -- optional.  Resting drive is not in the connectome, so learning it
  is legitimate; off by default so the baseline stays interpretable.
* ``plastic`` edges -- optional, small.  Stands in for connections the
  reconstruction missed or that are neuromodulatory; every plastic edge is a
  place where the model can cheat, so the count is explicit and logged.
* the encoder (screen -> eye) and the decoder (DNs -> keys) live in
  :mod:`flyds1.vision.encoder` and :mod:`flyds1.motor.decoder`.

Gradients through time: :meth:`ConnectomeRNN.forward` keeps the graph across
substeps, so supervised or offline training can backprop through a rollout.
For on-policy RL the state is detached between environment steps (see
``detach`` in :meth:`step`), which trains encoder and decoder as a readout of a
connectome-shaped reservoir.  Full BPTT through an episode needs a recurrent
policy (``sb3-contrib``'s ``RecurrentPPO``) -- noted in the README as the
upgrade path, not pretended to be free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyds1.connectome.graph import WiredNetwork
from flyds1.net.bias import resting_bias
from flyds1.net.dynamics import RateConfig, check_timestep

try:  # torch is an optional dependency
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in torch-less installs
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

    class _NNStub:
        class Module:  # minimal stand-in so the class body below still parses
            pass

    nn = _NNStub()  # type: ignore[assignment]

_TORCH_HINT = "the PyTorch backend needs torch: pip install -e '.[torch]'"


def require_torch() -> None:
    if not _TORCH_AVAILABLE:  # pragma: no cover - depends on environment
        raise ImportError(_TORCH_HINT)


@dataclass
class TrainableParts:
    """Which parts of the network the optimiser may touch."""

    bias: bool = False
    plastic_edges: int = 0        # number of extra learnable connections
    plastic_init_std: float = 0.01
    neuron_gain: bool = False     # per-neuron output gain, starts at 1
    seed: int = 0


class ConnectomeRNN(nn.Module):
    """Rate network whose recurrent weights come from a connectome."""

    def __init__(
        self,
        network: WiredNetwork,
        config: RateConfig | None = None,
        trainable: TrainableParts | None = None,
        *,
        bias: np.ndarray | None = None,
        dtype: "torch.dtype | None" = None,
    ) -> None:
        require_torch()
        super().__init__()
        self.cfg = config or RateConfig()
        self.parts = trainable or TrainableParts()
        check_timestep(self.cfg.dt, network.tau)

        dtype = dtype or torch.float32
        self.n_neurons = network.n_neurons
        self.input_index_np = np.asarray(network.input_index)
        self.output_index_np = np.asarray(network.output_index)

        coo = network.W.tocoo()
        self.register_buffer("w_post", torch.as_tensor(coo.row, dtype=torch.long))
        self.register_buffer("w_pre", torch.as_tensor(coo.col, dtype=torch.long))
        self.register_buffer("w_val", torch.as_tensor(coo.data, dtype=dtype))
        self.register_buffer("tau", torch.as_tensor(network.tau, dtype=dtype))
        self.register_buffer("input_index", torch.as_tensor(self.input_index_np, dtype=torch.long))
        self.register_buffer("output_index", torch.as_tensor(self.output_index_np, dtype=torch.long))

        bias_np = resting_bias(network) if bias is None else np.asarray(bias, dtype=float)
        bias_t = torch.as_tensor(bias_np, dtype=dtype)
        if self.parts.bias:
            self.bias = nn.Parameter(bias_t)
        else:
            self.register_buffer("bias", bias_t)

        if self.parts.neuron_gain:
            self.gain = nn.Parameter(torch.ones(self.n_neurons, dtype=dtype))
        else:
            self.register_buffer("gain", torch.ones(self.n_neurons, dtype=dtype))

        self._init_plastic_edges(network, dtype)
        self._sparse_cache: "torch.Tensor | None" = None

    # ------------------------------------------------------------------
    def _init_plastic_edges(self, network: WiredNetwork, dtype) -> None:
        n_extra = int(self.parts.plastic_edges)
        if n_extra <= 0:
            self.register_buffer("p_post", torch.zeros(0, dtype=torch.long))
            self.register_buffer("p_pre", torch.zeros(0, dtype=torch.long))
            self.plastic_val = None
            return
        rng = np.random.default_rng(self.parts.seed)
        existing = set(zip(network.W.tocoo().row.tolist(), network.W.tocoo().col.tolist()))
        picks: set[tuple[int, int]] = set()
        # Rejection-sample pairs that are not already wired.  Candidates are
        # drawn among neurons that already participate in the graph, so plastic
        # edges cannot resurrect neurons the reconstruction left isolated.
        active = np.unique(np.concatenate([network.W.tocoo().row, network.W.tocoo().col]))
        if len(active) < 2:
            active = np.arange(self.n_neurons)
        guard = 0
        while len(picks) < n_extra and guard < 100 * n_extra:
            post = int(rng.choice(active))
            pre = int(rng.choice(active))
            guard += 1
            if post == pre or (post, pre) in existing or (post, pre) in picks:
                continue
            picks.add((post, pre))
        arr = np.array(sorted(picks), dtype=np.int64).reshape(-1, 2)
        self.register_buffer("p_post", torch.as_tensor(arr[:, 0], dtype=torch.long))
        self.register_buffer("p_pre", torch.as_tensor(arr[:, 1], dtype=torch.long))
        self.plastic_val = nn.Parameter(
            torch.randn(len(arr), dtype=dtype) * self.parts.plastic_init_std
        )

    # ------------------------------------------------------------------
    @property
    def n_inputs(self) -> int:
        return len(self.input_index_np)

    @property
    def n_outputs(self) -> int:
        return len(self.output_index_np)

    def trainable_summary(self) -> str:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = int(self.w_val.numel())
        lines = [
            f"ConnectomeRNN: {self.n_neurons} neurons, {frozen} frozen connectome weights",
            f"  trainable parameters inside the network: {total}",
            f"    bias: {'yes' if self.parts.bias else 'no'}"
            f" | gain: {'yes' if self.parts.neuron_gain else 'no'}"
            f" | plastic edges: {int(self.p_post.numel())}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def init_state(self, batch_size: int = 1, device=None) -> "torch.Tensor":
        device = device or self.w_val.device
        return torch.full(
            (batch_size, self.n_neurons), float(self.cfg.v_init), dtype=self.w_val.dtype, device=device
        )

    def activation(self, v: "torch.Tensor") -> "torch.Tensor":
        kind = self.cfg.activation
        if kind == "relu":
            r = torch.relu(v)
        elif kind == "softplus":
            r = torch.nn.functional.softplus(v, beta=self.cfg.softplus_beta)
        elif kind == "tanh":
            r = torch.tanh(v)
        elif kind == "sigmoid":
            r = torch.sigmoid(v)
        elif kind == "identity":
            r = v
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown activation {kind!r}")
        if self.cfg.rate_clip is not None:
            r = torch.clamp(r, max=float(self.cfg.rate_clip))
        return r * self.gain

    def _recurrent_input(self, r: "torch.Tensor") -> "torch.Tensor":
        """``W @ r`` for the frozen connectome plus any plastic edges."""
        sparse = self._frozen_sparse()
        total = torch.sparse.mm(sparse, r.t()).t()
        if self.plastic_val is not None and self.p_post.numel():
            contrib = r.index_select(1, self.p_pre) * self.plastic_val
            total = total.index_add(1, self.p_post, contrib)
        return total

    def _frozen_sparse(self) -> "torch.Tensor":
        cache = self._sparse_cache
        if cache is None or cache.device != self.w_val.device or cache.dtype != self.w_val.dtype:
            indices = torch.stack([self.w_post, self.w_pre])
            cache = torch.sparse_coo_tensor(
                indices,
                self.w_val,
                (self.n_neurons, self.n_neurons),
                check_invariants=False,  # indices come from scipy, already valid
            ).coalesce()
            self._sparse_cache = cache
        return cache

    # ------------------------------------------------------------------
    def forward(
        self,
        v: "torch.Tensor",
        drive: "torch.Tensor | None" = None,
        n_steps: int | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Integrate substeps with constant drive; returns ``(v, rates)``.

        The autograd graph spans all substeps, so this is differentiable within
        one environment frame.
        """
        steps = self.cfg.steps_per_frame if n_steps is None else int(n_steps)
        k = (self.cfg.dt / self.tau).unsqueeze(0)
        r = self.activation(v)
        for _ in range(steps):
            total = self._recurrent_input(r) + self.bias
            if drive is not None:
                total = total + drive
            v = v + k * (total - v)
            r = self.activation(v)
        return v, r

    def step(
        self,
        v: "torch.Tensor",
        drive: "torch.Tensor | None" = None,
        *,
        detach: bool = False,
        n_steps: int | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """One environment frame.  ``detach=True`` truncates BPTT at this frame."""
        v, r = self.forward(v, drive, n_steps)
        if detach:
            v = v.detach()
        return v, r

    def scatter_drive(self, values: "torch.Tensor") -> "torch.Tensor":
        """Photoreceptor-sized tensor -> full-size drive vector."""
        if values.shape[-1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} input values, got {values.shape[-1]}")
        drive = torch.zeros(
            values.shape[:-1] + (self.n_neurons,), dtype=values.dtype, device=values.device
        )
        return drive.index_copy(-1, self.input_index, values)

    def scatter_into(self, values: "torch.Tensor", index: "torch.Tensor") -> "torch.Tensor":
        """Scatter drive into arbitrary neuron indices (e.g. T4/T5 units)."""
        drive = torch.zeros(
            values.shape[:-1] + (self.n_neurons,), dtype=values.dtype, device=values.device
        )
        return drive.index_copy(-1, index, values)

    def read_outputs(self, r: "torch.Tensor") -> "torch.Tensor":
        """Descending-neuron rates, ``(batch, n_outputs)``."""
        return r.index_select(-1, self.output_index)


__all__ = ["ConnectomeRNN", "TrainableParts", "require_torch"]
