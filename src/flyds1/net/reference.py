"""numpy reference implementation of the connectome-constrained rate network.

Purpose:

* run the whole pipeline without PyTorch installed (CI, quick experiments);
* serve as the ground truth the PyTorch module is checked against
  (``tests/test_net_backends.py``);
* give diagnostics -- fixed points, gain sweeps, impulse responses -- that are
  easier to write against numpy.

Not for training: no gradients here.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from flyds1.connectome.graph import WiredNetwork
from flyds1.net.bias import resting_bias
from flyds1.net.dynamics import RateConfig, activate, check_timestep


class RateNetwork:
    """Stateful forward simulation of a :class:`WiredNetwork`."""

    def __init__(
        self,
        network: WiredNetwork,
        config: RateConfig | None = None,
        *,
        bias: np.ndarray | None = None,
        batch_size: int = 1,
    ) -> None:
        self.net = network
        self.cfg = config or RateConfig()
        check_timestep(self.cfg.dt, network.tau)
        self.W = sp.csr_matrix(network.W)
        self.tau = network.tau
        # No bias given -> resting drive from the super-class table.  Zero bias
        # would leave a rectified network with inhibitory photoreceptors silent;
        # see flyds1.net.bias for why.
        self.bias = resting_bias(network) if bias is None else np.asarray(bias, dtype=float)
        if self.bias.shape != (network.n_neurons,):
            raise ValueError(f"bias must have shape ({network.n_neurons},)")
        self.reset(batch_size)

    # ------------------------------------------------------------------
    @property
    def n_neurons(self) -> int:
        return self.net.n_neurons

    def reset(self, batch_size: int | None = None) -> None:
        if batch_size is not None:
            self._batch = int(batch_size)
        self.v = np.full((self._batch, self.n_neurons), self.cfg.v_init, dtype=float)
        self.r = activate(self.v, self.cfg)

    # ------------------------------------------------------------------
    def step(self, drive: np.ndarray | None = None, n_steps: int | None = None) -> np.ndarray:
        """Integrate ``n_steps`` substeps with a constant external drive.

        ``drive`` is ``(batch, n_neurons)`` or ``(n_neurons,)``; ``None`` means
        no external input.  Returns the rates after the last substep.
        """
        steps = self.cfg.steps_per_frame if n_steps is None else int(n_steps)
        I = self._broadcast_drive(drive)
        k = self.cfg.dt / self.tau
        for _ in range(steps):
            total = (self.W @ self.r.T).T + self.bias + I
            self.v = self.v + k * (total - self.v)
            self.r = activate(self.v, self.cfg)
        return self.r

    def _broadcast_drive(self, drive: np.ndarray | None) -> np.ndarray:
        if drive is None:
            return np.zeros((self._batch, self.n_neurons))
        arr = np.asarray(drive, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[-1] != self.n_neurons:
            raise ValueError(f"drive has {arr.shape[-1]} columns, expected {self.n_neurons}")
        if arr.shape[0] == 1 and self._batch > 1:
            arr = np.repeat(arr, self._batch, axis=0)
        if arr.shape[0] != self._batch:
            self.reset(arr.shape[0])
        return arr

    def drive_from_inputs(self, values: np.ndarray) -> np.ndarray:
        """Scatter photoreceptor-sized input into a full drive vector."""
        vals = np.atleast_2d(np.asarray(values, dtype=float))
        if vals.shape[-1] != self.net.n_inputs:
            raise ValueError(f"expected {self.net.n_inputs} input values, got {vals.shape[-1]}")
        drive = np.zeros((vals.shape[0], self.n_neurons))
        drive[:, self.net.input_index] = vals
        return drive

    def outputs(self) -> np.ndarray:
        """Current descending-neuron rates, ``(batch, n_outputs)``."""
        return self.r[:, self.net.output_index]

    def rates_of(self, mask: np.ndarray) -> np.ndarray:
        return self.r[:, np.asarray(mask, dtype=bool)]

    # ------------------------------------------------------------------
    def run(self, drive_sequence: np.ndarray, *, record: str = "outputs") -> np.ndarray:
        """Run a sequence of frames.

        ``drive_sequence`` is ``(T, n_neurons)`` or ``(T, batch, n_neurons)``.
        ``record`` selects ``"outputs"`` (DN rates) or ``"all"`` (every rate).
        """
        seq = np.asarray(drive_sequence, dtype=float)
        if seq.ndim == 2:
            seq = seq[:, None, :]
        out = []
        for frame in seq:
            self.step(frame)
            out.append(self.outputs() if record == "outputs" else self.r)
        return np.stack(out, axis=0)

    # ------------------------------------------------------------------
    def fixed_point(self, drive: np.ndarray | None = None, *, tol: float = 1e-8, max_steps: int = 20000) -> tuple[np.ndarray, bool]:
        """Relax to a fixed point with no input change; returns ``(rates, converged)``."""
        I = self._broadcast_drive(drive)
        k = self.cfg.dt / self.tau
        for step in range(max_steps):
            prev = self.v
            total = (self.W @ self.r.T).T + self.bias + I
            self.v = self.v + k * (total - self.v)
            self.r = activate(self.v, self.cfg)
            if np.max(np.abs(self.v - prev)) < tol:
                return self.r, True
        return self.r, False

    def impulse_response(self, amplitude: float = 1.0, steps: int = 200) -> np.ndarray:
        """Response of the DNs to a one-frame flash on all photoreceptors."""
        self.reset()
        drive = self.drive_from_inputs(np.full((1, self.net.n_inputs), amplitude))
        trace = [self.step(drive).copy()[:, self.net.output_index]]
        for _ in range(steps - 1):
            trace.append(self.step(None)[:, self.net.output_index].copy())
        return np.concatenate(trace, axis=0)


__all__ = ["RateNetwork"]
