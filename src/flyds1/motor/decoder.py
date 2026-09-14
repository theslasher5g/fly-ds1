"""Step 5b: descending-neuron rates -> actions.

A small linear read-out, trained with the policy.  Small on purpose: if the
decoder is a deep network, it -- and not the connectome -- is doing the work,
and the experiment stops being about the fly.

DN rates sit at a positive baseline and vary by a few percent, so the decoder
first standardises them with a running mean/std (frozen at evaluation).
Without that normalisation the policy gradient spends its first thousand steps
just discovering the offset.
"""

from __future__ import annotations

import numpy as np

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


class RunningStandardiser(nn.Module):
    """Streaming mean/variance normalisation of DN rates.

    Uses Chan's parallel-variance update, including the ``delta**2`` term that
    accounts for the drift between the batch mean and the running mean.  Without
    that term the estimate only ever sees the variance *within* a batch -- and
    since a batch is a handful of near-simultaneous environment steps, that
    variance is minuscule while the variance *over time* (the part that carries
    the signal) is not.  The result was DN features with a standard deviation of
    a few thousandths, a policy that could not move, and an ``approx_kl`` pinned
    at 1e-6.

    ``max_count`` caps the effective sample count so the statistics keep
    adapting instead of freezing early in training.
    """

    def __init__(
        self,
        n_features: int,
        *,
        max_count: float = 100_000.0,
        eps: float = 1e-8,
        prior_count: float = 1e-2,
        clip: float = 10.0,
    ) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("needs torch: pip install -e '.[torch]'")
        super().__init__()
        self.max_count = float(max_count)
        self.eps = float(eps)
        self.prior_count = float(prior_count)
        #: Z-scores are clipped like ``VecNormalize`` does: early in a run the
        #: variance estimate is built from a couple of near-identical samples
        #: and can be far too small, which would hand PPO enormous features.
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("var", torch.ones(n_features))
        # The unit variance is only a prior, and it must not outweigh the data:
        # entering it with a full sample's worth of confidence leaves the first
        # few hundred updates dividing by a variance that is orders of magnitude
        # too large, which squashes the features and stalls the policy.  A tiny
        # pseudo-count keeps the prior as a guard against a zero-variance start
        # and nothing more.
        self.register_buffer("count", torch.full((1,), float(prior_count)))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Normalise with the statistics as they stood *before* this batch, then
        # fold the batch in.  Using post-update statistics is self-referential:
        # for a batch of one it subtracts the sample from itself and returns
        # exactly zero, which silently kills the gradient of everything
        # upstream -- and a batch of one is just ``n_envs=1``.
        z = (x - self.mean) / torch.sqrt(self.var + self.eps)
        if self.training:
            with torch.no_grad():
                self._update(x.detach())
        return torch.clamp(z, -self.clip, self.clip)

    def _update(self, x: "torch.Tensor") -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        n_b = float(x.shape[0])
        count = float(self.count.item())
        if count <= self.prior_count:
            # First real batch: adopt its statistics outright.  Letting the
            # zero-initialised mean take part would fold ``mean**2`` (the DN
            # baseline rate, squared) into the variance, inflating it by orders
            # of magnitude and squashing every feature that follows.
            self.mean.copy_(batch_mean)
            if n_b > 1:
                self.var.copy_(torch.clamp(batch_var, min=self.eps))
                self.count.fill_(n_b)
            return
        delta = batch_mean - self.mean
        total = count + n_b
        self.mean.add_(delta * (n_b / total))
        m2 = self.var * count + batch_var * n_b + delta.pow(2) * (count * n_b / total)
        self.var.copy_(m2 / total)
        self.count.fill_(min(total, self.max_count))


class DescendingDecoder(nn.Module):
    """``n_dn`` descending rates -> ``n_actions`` outputs, one linear layer."""

    def __init__(
        self,
        n_descending: int,
        n_actions: int,
        *,
        standardise: bool = True,
        bias: bool = True,
        init_scale: float = 0.1,
    ) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("needs torch: pip install -e '.[torch]'")
        super().__init__()
        self.norm = RunningStandardiser(n_descending) if standardise else None
        self.linear = nn.Linear(n_descending, n_actions, bias=bias)
        nn.init.orthogonal_(self.linear.weight, gain=init_scale)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, dn_rates: "torch.Tensor") -> "torch.Tensor":
        x = self.norm(dn_rates) if self.norm is not None else dn_rates
        return self.linear(x)

    def contribution(self) -> "torch.Tensor":
        """``|weight|`` per descending neuron -- which DNs drive which action."""
        return self.linear.weight.detach().abs()


class NumpyDecoder:
    """Frozen linear decoder for the torch-free path (rollouts, tests)."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
        self.weight = np.asarray(weight, dtype=float)
        if self.weight.ndim != 2:
            raise ValueError("weight must be 2-D (n_actions, n_descending)")
        self.bias = np.zeros(self.weight.shape[0]) if bias is None else np.asarray(bias, dtype=float)
        self.mean = np.zeros(self.weight.shape[1])
        self.std = np.ones(self.weight.shape[1])

    @classmethod
    def random(cls, n_descending: int, n_actions: int, *, seed: int = 0, scale: float = 0.5):
        rng = np.random.default_rng(seed)
        return cls(rng.normal(scale=scale, size=(n_actions, n_descending)))

    def fit_normaliser(self, dn_rates: np.ndarray) -> "NumpyDecoder":
        arr = np.atleast_2d(np.asarray(dn_rates, dtype=float))
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0) + 1e-6
        return self

    def decode(self, dn_rates: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(dn_rates, dtype=float))
        z = (arr - self.mean) / self.std
        out = z @ self.weight.T + self.bias
        return out[0] if np.ndim(dn_rates) == 1 else out

    @classmethod
    def from_torch(cls, decoder: "DescendingDecoder") -> "NumpyDecoder":
        w = decoder.linear.weight.detach().cpu().numpy()
        b = decoder.linear.bias.detach().cpu().numpy() if decoder.linear.bias is not None else None
        out = cls(w, b)
        if decoder.norm is not None:
            out.mean = decoder.norm.mean.detach().cpu().numpy()
            out.std = np.sqrt(decoder.norm.var.detach().cpu().numpy() + decoder.norm.eps)
        return out


__all__ = ["DescendingDecoder", "NumpyDecoder", "RunningStandardiser"]
