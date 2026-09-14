"""Shared fixtures.  Everything uses the smallest synthetic connectome that
still contains a full visual hierarchy, so the suite stays fast."""

from __future__ import annotations

import numpy as np
import pytest

from flyds1.connectome.graph import build_wired_network
from flyds1.connectome.synthetic import SyntheticConfig, build_synthetic_connectome


@pytest.fixture(scope="session")
def connectome():
    return build_synthetic_connectome(SyntheticConfig(rings=3, n_central=60, n_descending=8, seed=0))


@pytest.fixture(scope="session")
def network(connectome):
    return build_wired_network(connectome, normalise="in_degree", gain=3.0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: needs PyTorch")
    config.addinivalue_line("markers", "gym: needs gymnasium")
