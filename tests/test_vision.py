import numpy as np
import pytest

from flyds1.hexlattice import HEX_AXES, axial_to_cartesian, hex_lattice, neighbour_pairs
from flyds1.connectome.retinotopy import eye_coords_from_soma, snap_to_hex_lattice
from flyds1.vision.ommatidia import OmmatidiaSampler, ScreenGeometry, luminance, resize_nearest
from flyds1.vision.reichardt import ReichardtBank, ReichardtConfig


def test_hex_lattice_size_and_neighbours():
    axial, coords = hex_lattice(rings=4, spacing=5.0)
    assert len(axial) == 3 * 4 * 5 + 1
    pairs = neighbour_pairs(axial)
    assert set(pairs) == set(range(len(HEX_AXES)))
    for axis, pp in pairs.items():
        dq, dr = HEX_AXES[axis]
        for i, j in pp:
            assert tuple(axial[j] - axial[i]) == (dq, dr)


def test_snap_is_the_inverse_of_the_lattice():
    axial, coords = hex_lattice(rings=3, spacing=5.0)
    assert np.array_equal(snap_to_hex_lattice(coords, 5.0), axial)


def test_soma_projection_recovers_spacing(rng):
    _, coords = hex_lattice(rings=4, spacing=5.0)
    pts = np.concatenate([coords * 1000.0, np.zeros((len(coords), 1))], axis=1)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    soma = pts @ rotation.T + rng.normal(scale=30, size=pts.shape) + 1e5
    recovered = eye_coords_from_soma(soma, spacing_deg=5.0)
    d = np.linalg.norm(recovered[:, None] - recovered[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert np.median(d.min(axis=1)) == pytest.approx(5.0, rel=0.1)


def test_sampler_has_unit_gain():
    _, coords = hex_lattice(rings=5, spacing=5.0)
    screen = ScreenGeometry(160, 90, 90.0)
    sampler = OmmatidiaSampler(coords, screen)
    out = sampler.sample(np.full((90, 160), 0.42))
    assert np.allclose(out[sampler.in_screen], 0.42)
    assert (out[~sampler.in_screen] == 0).all()


def test_sampler_is_retinotopic():
    """A bar on the left of the screen must excite left-looking facets."""
    _, coords = hex_lattice(rings=6, spacing=5.0)
    screen = ScreenGeometry(160, 90, 90.0)
    sampler = OmmatidiaSampler(coords, screen)
    frame = np.zeros((90, 160))
    frame[:, :20] = 1.0
    out = sampler.sample(frame)
    assert coords[np.argmax(out), 0] < 0  # negative azimuth = left


def test_sampler_rejects_wrong_frame_size():
    _, coords = hex_lattice(rings=2)
    sampler = OmmatidiaSampler(coords, ScreenGeometry(64, 32, 90.0))
    with pytest.raises(ValueError, match="expected"):
        sampler.sample(np.zeros((10, 10)))


def test_luminance_and_resize():
    assert luminance(np.array([[[255, 255, 255]]]))[0, 0] == pytest.approx(1.0)
    assert resize_nearest(np.zeros((40, 80, 3)), 20, 40).shape == (20, 40, 3)


def _drift(sampler, bank, speed_px, steps=90, dt=1 / 60, width=160, height=90):
    outs = []
    for k in range(steps):
        x = np.arange(width)
        row = 0.5 + 0.4 * np.sin(2 * np.pi * (x - speed_px * k * dt) / 37.0)
        frame = np.repeat(row[None, :], height, axis=0)
        outs.append(bank.step(sampler.sample(frame)[None, :], dt))
    return np.mean(outs[steps // 2 :], axis=0)[0]


def test_reichardt_is_direction_selective():
    axial, coords = hex_lattice(rings=6, spacing=5.0)
    screen = ScreenGeometry(160, 90, 90.0)
    sampler = OmmatidiaSampler(coords, screen)

    right = _drift(sampler, ReichardtBank(axial, ReichardtConfig()), +120)
    left = _drift(sampler, ReichardtBank(axial, ReichardtConfig()), -120)
    axis0 = lambda m: (m[sampler.in_screen][:, 0] + m[sampler.in_screen][:, 3]).mean()

    assert axis0(right) > 0
    assert axis0(left) < 0
    assert np.sign(axis0(right)) != np.sign(axis0(left))


def test_reichardt_is_silent_without_motion():
    axial, coords = hex_lattice(rings=4, spacing=5.0)
    sampler = OmmatidiaSampler(coords, ScreenGeometry(160, 90, 90.0))
    still = _drift(sampler, ReichardtBank(axial), 0.0)
    assert np.abs(still).max() < 1e-9


def test_reichardt_channel_count():
    axial, _ = hex_lattice(rings=2)
    assert ReichardtBank(axial, ReichardtConfig(on_off_split=True)).n_channels == 6
    assert ReichardtBank(axial, ReichardtConfig(on_off_split=False)).n_channels == 3
