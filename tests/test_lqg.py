import numpy as np
import pytest

from lqg import black_hole, cosmology, spectra, spin_foam
from lqg.constants import RHO_CRIT
from lqg.spin_network import SpinNetwork
from lqg.su2 import intertwiner_dim, wigner_6j


def test_6j_known_values():
    assert wigner_6j(2, 2, 2, 2, 2, 2) == pytest.approx(1 / 6)
    assert wigner_6j(1, 1, 1, 1, 1, 1) == 0.0  # violation de parité


def test_intertwiner_dimensions():
    assert intertwiner_dim([1, 1, 1, 1]) == 2
    assert intertwiner_dim([2, 2, 2, 2, 2, 2]) == 15
    assert intertwiner_dim([2, 2, 4]) == 1 and intertwiner_dim([1, 1, 4]) == 0


def test_area_spectrum():
    assert spectra.area_gap() == pytest.approx(8 * np.pi * 0.23753295796592 * np.sqrt(3) / 2)


def test_tetra_volume_half_spin():
    v = spectra.volume_spectrum((1, 1, 1, 1))
    expected = (np.sqrt(2) / 3) * (8 * np.pi * 0.23753295796592) ** 1.5 * np.sqrt(np.sqrt(3) / 4)
    assert v == pytest.approx([expected, expected])


def test_ponzano_regge():
    for s in (20, 40):
        tj = [int(2 * s * x) for x in (1, 1, 1, 1.1, 1.2, 1.3)]
        assert wigner_6j(*tj) == pytest.approx(spin_foam.ponzano_regge_asymptotic(tj), abs=2e-4)


def test_black_hole_entropy_slope():
    g = black_hole.solve_gamma()
    A, N = black_hole.count_microstates(100.0, g, delta=0.05)
    S = np.log(N)
    m = A > 40
    slope = np.linalg.lstsq(np.c_[A[m], np.log(A[m]), np.ones(m.sum())], S[m], rcond=None)[0][0]
    assert slope == pytest.approx(0.25, abs=0.01)


def test_bounce_matches_analytic():
    tr = cosmology.bounce(0.0, t_max=2.0, n=800)
    a, rho = cosmology.massless_analytic(tr.t)
    assert np.allclose(tr.a, a, rtol=1e-8) and np.allclose(tr.rho, rho, rtol=1e-8)
    assert tr.rho.max() <= RHO_CRIT * (1 + 1e-9)
    assert tr.a.min() == pytest.approx(1.0)


def test_spin_network_sampling_keeps_gauge_invariance():
    net = SpinNetwork.cubic(3, two_j=2).sample(steps=300, beta=0.01, max_two_j=4, seed=0)
    assert net.is_admissible()
    assert net.total_area() > 0 and net.total_volume() >= 0
