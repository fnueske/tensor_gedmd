from __future__ import annotations

import numpy as np
import pytest

from tensor_gedmd.systems.decoupled_potential_3d import Decoupled_3d
from tensor_gedmd.systems.ol_generic import OL_Generic


class TestDecoupled3D:
    def test_potential_shape_and_value(self) -> None:
        LS = Decoupled_3d(beta=1.0, k=1.0, l=1.0, m=5.0)
        x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        V = LS.potential(x)
        assert V.shape == (2,)
        # At (1, 1, 0): both wells are at their minimum (V=0 each), harmonic term is 0.
        assert np.isclose(V[0], 0.0)
        # At (0, 0, 2): k*(0-1)^2 + l*(0-1)^2 + 0.5*m*4 = 1 + 1 + 10 = 12
        assert np.isclose(V[1], 1.0 + 1.0 + 0.5 * 5.0 * 4.0)

    def test_gradient_matches_numerical_differentiation(self) -> None:
        """
        Independent correctness check: the analytic gradient should match a
        finite-difference approximation of the potential, for arbitrary
        (non-special-cased) points.
        """
        rng = np.random.default_rng(0)
        LS = Decoupled_3d(beta=1.0, k=1.3, l=0.7, m=2.5)
        x = rng.normal(size=(3, 20))

        analytic = LS.gradient(x)

        eps = 1e-6
        numeric = np.zeros_like(analytic)
        for dim in range(3):
            x_plus = x.copy()
            x_plus[dim, :] += eps
            x_minus = x.copy()
            x_minus[dim, :] -= eps
            numeric[dim, :] = (LS.potential(x_plus) - LS.potential(x_minus)) / (2 * eps)

        assert np.allclose(analytic, numeric, atol=1e-4)

    def test_gradient_shape(self) -> None:
        LS = Decoupled_3d(beta=1.0, k=1.0, l=1.0, m=5.0)
        x = np.zeros((3, 7))
        dV = LS.gradient(x)
        assert dV.shape == (3, 7)

    def test_is_ol_generic_subclass(self) -> None:
        assert issubclass(Decoupled_3d, OL_Generic)

    def test_simulate_single_trajectory_shape(self) -> None:
        LS = Decoupled_3d(beta=1.0, k=1.0, l=1.0, m=5.0)
        x0 = np.ones(3)
        X = LS.simulate(x0, m=50, dt=1e-3)
        assert X.shape == (3, 51)

    def test_simulate_multi_trajectory_shape(self) -> None:
        LS = Decoupled_3d(beta=1.0, k=1.0, l=1.0, m=5.0)
        x0 = np.ones((4, 3))
        X = LS.simulate(x0, m=50, dt=1e-3)
        assert X.shape == (4, 3, 51)

    def test_simulate_stays_bounded(self) -> None:
        """
        Sanity check: with a reasonable dt, the double-well + harmonic
        dynamics shouldn't blow up over a short trajectory.
        """
        LS = Decoupled_3d(beta=1.0, k=1.0, l=1.0, m=5.0)
        x0 = np.ones(3)
        X = LS.simulate(x0, m=2000, dt=1e-3)
        assert np.all(np.isfinite(X))
        assert np.max(np.abs(X)) < 10.0

    def test_cannot_instantiate_ol_generic_directly(self) -> None:
        with pytest.raises(TypeError):
            OL_Generic(beta=1.0)
