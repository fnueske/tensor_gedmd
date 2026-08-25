"""
Three-dimensional decoupled potential.

    V(x, y, z) = k*(x^2-1)^2 + l*(y^2-1)^2 + (1/2)*m*z^2

Dynamics given by the overdamped Langevin SDE:

    dX_t = -grad V(X_t) dt + sqrt(2 / beta) dW_t.

This is the Lemon-Slice toy system used in lemon_slice_3d/run_calculation.py
and introduction/run_calculation.py -- an isotropic double-well in the first
two coordinates, a harmonic well in the third.

Reference: Bittracher et al., "Transition manifolds of complex metastable
systems", Journal of Nonlinear Science, 2018.
"""

from __future__ import annotations

import numpy as np

from tensor_gedmd.systems.ol_generic import OL_Generic


class Decoupled_3d(OL_Generic):
    """
    Stochastic dynamics in the three-dimensional decoupled potential.

    Parameters
    ----------
    beta : float
        Inverse temperature controlling noise intensity.
    k, l, m : float
        Coefficients for the terms in the potential function.
    """

    def __init__(self, beta: float, k: float, l: float, m: float):
        self.beta = beta
        self.k = k
        self.l = l
        self.m = m

    def potential(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the potential energy V(x, y, z).

        Parameters
        ----------
        x : np.ndarray, shape (3, m)
            Coordinates.

        Returns
        -------
        V : np.ndarray, shape (m,)
        """
        return self.k * (x[0, :] ** 2 - 1) ** 2 + self.l * (x[1, :] ** 2 - 1) ** 2 \
            + 0.5 * self.m * x[2, :] ** 2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the potential at positions x.

        Parameters
        ----------
        x : np.ndarray, shape (3, m)

        Returns
        -------
        dV : np.ndarray, shape (3, m)
        """
        dV = np.zeros((3, x.shape[1]))
        dV[0, :] = 4 * self.k * x[0, :] * (x[0, :] ** 2 - 1)
        dV[1, :] = 4 * self.l * x[1, :] * (x[1, :] ** 2 - 1)
        dV[2, :] = self.m * x[2, :]
        return dV
