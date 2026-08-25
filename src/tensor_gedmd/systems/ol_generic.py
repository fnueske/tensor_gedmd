"""
Generic Overdamped Langevin dynamics.

Base class for stochastic dynamics given by overdamped Langevin dynamics
in a potential V, at inverse temperature beta:

    dX_t = -grad V(X_t) dt + sqrt(2 / beta) dW_t.

This class only provides a routine to generate simulation data (Euler-
Maruyama integration). Subclasses provide the potential and its gradient
for a specific system -- see decoupled_potential_3d.Decoupled_3d for an
example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class OL_Generic(ABC):
    """
    Overdamped Langevin dynamics base class.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    """

    def __init__(self, beta: float):
        self.beta = beta

    def simulate(self, x0: np.ndarray, m: int, dt: float) -> np.ndarray:
        """
        Generate a trajectory of the overdamped Langevin dynamics using a
        first-order Euler-Maruyama scheme.

        Parameters
        ----------
        x0 : np.ndarray, shape (d,) or (msim, d)
            Initial value(s). A single d-dimensional vector, or an array
            of msim initial positions (msim independent simulations).
        m : int
            Number of time steps to return (not including the initial
            value).
        dt : float
            Integration time step.

        Returns
        -------
        X : np.ndarray, shape (d, m+1) or (msim, d, m+1)
            Simulated trajectory/trajectories.
        """
        if len(x0.shape) == 1:
            msim = 1
            x0 = x0[None, :]
        else:
            msim = x0.shape[0]
        d = x0.shape[1]

        X = np.zeros((msim, d, m + 1))
        X[:, :, 0] = x0.copy()
        X_old = x0.T

        for t in range(1, m + 1):
            X_new = X_old - self.gradient(X_old) * dt + \
                np.sqrt(2 * dt / self.beta) * np.random.randn(d, msim)
            X[:, :, t] = (X_new.T).copy()
            X_old = X_new

        if msim == 1:
            X = X[0, :, :]

        return X

    @abstractmethod
    def potential(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the potential at data points x.

        Parameters
        ----------
        x : np.ndarray, shape (d, m)
            m positions in d-dimensional Euclidean space.

        Returns
        -------
        V : np.ndarray, shape (m,)
            Potential values.
        """
        raise NotImplementedError("Method potential not available in abstract class.")

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the potential gradient at data points x.

        Parameters
        ----------
        x : np.ndarray, shape (d, m)
            m positions in d-dimensional Euclidean space.

        Returns
        -------
        dV : np.ndarray, shape (d, m)
            Gradient of the potential.
        """
        raise NotImplementedError("Method gradient not available in abstract class.")
