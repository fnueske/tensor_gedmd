
from __future__ import annotations

from typing import Any, Optional

import numpy as np




from tensor_gedmd.basis_sets.basis_set import BasisSet


class RandomFourierFeatures(BasisSet):
    r"""
    Minimal cosine Random Fourier Features basis.

    Parameters
    ----------
    omega : array-like, shape (n, d)
        Frequency matrix.
    b : array-like, optional, shape (n,)
        Phase offsets. If None, uses deterministic linspace in [0, 2pi).
    dtype : float dtype
        Computation dtype.

    Dimension conventions
    ---------------------
    Input x:  (m, d)
    Output:   (m, n)
    Gradient: (m, n, d)
    """

    def __init__(self, *, omega: Any, b: Optional[Any] = None, dtype=np.float64) -> None:
        super().__init__(dtype=dtype)

        omega = np.asarray(omega, dtype=self._dtype)
        if omega.ndim != 2:
            raise ValueError("omega must have shape (n, d).")
        self.omega = omega  # (n, d)

        self.n = int(omega.shape[0])
        self.d = int(omega.shape[1])

        if b is None:
            # deterministic phases (no randomness)
            self.b = np.linspace(0.0, 2.0 * np.pi, self.n, endpoint=False).astype(self._dtype)
        else:
            b_arr = np.asarray(b, dtype=self._dtype).reshape(-1)
            if b_arr.shape[0] != self.n:
                raise ValueError(f"b must have shape (n,) with n={self.n}; got {b_arr.shape}.")
            self.b = b_arr

        self.scale = np.sqrt(2.0 / self.n).astype(self._dtype)

    def __call__(self, x: Any) -> np.ndarray:
        x_arr = self._format_input(x, expected_dim=self.d)  # (m, d)
        # (m, d) @ (d, n) -> (m, n)
        arg = x_arr @ self.omega.T + self.b[None, :]
        return self.scale * np.cos(arg)

    def _gradient(self, x: Any) -> np.ndarray:
        x_arr = self._format_input(x, expected_dim=self.d)  # (m, d)
        arg = x_arr @ self.omega.T + self.b[None, :]  # (m, n)

        # d/dx cos(omega x + b) = -sin(...) * omega
        # (-scale*sin(arg)) -> (m, n)
        # multiply by omega -> (m, n, d)
        sin_term = np.sin(arg)[:, :, None]  # (m, n, 1)
        return (-self.scale * sin_term) * self.omega[None, :, :]  # (m, n, d)














