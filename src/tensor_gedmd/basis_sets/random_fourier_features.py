
from __future__ import annotations

from typing import Any, Optional

import numpy as np


from tensor_gedmd.basis_sets.basis_sets import BasisSet


class RandomFourierFeatures(BasisSet):
    r"""
    Cosine Random Fourier Features basis.

    Parameters
    ----------
    omega : array-like, shape (n, d)
        Frequency matrix. Typically d = 1 (one physical dimension per basis
        instance, as consumed by Transformed_Data_Tensor_TT / TgStiffnessOperator).
    b : array-like, optional, shape (n,)
        Phase offsets. If None, uses a deterministic linspace in [0, 2*pi).
        If given, it is actually used (unlike some reference implementations
        that accept ``b`` but silently overwrite it with the default).
    dtype : float dtype
        Computation dtype.

    Dimension conventions
    ---------------------
    Input X:    (d, m)
    Output:     (n, m)
    Derivative: (n, d, m)
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

    def __call__(self, X: Any) -> np.ndarray:
        X_arr = self._format_input(X, expected_dim=self.d)  # (d, m)
        # (n, d) @ (d, m) + (n, 1) -> (n, m)
        arg = self.omega @ X_arr + self.b[:, None]
        return self.scale * np.cos(arg)

    def _gradient(self, X: Any) -> np.ndarray:
        X_arr = self._format_input(X, expected_dim=self.d)  # (d, m)
        arg = self.omega @ X_arr + self.b[:, None]  # (n, m)

        # d/dX cos(omega @ X + b) = -sin(arg) * omega, per input dimension.
        # (-scale*sin(arg)) -> (n, 1, m); omega -> (n, d, 1); product -> (n, d, m)
        sin_term = np.sin(arg)[:, None, :]  # (n, 1, m)
        return (-self.scale * sin_term) * self.omega[:, :, None]  # (n, d, m)
