
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BasisSet(ABC):
    r"""
    Minimal base class for basis sets.

    Dimension conventions
    ---------------------
    Let:
      - d = input/state dimension
      - n = number of basis functions
      - m = batch size

    For an input batch X with shape (d, m):
      - __call__(X)   -> (n, m)
      - gradient(X)   -> (n, d, m)

    This (features, samples) orientation matches what
    ``tensor_gedmd.reps.transformed_data_tensor.Transformed_Data_Tensor_TT``,
    ``tensor_gedmd.reps.stiffness_tt.TgStiffnessOperator``, and
    ``tensor_gedmd.algorithms.mat_vec_prod.compute_A_r`` all expect for
    ``psi``/``dpsi``, so basis evaluations can be used directly without any
    manual transposition.
    """

    def __init__(self, *, dtype=np.float64) -> None:
        self._dtype = np.dtype(dtype)
        if self._dtype.kind != "f":
            raise TypeError("dtype must be a floating-point type.")
        self.d: Optional[int] = None
        self.n: Optional[int] = None

    # ------------------------
    # Utilities
    # ------------------------
    def _format_input(self, X: Any, *, expected_dim: int) -> np.ndarray:
        """Ensure X has shape (expected_dim, m)."""
        X_arr = np.asarray(X, dtype=self._dtype)
        if X_arr.ndim == 1:
            # treat as a single physical dimension, m samples
            X_arr = X_arr[None, :]
        if X_arr.ndim != 2 or X_arr.shape[0] != expected_dim:
            raise ValueError(f"X must have shape ({expected_dim}, m); got {X_arr.shape}.")
        return X_arr

    # ------------------------
    # Public API
    # ------------------------
    @abstractmethod
    def __call__(self, X: Any) -> np.ndarray:
        """Evaluate basis functions. Must return shape (n, m)."""
        raise NotImplementedError

    def gradient(self, X: Any) -> np.ndarray:
        """
        Gradient of basis functions w.r.t. X.

        Returns shape (n, d, m).
        """
        return self._gradient(X)

    # ------------------------
    # Subclass hooks
    # ------------------------
    def _gradient(self, X: Any) -> np.ndarray:
        """Subclasses should override if gradients are needed."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _gradient().")






















