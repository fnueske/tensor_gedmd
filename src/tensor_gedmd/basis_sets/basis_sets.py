
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

    For an input batch x with shape (m, d):
      - __call__(x)   -> (m, n)
      - gradient(x)   -> (m, n, d)
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
    def _format_input(self, x: Any, *, expected_dim: int) -> np.ndarray:
        """Ensure x has shape (m, expected_dim)."""
        x_arr = np.asarray(x, dtype=self._dtype)
        if x_arr.ndim == 1:
            # treat as single sample
            x_arr = x_arr[None, :]
        if x_arr.ndim != 2 or x_arr.shape[1] != expected_dim:
            raise ValueError(f"x must have shape (m, {expected_dim}); got {x_arr.shape}.")
        return x_arr

    # ------------------------
    # Public API
    # ------------------------
    @abstractmethod
    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate basis functions. Must return shape (m, n)."""
        raise NotImplementedError

    def gradient(self, x: Any) -> np.ndarray:
        """
        Gradient of basis functions w.r.t. x.

        Returns shape (m, n, d).
        """
        return self._gradient(x)

    # ------------------------
    # Subclass hooks
    # ------------------------
    def _gradient(self, x: Any) -> np.ndarray:
        """Subclasses should override if gradients are needed."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _gradient().")






















