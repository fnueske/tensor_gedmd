# product_basis.py
"""
Tensor-product basis utilities.

This module defines :class:`~ProductBasis`, a dense tensor-product
(Kronecker-style) basis constructed from a list of one-dimensional basis
sets. It is intended as a dense reference/comparison basis (e.g. for a
"vanilla", non-TT gEDMD baseline) rather than for feeding the TT pipeline,
which consumes per-dimension psi/dpsi lists directly.

Notes
-----
- Each 1D basis is assumed to act on a single physical dimension, taking
  inputs shaped ``(1, m)`` and returning features shaped ``(n, m)``.
- Gradients are assumed to be returned by each 1D basis as ``(n, 1, m)``.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np


from tensor_gedmd.basis_sets.basis_sets import BasisSet


class ProductBasis(BasisSet):
    r"""
    Dense tensor-product basis built from a list of 1D basis sets.

    The resulting feature map is the per-sample tensor (outer/Kronecker)
    product of the 1D feature maps across dimensions.

    Parameters
    ----------
    basis_list
        List of 1D basis sets, one per coordinate/dimension. Each element must
        be a :class:`~BasisSet` instance. All bases must have the same number
        of features ``n``.
    dtype
        Numeric dtype used for internal arrays and outputs.

    Attributes
    ----------
    basis_list : list[BasisSet]
        The stored list of 1D basis sets.
    d : int
        Input dimension (number of coordinates / number of bases in
        ``basis_list``).
    n : int
        Total number of tensor-product features, equal to ``(n0 ** d)``, where
        ``n0`` is the number of features of each 1D basis.

    Dimension conventions
    ---------------------
    - Input ``X``:       ``(d, m)``
    - Output ``Psi(X)``: ``(n0**d, m)``
    - Gradient:          ``(n0**d, d, m)``
    """

    def __init__(self, basis_list: List[BasisSet], dtype=np.float64) -> None:
        super().__init__(dtype=dtype)

        if not isinstance(basis_list, (list, tuple)) or len(basis_list) == 0:
            raise ValueError("basis_list must be a non-empty list/tuple of BasisSet objects.")

        self.basis_list = list(basis_list)
        self.d = len(self.basis_list)

        n0 = getattr(self.basis_list[0], "n", None)
        if n0 is None:
            raise ValueError("basis_list[0].n must be set (number of features).")

        for i, b in enumerate(self.basis_list):
            if not isinstance(b, BasisSet):
                raise TypeError(f"basis_list[{i}] is not a BasisSet.")
            if getattr(b, "n", None) != n0:
                raise ValueError("All basis functions in basis_list must have the same n.")

        self.n0 = int(n0)
        self.n = self.n0 ** self.d  # total number of product features

    # -------------------------
    # Internal tensor utilities
    # -------------------------
    @staticmethod
    def _outer_features(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Per-sample outer product over feature axes.

        Parameters
        ----------
        A : (p, m)
        B : (q, m)

        Returns
        -------
        (p*q, m) : the flattened outer product A[:, l] (x) B[:, l] per sample l.
        """
        p, m = A.shape
        q, m2 = B.shape
        if m != m2:
            raise ValueError(f"Sample counts must match: {m} != {m2}.")
        return (A[:, None, :] * B[None, :, :]).reshape(p * q, m)

    # -------------------------
    # BasisSet API
    # -------------------------
    def __call__(self, X: Any) -> np.ndarray:
        """
        Evaluate the tensor-product basis at inputs ``X``.

        Parameters
        ----------
        X
            Input array-like of shape ``(d, m)``.

        Returns
        -------
        np.ndarray
            Basis evaluation Psi(X) with shape ``(n0**d, m)``.
        """
        X_arr = self._format_input(X, expected_dim=self.d)  # (d, m)

        Psi = np.asarray(self.basis_list[0](X_arr[0:1, :]), dtype=self._dtype)  # (n0, m)
        if Psi.shape[1] != X_arr.shape[1]:
            raise ValueError("Basis evaluation must preserve the sample count m.")

        for j in range(1, self.d):
            psi_j = np.asarray(self.basis_list[j](X_arr[j:j + 1, :]), dtype=self._dtype)  # (n0, m)
            Psi = self._outer_features(Psi, psi_j)  # (n0**(j+1), m)

        return Psi

    def _gradient(self, X: Any) -> np.ndarray:
        """
        Evaluate the gradient of the tensor-product basis with respect to X.

        Parameters
        ----------
        X
            Input array-like of shape ``(d, m)``.

        Returns
        -------
        np.ndarray
            Gradient array with shape ``(n0**d, d, m)``.
        """
        X_arr = self._format_input(X, expected_dim=self.d)  # (d, m)
        m = X_arr.shape[1]

        # Precompute all 1D evaluations psi_j: (n0, m)
        psi_all = []
        for j in range(self.d):
            psi_j = np.asarray(self.basis_list[j](X_arr[j:j + 1, :]), dtype=self._dtype)
            psi_all.append(psi_j)

        out = np.zeros((self.n, self.d, m), dtype=self._dtype)

        for dim in range(self.d):
            grad_dim = self.basis_list[dim].gradient(X_arr[dim:dim + 1, :])  # expected (n0, 1, m)
            grad_dim = np.asarray(grad_dim, dtype=self._dtype)

            if grad_dim.ndim != 3 or grad_dim.shape[0] != self.n0 or grad_dim.shape[1] != 1:
                raise ValueError(
                    f"Expected gradient from basis_list[{dim}] to have shape ({self.n0}, 1, m); "
                    f"got {grad_dim.shape}."
                )

            # Convert (n0, 1, m) -> (n0, m) for the dim we differentiate
            dpsi = grad_dim[:, 0, :]

            G = dpsi if dim == 0 else psi_all[0]
            for j in range(1, self.d):
                factor = dpsi if j == dim else psi_all[j]
                G = self._outer_features(G, factor)

            out[:, dim, :] = G

        return out
