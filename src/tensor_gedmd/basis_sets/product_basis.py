# product_basis.py
"""
Tensor-product basis utilities.

This module defines :class:`~ProductBasis`, a tensor-product (Kronecker-style)
basis constructed from a list of one-dimensional basis sets.

Notes
-----
- The implementation assumes each 1D basis acts on inputs shaped ``(m, 1)``
  and returns features shaped ``(m, n)``.
- Gradients are assumed to be returned by each 1D basis as ``(m, n, 1)``.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

import sys
import os

this_dir = os.path.dirname(os.path.abspath("file"))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath("file")))
sys.path.append("/Users/minakshi/Desktop/")

from Project.tensor_gedmd.src.tensor_gedmd.basis_sets.basis_sets import BasisSet


class ProductBasis(BasisSet):
    r"""
    Tensor-product basis built from a list of 1D basis sets.

    The resulting feature map is the per-sample tensor (outer/Kronecker) product
    of the 1D feature maps across dimensions.

    Parameters
    ----------
    basis_list
        List of 1D basis sets, one per coordinate/dimension. Each element must
        be a :class:`~BasisSet` instance. All bases must have the same number of
        features ``n``.
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

    Assumptions
    -----------
    - ``basis_list`` has length ``d`` (one basis per coordinate).
    - Each ``basis_list[j]`` expects inputs of shape ``(m, 1)`` and returns
      ``(m, n)``, and ``gradient`` returns ``(m, n, 1)``.
    - All bases have the same number of features ``n``.

    Dimension conventions
    ---------------------
    - Input ``x``:       ``(m, d)``
    - Output ``Psi(x)``: ``(m, n**d)``
    - Gradient:          ``(m, n**d, d)``
    """

    def __init__(self, basis_list: List[BasisSet], dtype=np.float64) -> None:
        """
        Construct a tensor-product basis from 1D basis sets.

        Parameters
        ----------
        basis_list
            Non-empty list/tuple of :class:`~BasisSet` objects.
        dtype
            Numeric dtype used for internal arrays and outputs.

        Raises
        ------
        ValueError
            If ``basis_list`` is empty, or if the first basis does not define
            ``n``, or if the bases do not all share the same ``n``.
        TypeError
            If any element of ``basis_list`` is not a :class:`~BasisSet`.
        """
        super().__init__(dtype=dtype)

        if not isinstance(basis_list, (list, tuple)) or len(basis_list) == 0:
            raise ValueError("basis_list must be a non-empty list/tuple of BasisSet objects.")

        self.basis_list = list(basis_list)
        self.d = len(self.basis_list)

        # Ensure each basis has n defined, and all n match.
        n0 = getattr(self.basis_list[0], "n", None)
        if n0 is None:
            raise ValueError("basis_list[0].n must be set (number of features).")

        for i, b in enumerate(self.basis_list):
            if not isinstance(b, BasisSet):
                raise TypeError(f"basis_list[{i}] is not a BasisSet.")
            if getattr(b, "n", None) != n0:
                raise ValueError("All basis functions in basis_list must have the same n.")

        self.n = int(n0) ** self.d  # total number of product features

    # -------------------------
    # Internal tensor utilities
    # -------------------------
    @staticmethod
    def _tensor_product_features(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute a per-sample outer product over feature axes.

        Given feature matrices ``A`` and ``B`` evaluated at the same batch of
        samples, this returns the tensor-product features for each sample.

        Parameters
        ----------
        A
            Feature matrix of shape ``(m, p)``.
        B
            Feature matrix of shape ``(m, q)``.

        Returns
        -------
        np.ndarray
            Tensor-product feature matrix of shape ``(m, p*q)``.

        Notes
        -----
        This is equivalent to building, for each sample ``i``, the flattened
        outer product ``A[i, :] ⊗ B[i, :]``.
        """
        return (A[:, :, None] * B[:, None, :]).reshape(A.shape[0], A.shape[1] * B.shape[1])

    # -------------------------
    # BasisSet API
    # -------------------------
    def __call__(self, x: Any) -> np.ndarray:
        """
        Evaluate the tensor-product basis at inputs ``x``.

        Parameters
        ----------
        x
            Input array-like of shape ``(m, d)`` (or any format accepted by the
            parent :class:`~BasisSet` input formatter). Here, ``m`` is the batch
            size and ``d`` is the input dimension.

        Returns
        -------
        np.ndarray
            Basis evaluation ``Psi(x)`` with shape ``(m, n**d)``.

        Raises
        ------
        ValueError
            If the basis evaluation does not preserve the batch dimension.
        """
        x_arr = self._format_input(x, expected_dim=self.d)  # (m, d)
        m = x_arr.shape[0]

        # Evaluate first dimension basis: (m, n)
        Psi = self.basis_list[0](x_arr[:, [0]])
        Psi = np.asarray(Psi, dtype=self._dtype)
        if Psi.shape[0] != m:
            raise ValueError("Basis evaluation must preserve batch size m.")

        # Iteratively build tensor product across dimensions
        for j in range(1, self.d):
            psi_j = self.basis_list[j](x_arr[:, [j]])  # (m, n)
            psi_j = np.asarray(psi_j, dtype=self._dtype)
            Psi = self._tensor_product_features(Psi, psi_j)  # (m, n^k)

        # Psi shape: (m, n^d)
        return Psi

    def _gradient(self, x: Any) -> np.ndarray:
        """
        Evaluate the gradient of the tensor-product basis with respect to ``x``.

        Parameters
        ----------
        x
            Input array-like of shape ``(m, d)`` (or any format accepted by the
            parent :class:`~BasisSet` input formatter).

        Returns
        -------
        np.ndarray
            Gradient array with shape ``(m, n**d, d)``, where the last axis
            corresponds to partial derivatives with respect to each coordinate.

        Raises
        ------
        ValueError
            If any 1D basis gradient does not have the expected shape
            ``(m, n, 1)``.
        """
        x_arr = self._format_input(x, expected_dim=self.d)  # (m, d)
        m = x_arr.shape[0]

        # Precompute all 1D evaluations psi_j: (m, n)
        psi_all = []
        for j in range(self.d):
            psi_j = np.asarray(self.basis_list[j](x_arr[:, [j]]), dtype=self._dtype)  # (m, n)
            psi_all.append(psi_j)

        out = np.zeros((m, self.n, self.d), dtype=self._dtype)

        # For each coordinate dim, use derivative basis there and normal basis elsewhere
        for dim in range(self.d):
            grad_dim = self.basis_list[dim].gradient(x_arr[:, [dim]])  # expected (m, n, 1)
            grad_dim = np.asarray(grad_dim, dtype=self._dtype)

            if grad_dim.ndim != 3 or grad_dim.shape[0] != m or grad_dim.shape[2] != 1:
                raise ValueError(
                    f"Expected gradient from basis_list[{dim}] to have shape (m, n, 1); got {grad_dim.shape}."
                )

            # Convert (m, n, 1) -> (m, n) for the dim we differentiate
            dpsi = grad_dim[:, :, 0]

            # Build tensor product: start with first coordinate
            if dim == 0:
                G = dpsi
            else:
                G = psi_all[0]

            for j in range(1, self.d):
                if j == dim:
                    factor = dpsi
                else:
                    factor = psi_all[j]
                G = self._tensor_product_features(G, factor)

            # Place into the right gradient column
            out[:, :, dim] = G

        return out



# import numpy as np
# from py_molkin.models.basis_sets import BasisSet



# class Product_Basis:
#     def __init__(self, basis_list):
#         self.basis_list = basis_list
#         self.d = len(basis_list)
#         self.n = basis_list[0].n
    


#     def __call__(self, X=None):
#         # Check if this is a derivative-aware basis
#         if getattr(self.basis_list[0], "d", None) is None:
#             # ---- Standard basis: shape (n**p, m) ----
#             m = X.shape[1]
#             Psi_full = self.basis_list[0](X[0, :][None, :])  # (n, m)  # initialize
#             for ii in range(1, self.d):
#                 psi_i = self.basis_list[ii](X[ii, :][None, :])  # (n, m)
#                 Psi_full = np.einsum("ij,kj->ikj", Psi_full, psi_i)  # (n^ii, n, m)
#                 Psi_full = Psi_full.reshape((self.n**(ii+1), m))
#             return Psi_full

  

#     def diff(self, X):
#         """
#         Compute derivatives wrt all coordinates.
#         Returns shape (n^d, d, m)
#         """
#         m = X.shape[1]
#         n = self.n
#         d = self.d
#         out = np.zeros((n**d, d, m))

#         for dim in range(d):
#             # derivative for this dimension
#             dPsi = self.basis_list[dim].derivative(X[dim:dim+1, :])  # (n,1,m)
#             Psi_list_temp = []

#             for j in range(d):
#                 if j == dim:
#                     Psi_list_temp.append(dPsi)  # (n,1,m)
#                 else:
#                     psi_j = self.basis_list[j](X[j:j+1, :])  # (n,m)
#                     psi_j = psi_j[:, None, :]  # (n,1,m)
#                     Psi_list_temp.append(psi_j)

#             # Compute full tensor-product using iterative broadcasting
#             Psi_dim = Psi_list_temp[0]  # (n,1,m)
#             for k in range(1, d):
#                 # broadcast multiply along feature axis
#                 shape_prev = Psi_dim.shape[0]
#                 shape_next = Psi_list_temp[k].shape[0]
#                 Psi_dim = (Psi_dim[:, None, :, :] * Psi_list_temp[k][None, :, :, :]).reshape(shape_prev*shape_next, 1, m)

#             out[:, dim, :] = Psi_dim[:, 0, :]

#         return out
