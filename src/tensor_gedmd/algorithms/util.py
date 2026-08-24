# -*- coding: utf-8 -*-

"""
TT Core Truncation Utilities
============================

This module provides utility functions for truncating Tensor Train (TT)
cores using either singular value truncation rules or CUR-based low-rank
approximation.

Functions
---------
_truncate_rank
    Compute truncation rank from singular values using rank cap and/or
    relative tail-energy tolerance.

truncate_tt_core_cur
    Truncate a single TT core using a CUR-based low-rank approximation.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


# ============================================================
# Rank truncation rule from singular values
# ============================================================

def _truncate_rank(
    s: np.ndarray,
    rmax: int = None,
    tol: float = 0,
) -> int:
    """
    Compute truncated rank from singular values.

    This function determines a truncation rank based on two criteria:
    - Maximum allowed rank (rmax)
    - Relative tail-energy tolerance (tol)

    The retained rank is the smallest rank satisfying both constraints.

    Parameters
    ----------
    s : np.ndarray
        1D array of singular values (sorted in descending order).
    rmax : int, optional
        Maximum allowed rank. If None, no rank cap is applied.
    tol : float, optional
        Relative tail-energy truncation tolerance.
        If tol == 0, no tolerance-based truncation is applied.

    Returns
    -------
    int
        Truncated rank (at least 1).

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    The truncation based on tolerance uses the condition

        sum_{i > r} s_i^2 / sum_i s_i^2 <= tol.

    This corresponds to Frobenius-norm truncation.
    """
    s = np.asarray(s)

    if s.ndim != 1:
        raise ValueError("s must be a 1D array of singular values.")

    n = len(s)
    if n == 0:
        return 1

    if rmax is not None and rmax < 1:
        raise ValueError(f"rmax must be at least 1, got {rmax}.")

    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}.")

    rank_rmax = n if rmax is None else min(rmax, n)
    rank_tol = n

    if tol > 0:
        sq = s ** 2
        total_energy = sq.sum()

        if total_energy == 0:
            return 1

        tail_energy = np.cumsum(sq[::-1])[::-1]
        rel_tail_energy = tail_energy / total_energy

        idx = np.where(rel_tail_energy <= tol)[0]
        if idx.size > 0:
            rank_tol = max(1, int(idx[0]))
        else:
            rank_tol = n

    return max(1, min(rank_rmax, rank_tol, n))


# ============================================================
# CUR-based TT core truncation
# ============================================================

def truncate_tt_core_cur(
    core: np.ndarray,
    max_rank: int = 100,
    tol: float = 1e-10,
    sample_rows: Optional[int] = None,
    sample_cols: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncate a TT core using a CUR-based low-rank approximation.

    The TT core of shape (r_left, n, r_right) is reshaped into a matrix

        A ∈ R^{(r_left * n) × r_right},

    and approximated via a CUR decomposition with randomized row/column
    sampling. The approximation is then compressed to rank at most max_rank.

    Parameters
    ----------
    core : np.ndarray
        TT core of shape (r_left, n, r_right).
    max_rank : int, default=100
        Maximum retained rank.
    tol : float, default=1e-10
        Cutoff used in the pseudoinverse computation.
    sample_rows : int, optional
        Number of sampled rows from the unfolded matrix.
        If None, defaults to min(max_rank + 50, r_left * n).
    sample_cols : int, optional
        Number of sampled columns from the unfolded matrix.
        If None, defaults to min(max_rank + 50, r_right).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    truncated_core : np.ndarray
        Truncated TT core of shape (r_left, n, r_new).
    transfer_matrix : np.ndarray
        Matrix of shape (r_new, r_right) to be absorbed into the next TT core.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    The returned matrices satisfy approximately

        A ≈ A_trunc @ transfer_matrix,

    where
        A = core.reshape(r_left * n, r_right)
        A_trunc = truncated_core.reshape(r_left * n, r_new)

    In a TT truncation sweep, the transfer_matrix should be contracted
    into the next TT core.
    """
    core = np.asarray(core)

    if core.ndim != 3:
        raise ValueError(
            f"core must have shape (r_left, n, r_right), got {core.shape}."
        )

    if max_rank < 1:
        raise ValueError(f"max_rank must be >= 1, got {max_rank}.")

    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}.")

    rng = np.random.default_rng(random_state)

    r_left, n, r_right = core.shape
    A = core.reshape(r_left * n, r_right)

    n_rows_total, n_cols_total = A.shape

    if sample_rows is None:
        sample_rows = min(max_rank + 50, n_rows_total)
    if sample_cols is None:
        sample_cols = min(max_rank + 50, n_cols_total)

    sample_rows = min(sample_rows, n_rows_total)
    sample_cols = min(sample_cols, n_cols_total)

    row_idx = rng.choice(n_rows_total, size=sample_rows, replace=False)
    col_idx = rng.choice(n_cols_total, size=sample_cols, replace=False)

    C = A[:, col_idx]
    R = A[row_idx, :]
    W = A[np.ix_(row_idx, col_idx)]

    U_mid = np.linalg.pinv(W, rcond=tol)
    CU = C @ U_mid

    if CU.shape[0] >= CU.shape[1]:
        Uc, Sc, Vhc = np.linalg.svd(CU, full_matrices=False)
        r_new = min(max_rank, Sc.shape[0])

        Uc = Uc[:, :r_new]
        Sc = Sc[:r_new]
        Vhc = Vhc[:r_new, :]

        transfer_matrix = (Sc[:, None] * Vhc) @ R
        truncated_core = Uc.reshape(r_left, n, r_new)
    else:
        Ur, Sr, Vhr = np.linalg.svd(R, full_matrices=False)
        r_new = min(max_rank, Sr.shape[0])

        Ur = Ur[:, :r_new]
        Sr = Sr[:r_new]
        Vhr = Vhr[:r_new, :]

        Z = C @ (U_mid @ (Ur * Sr[None, :]))
        truncated_core = Z.reshape(r_left, n, r_new)
        transfer_matrix = Vhr

    return truncated_core, transfer_matrix







# ============================================================
# Eigenvalue filtering
# ============================================================

def filter_ev(d: np.ndarray, W: np.ndarray, eps1: float = -np.inf, eps2: float = np.inf):
    """
    Sort eigenpairs by ascending real part of the eigenvalue and keep only
    those whose real part lies strictly within (eps1, eps2).

    Typically used after eigendecomposing a reduced generator matrix (e.g.
    from ``compute_A_r``) to discard spurious/unstable eigenvalues before
    interpreting the remaining ones as approximate generator eigenfunctions.

    Parameters
    ----------
    d : np.ndarray
        1D array of eigenvalues (may be complex).
    W : np.ndarray
        2D array of eigenvectors, shape (n, len(d)), with W[:, i]
        corresponding to d[i].
    eps1 : float, default=-inf
        Lower bound (exclusive) on the real part of retained eigenvalues.
    eps2 : float, default=inf
        Upper bound (exclusive) on the real part of retained eigenvalues.

    Returns
    -------
    d_filtered : np.ndarray
        Retained eigenvalues, sorted ascending by real part.
    W_filtered : np.ndarray
        Corresponding eigenvectors, shape (n, len(d_filtered)).
    """
    d = np.asarray(d)
    W = np.asarray(W)

    if d.ndim != 1:
        raise ValueError(f"d must be a 1D array of eigenvalues, got shape {d.shape}.")
    if W.ndim != 2 or W.shape[1] != d.shape[0]:
        raise ValueError(
            f"W must have shape (n, len(d)) = (n, {d.shape[0]}), got {W.shape}."
        )

    ind = np.argsort(np.real(d))
    d = d[ind]
    W = W[:, ind]

    ind = np.where(np.logical_and(np.real(d) > eps1, np.real(d) < eps2))[0]
    return d[ind], W[:, ind]


# ============================================================
# Dense "vanilla" gEDMD reference method (reversible generator)
#
# Ports of dmp_methods.Util.util.whitening_transform and
# dmp_methods.gEDMD.gEDMD.evaluate_generator_rev, the actual reference
# implementation these were originally modeled on.
#
# whitening_transform and evaluate_generator_rev are faithful, unmodified
# ports of the general library (neither mean-centers PhiX or divides by
# the sample count m anywhere).
#
# spectral_analysis_gedmd_rev below is NOT the general library's version,
# though -- it's this project's own mean-free variant: PhiX is
# mean-centered before whitening (see the function's docstring). This is a
# deliberate, project-specific choice, not the plain dmp_methods.gEDMD.gEDMD
# behavior, and every dense/"Matrix" reference-method call in
# examples/publication/ relies on this mean-free behavior specifically.
# ============================================================

def whitening_transform(
    PhiX: np.ndarray,
    tol: float,
    rmin: int = 0,
    return_svd: bool = False,
    complex: bool = False,
):
    """
    Compute whitening transformation of the mass matrix based on truncated
    singular value decomposition.

    Parameters
    ----------
    PhiX : np.ndarray, shape (n, m)
        Functional (lifted) time series: n basis functions evaluated at m
        data points.
    tol : float
        Relative truncation threshold for the SVD (singular values below
        ``tol * s[0]`` are dropped).
    rmin : int, default=0
        Minimum rank to retain regardless of tol.
    return_svd : bool, default=False
        If True, also return the (possibly complex-conjugated) SVD
        components ``(U, s, V)`` used to build L.
    complex : bool, default=False
        If True, use the complex conjugate when forming V.

    Returns
    -------
    L : np.ndarray, shape (n, r)
        Linear transformation to a whitened reduced basis, such that
        ``L.T @ (PhiX @ PhiX.T) @ L`` is (approximately) the identity.
    (U, s, V) : tuple, only if return_svd=True
        Truncated SVD components of PhiX (V has shape (m, r)).
    """
    from scipy.linalg import svd

    U, s, V = svd(PhiX, full_matrices=False)
    ind = np.where(s / s[0] >= tol)[0]
    r = np.maximum(ind.shape[0], rmin)
    U = U[:, :r]
    s = s[:r]
    if complex:
        V = V[:r, :].conj().T
    else:
        V = V[:r, :].T
    L = U * (s ** (-1))[None, :]
    if return_svd:
        return L, (U, s, V)
    return L


def evaluate_generator_rev(X: np.ndarray, phi, a, is_complex: bool = False) -> np.ndarray:
    """
    Compute the stiffness matrix for the reversible generator on a given
    finite-dimensional basis set.

    Parameters
    ----------
    X : np.ndarray, shape (d, m)
        m data points in d-dimensional space.
    phi : object with a ``.gradient(X)`` method
        Basis set; ``phi.gradient(X)`` must return shape (n, d, m) (e.g.
        ``tensor_gedmd.basis_sets.product_basis.ProductBasis``).
    a : float or np.ndarray, shape (d, d, m)
        Diffusion tensor. A Python/NumPy scalar means a constant isotropic
        diffusion; an array means a samplewise diffusion tensor.
    is_complex : bool, default=False
        If True, conjugate the left factor (for a complex basis set).

    Returns
    -------
    A_L : np.ndarray, shape (n, n)
        Stiffness matrix for the reversible generator.
    """
    dPhiX = phi.gradient(X)  # (n, d, m)

    if np.isscalar(a):
        if is_complex:
            A_L = -0.5 * a * np.tensordot(dPhiX.conj(), dPhiX, axes=[(1, 2), (1, 2)])
        else:
            A_L = -0.5 * a * np.tensordot(dPhiX, dPhiX, axes=[(1, 2), (1, 2)])
    else:
        if is_complex:
            A_L = -0.5 * np.einsum("ril, ijl, sjl -> rs", dPhiX.conj(), a, dPhiX, optimize=True)
        else:
            A_L = -0.5 * np.einsum("ril, ijl, sjl -> rs", dPhiX, a, dPhiX, optimize=True)

    return A_L


def spectral_analysis_gedmd_rev(X: np.ndarray, phi, nev: int, a, tol: float = 0.0, eps_ev: float = 0.0):
    """
    Spectral decomposition of the reversible Koopman generator on a
    finite-dimensional basis set, via whitening transform + eigh.

    Mean-free variant: PhiX is mean-centered (per basis function, across
    samples) before whitening. This is a project-specific choice -- the
    general dmp_methods.gEDMD.gEDMD library does not mean-center -- kept
    here deliberately because every dense/"Matrix" reference-method call in
    examples/publication/ relies on it: mean-centering PhiX before
    whitening is what keeps the trivial constant-function eigenvalue
    (every generator has exactly one, at 0) from surfacing as a spurious
    "leading" eigenvalue in the returned spectrum.

    Parameters
    ----------
    X : np.ndarray, shape (d, m)
    phi : object with ``__call__(X)`` -> (n, m) and ``.gradient(X)`` -> (n, d, m)
    nev : int
        Number of leading eigenvalues to keep.
    a : float or np.ndarray, shape (d, d, m)
        Diffusion tensor (see evaluate_generator_rev).
    tol : float, default=0.0
        Relative SVD cutoff for the whitening transform.
    eps_ev : float, default=0.0
        Discard eigenvalues with real part >= -eps_ev (i.e. keep the
        strictly-negative/stable part of the spectrum).

    Returns
    -------
    d : np.ndarray, shape (nev,)
        Leading eigenvalues, ascending-to-descending as filter_ev leaves
        them (largest/least-negative last).
    W : np.ndarray, shape (n, nev)
        Corresponding eigenvectors in the original (unwhitened) basis.
    Wdata : np.ndarray, shape (nev, m)
        Eigenvectors evaluated at the data points.
    """
    from scipy.linalg import eigh

    PhiX = phi(X)
    PhiX_meanfree = PhiX - PhiX.mean(axis=1, keepdims=True)
    A_L = evaluate_generator_rev(X, phi, a)

    L = whitening_transform(PhiX_meanfree, tol, return_svd=False)
    R = L.T @ A_L @ L
    d, W = eigh(R)

    d, W = filter_ev(d, W, eps2=-eps_ev)
    d = d[-nev:]
    W = W[:, -nev:]

    W = L @ W
    Wdata = W.T @ PhiX

    return d, W, Wdata
