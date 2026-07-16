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
