import numpy as np

from tensor_gedmd.reps.Tensor_Train import TT
from tensor_gedmd.algorithms.Util import _truncate_rank


def global_svd_data_tensor(
    basis_evals: list[np.ndarray], rmax: int = None, tol: float = 0.0
):
    """
    Global SVD of Psi(X) from basis set evaluations.

    basis_evals[k] has shape (m, n_k), for k = 0, ..., p-1
    """
    if not isinstance(basis_evals, list) or len(basis_evals) == 0:
        raise TypeError("basis_evals must be a non-empty list of ndarrays.")

    for k, B in enumerate(basis_evals):
        if not isinstance(B, np.ndarray):
            raise TypeError(f"basis_evals[{k}] must be a numpy.ndarray.")
        if B.ndim != 2:
            raise ValueError(
                f"basis_evals[{k}] must be 2D with shape (m, n_{k}), got {B.shape}."
            )

    m = basis_evals[0].shape[0]
    for k, B in enumerate(basis_evals):
        if B.shape[0] != m:
            raise ValueError(
                f"All basis evaluations must have the same number of samples m. "
                f"basis_evals[0].shape[0] = {m}, but basis_evals[{k}].shape[0] = {B.shape[0]}."
            )

    p = len(basis_evals)
    U_cores = []

    # Start with the first core: shape (1, n_0, m)
    G_curr = basis_evals[0].T[None, :, :]

    # Sweep over feature cores except the last feature core: k = 0, ..., p-2
    for k in range(p - 1):
        r_prev, n_k, r_next = G_curr.shape

        # Unfold G^(k) into shape (r_prev * n_k, r_next)
        A = G_curr.reshape(r_prev * n_k, r_next)

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        r_new = _truncate_rank(S, rmax=rmax, tol=tol)

        U = U[:, :r_new]
        S = S[:r_new]
        Vt = Vt[:r_new, :]

        # Store updated left core
        U_cores.append(U.reshape(r_prev, n_k, r_new))

        # Push Sigma V^T into the next basis evaluation
        SVt = S[:, None] * Vt
        B_next = basis_evals[k + 1]

        if B_next.ndim != 2:
            raise ValueError(
                f"basis_evals[{k + 1}] must be 2D, got shape {B_next.shape}."
            )

        if B_next.shape[0] != r_next:
            raise ValueError(
                f"Incompatible sample dimension between contraction factor and "
                f"basis_evals[{k + 1}]: {r_next} != {B_next.shape[0]}."
            )

        # Equivalent to contracting with the diagonal-coupled core, but without building it
        G_curr = np.einsum("aj,jk->akj", SVt, B_next, optimize=True)

    # Final SVD on the last feature core G^(p-1)
    Gp = G_curr

    if Gp.ndim != 3:
        raise ValueError(f"Last feature core must be 3D, got shape {Gp.shape}.")

    r_prev, n_p, r_right = Gp.shape
    if r_right != m:
        raise ValueError(
            f"The last feature core must have right rank m={m}, got {r_right}."
        )

    # Unfold G^(p-1) into shape (r_prev * n_p, m)
    A = Gp.reshape(r_prev * n_p, m)

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = _truncate_rank(S, rmax=rmax, tol=tol)

    U = U[:, :r]
    S = S[:r]
    Vt = Vt[:r, :]

    U_cores.append(U.reshape(r_prev, n_p, r))

    # IMPORTANT: U_tt has open right boundary, so do not require last rank = 1
    U_tt = TT(U_cores, require_right_rank_one=False)
    Sigma = np.diag(S)
    V = Vt.reshape(r, m, 1)

    return U_tt, Sigma, V


def global_svd_tt(
    basis_evals: list[np.ndarray], rmax: int = None, tol: float = 0.0
):
    """
    Global SVD-like factorization of Psi(X) in TT format (general case).

    The function receives a list of basis evaluations:
        basis_evals[k] has shape (m, n_k)

    Interpretation:
      - basis_evals[0], ..., basis_evals[p-1] are used as feature cores
        with diagonal rank coupling across the sample index,
      - basis_evals[p] is used as the final core of shape (m, n_p, 1).
    """
    if not isinstance(basis_evals, list) or len(basis_evals) < 2:
        raise TypeError(
            "basis_evals must be a non-empty list of ndarrays with at least two entries."
        )

    for k, B in enumerate(basis_evals):
        if not isinstance(B, np.ndarray):
            raise TypeError(f"basis_evals[{k}] must be a numpy.ndarray.")
        if B.ndim != 2:
            raise ValueError(
                f"basis_evals[{k}] must be 2D with shape (m, n_{k}), got {B.shape}."
            )

    m = basis_evals[0].shape[0]
    for k, B in enumerate(basis_evals):
        if B.shape[0] != m:
            raise ValueError(
                f"All basis evaluations must have the same number of samples m. "
                f"basis_evals[0].shape[0] = {m}, but basis_evals[{k}].shape[0] = {B.shape[0]}."
            )

    # Number of feature cores; the last entry is treated as the final core
    p = len(basis_evals) - 1
    if p < 1:
        raise ValueError(
            "basis_evals must contain at least one feature core and one final core."
        )

    # Final core G^(p): shape (r_p, n_p, 1) with r_p = m
    G_last = basis_evals[p][:, :, None]
    r_last, n_last, one = G_last.shape

    if one != 1:
        raise ValueError(
            f"Final core must have shape (r_p, n_p, 1), got {G_last.shape}."
        )

    U_cores = []

    # Start with first feature core
    G_curr = basis_evals[0].T[None, :, :]
    if G_curr.ndim != 3:
        raise ValueError(f"Core 0 must be 3D, got shape {G_curr.shape}.")

    # Sweep over feature cores G^(0), ..., G^(p-2)
    for k in range(p - 1):
        r_prev, n_k, r_next = G_curr.shape

        # Unfold G^(k) into shape (r_prev * n_k, r_next)
        A = G_curr.reshape(r_prev * n_k, r_next)

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        r_new = _truncate_rank(S, rmax=rmax, tol=tol)

        U = U[:, :r_new]
        S = S[:r_new]
        Vt = Vt[:r_new, :]

        # Store updated left core
        U_cores.append(U.reshape(r_prev, n_k, r_new))

        # Push Sigma V^T into the next feature core
        SVt = S[:, None] * Vt
        B_next = basis_evals[k + 1]

        if B_next.ndim != 2:
            raise ValueError(
                f"basis_evals[{k + 1}] must be 2D, got shape {B_next.shape}."
            )

        if B_next.shape[0] != r_next:
            raise ValueError(
                f"Incompatible sample dimension between contraction factor and "
                f"basis_evals[{k + 1}]: {r_next} != {B_next.shape[0]}."
            )

        # Equivalent to contracting with the diagonal-coupled next core
        G_curr = np.einsum("aj,jk->akj", SVt, B_next, optimize=True)

    # SVD of the final core G^(p)
    G_last_mat = G_last.reshape(r_last, n_last)

    U_last, S_last, Vt_last = np.linalg.svd(G_last_mat, full_matrices=False)
    r_last_new = _truncate_rank(S_last, rmax=rmax, tol=tol)

    U_last = U_last[:, :r_last_new]
    S_last = S_last[:r_last_new]
    Vt_last = Vt_last[:r_last_new, :]

    # Push U_last * S_last into the last feature core G^(p-1)
    G_prev = G_curr
    r_prev, n_prev, r_mid = G_prev.shape

    if r_mid != r_last:
        raise ValueError(
            f"Incompatible TT ranks between cores {p-1} and {p}: "
            f"{r_mid} != {r_last}."
        )

    US_last = U_last * S_last[None, :]
    G_prev_updated = np.tensordot(G_prev, US_last, axes=(2, 0))

    # Final SVD on the updated last feature core G^(p-1)
    r_prev, n_prev, r_right = G_prev_updated.shape
    A_prev = G_prev_updated.reshape(r_prev * n_prev, r_right)

    U_prev, S_prev, Vt_prev = np.linalg.svd(A_prev, full_matrices=False)
    r = _truncate_rank(S_prev, rmax=rmax, tol=tol)

    U_prev = U_prev[:, :r]
    S_prev = S_prev[:r]
    Vt_prev = Vt_prev[:r, :]

    U_cores.append(U_prev.reshape(r_prev, n_prev, r))

    # IMPORTANT: U_tt has open right boundary, so do not require last rank = 1
    U_tt = TT(U_cores, require_right_rank_one=False)
    Sigma = np.diag(S_prev)

    Vt_combined = Vt_prev @ Vt_last
    V_core = Vt_combined.reshape(r, n_last, 1)

    return U_tt, Sigma, V_core





