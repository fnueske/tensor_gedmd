import numpy as np

from tensor_gedmd.reps.Tensor_Train import TT
from tensor_gedmd.algorithms.Util import _truncate_rank

def global_svd_tt(psi_tt: TT, rmax: int = None, tol: float = 0.0):
    """
    Global SVD of Psi(X) in TT format.

    Assumed TT structure
    --------------------
    Psi(X) = [[G^(0), ..., G^(p)]]

    where
        G^(k) has shape (r_{k-1}, n_k, r_k), for k = 0, ..., p-1
        G^(p) has shape (m, m, 1)

    The final core G^(p) is the terminal/sample core and is replaced by V^T
    conceptually; this function returns V stored as a TT-like core of shape
    (r, m, 1).

    Parameters
    ----------
    psi_tt : TT
        Tensor-train object with 3D cores.
    rmax : int, optional
        Maximum allowed TT rank. Default is None.
    tol : float, optional
        Relative tail-energy truncation tolerance. Default is 0.

    Returns
    -------
    U_tt : TT
        TT object containing the left singular tensor cores
        [G^(0), ..., G^(p-1)] with open right boundary.
    Sigma : np.ndarray
        Diagonal matrix of retained singular values, shape (r, r).
    V : np.ndarray
        Right singular vectors stored as a core of shape (r, m, 1).
    """
    if not isinstance(psi_tt, TT):
        raise TypeError("psi_tt must be an instance of TT.")

    if psi_tt.is_operator:
        raise ValueError(
            "global_svd_tt expects a tensor TT with 3D cores, not an operator TT."
        )

    p = len(psi_tt) - 1  # last core is the terminal/sample core

    if p < 1:
        raise ValueError(
            "psi_tt must contain at least one feature core and one final core."
        )

    # Validate final core G^(p): shape (m, m, 1)
    G_last = psi_tt.get_core(p)
    if G_last.ndim != 3:
        raise ValueError(
            f"Final core must be 3D with shape (m, m, 1), got {G_last.shape}."
        )

    m1, m2, one = G_last.shape
    if m1 != m2 or one != 1:
        raise ValueError(
            f"Expected final core shape (m, m, 1), got {G_last.shape}."
        )

    m = m1
    U_cores = []

    # Start with the first core
    G_curr = psi_tt.get_core(0).copy()
    if G_curr.ndim != 3:
        raise ValueError(f"Core 0 must be 3D, got shape {G_curr.shape}.")

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

        # Push Sigma V^T into the next core
        SVt = S[:, None] * Vt
        G_next = psi_tt.get_core(k + 1).copy()

        if G_next.ndim != 3:
            raise ValueError(f"Core {k + 1} must be 3D, got shape {G_next.shape}.")

        if G_next.shape[0] != r_next:
            raise ValueError(
                f"Incompatible TT ranks between cores {k} and {k+1}: "
                f"{r_next} != {G_next.shape[0]}."
            )

        # Optimized contraction if G_next has diagonal rank coupling
        if G_next.shape[0] == G_next.shape[2]:
            idx = np.arange(G_next.shape[0])
            mask = np.ones_like(G_next, dtype=bool)
            mask[idx, :, idx] = False

            if np.allclose(G_next[mask], 0.0, atol=1e-14):
                diag_vecs = np.diagonal(G_next, axis1=0, axis2=2).T
                G_curr = np.einsum("aj,jk->akj", SVt, diag_vecs, optimize=True)
            else:
                G_curr = np.tensordot(SVt, G_next, axes=(1, 0))
        else:
            G_curr = np.tensordot(SVt, G_next, axes=(1, 0))

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


def global_svd_tt_general(psi_tt: TT, rmax: int = None, tol: float = 0.0):
    """
    Global SVD-like factorization of Psi(X) in TT format (general case).

    Assumed TT structure
    --------------------
    Psi(X) = [[G^(0), ..., G^(p)]]

    where
        G^(k) has shape (r_{k-1}, n_k, r_k), for k = 0, ..., p-1
        G^(p) has shape (r_p, n_p, 1)

    Thus:
      - G^(0), ..., G^(p-1) are the feature cores,
      - G^(p) is the final core with trailing TT rank 1.

    Procedure
    ---------
    1. Left-orthonormalize cores G^(0), ..., G^(p-2).
    2. Compute an SVD of the final core G^(p).
    3. Push U_last * S_last into G^(p-1).
    4. Compute an SVD of the updated G^(p-1).
    5. Store the retained singular values in Sigma.
    6. Merge the right factors into a single final core.

    Parameters
    ----------
    psi_tt : TT
        Tensor-train object with 3D cores.
    rmax : int, optional
        Maximum allowed truncation rank. Default is None.
    tol : float, optional
        Relative tail-energy truncation tolerance. Default is 0.

    Returns
    -------
    U_tt : TT
        TT object containing the left factor cores [G^(0), ..., G^(p-1)]
        with open right boundary.
    Sigma : np.ndarray
        Diagonal matrix of retained singular values, shape (r, r).
    V_core : np.ndarray
        Final merged right factor, stored as a TT core of shape (r, n_p, 1).
    """
    if not isinstance(psi_tt, TT):
        raise TypeError("psi_tt must be an instance of TT.")

    if psi_tt.is_operator:
        raise ValueError(
            "global_svd_tt_general expects a tensor TT with 3D cores, not an operator TT."
        )

    p = len(psi_tt) - 1

    if p < 1:
        raise ValueError(
            "psi_tt must contain at least one feature core and one final core."
        )

    # Validate final core G^(p): shape (r_p, n_p, 1)
    G_last = psi_tt.get_core(p)
    if G_last.ndim != 3:
        raise ValueError(
            f"Final core must be 3D with shape (r_p, n_p, 1), got {G_last.shape}."
        )

    r_last, n_last, one = G_last.shape
    if one != 1:
        raise ValueError(
            f"Final core must have shape (r_p, n_p, 1), got {G_last.shape}."
        )

    U_cores = []

    # Start with first core
    G_curr = psi_tt.get_core(0).copy()
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

        # Push Sigma V^T into the next core
        SVt = S[:, None] * Vt
        G_next = psi_tt.get_core(k + 1).copy()

        if G_next.ndim != 3:
            raise ValueError(f"Core {k + 1} must be 3D, got shape {G_next.shape}.")

        if G_next.shape[0] != r_next:
            raise ValueError(
                f"Incompatible TT ranks between cores {k} and {k+1}: "
                f"{r_next} != {G_next.shape[0]}."
            )

        # If G_next has diagonal rank coupling, use the optimized contraction
        if G_next.shape[0] == G_next.shape[2]:
            idx = np.arange(G_next.shape[0])
            mask = np.ones_like(G_next, dtype=bool)
            mask[idx, :, idx] = False

            if np.allclose(G_next[mask], 0.0, atol=1e-14):
                diag_vecs = np.diagonal(G_next, axis1=0, axis2=2).T
                G_curr = np.einsum("aj,jk->akj", SVt, diag_vecs, optimize=True)
            else:
                G_curr = np.tensordot(SVt, G_next, axes=(1, 0))
        else:
            G_curr = np.tensordot(SVt, G_next, axes=(1, 0))

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







