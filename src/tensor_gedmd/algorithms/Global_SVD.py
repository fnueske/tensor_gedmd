import numpy as np

from tensor_gedmd.reps.Tensor_Train import TT


def _truncate_rank(s: np.ndarray, rmax: int = None, tol: float = 0) -> int:
    """
    Compute truncated rank from singular values.

    Parameters
    ----------
    s : np.ndarray
        1D array of singular values.
    rmax : int, optional
        Maximum allowed rank. If None, no rank cap is applied.
    tol : float, optional
        Relative tail-energy truncation tolerance.
        If tol == 0, no tolerance-based truncation is applied.

    Returns
    -------
    int
        Truncated rank.
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
        tail_energy = np.cumsum(sq[::-1])[::-1]
        rel_tail_energy = tail_energy / tail_energy[0]

        idx = np.where(rel_tail_energy <= tol)[0]
        if idx.size > 0:
            rank_tol = max(1, int(idx[0]))
        else:
            rank_tol = n

    return max(1, min(rank_rmax, rank_tol, n))



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
    conceptually; this function returns V with shape (m, r).

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
    U_cores : list[np.ndarray]
        TT cores of the left singular tensor, i.e. updated
        [G^(0), ..., G^(p-1)], with open right boundary rank r.
    Sigma : np.ndarray
        Diagonal matrix of retained singular values, shape (r, r).
    V : np.ndarray
        Right singular vectors, shape (m, r).
    """
    if not isinstance(psi_tt, TT):
        raise TypeError("psi_tt must be an instance of TT.")

    if psi_tt.is_operator:
        raise ValueError(
            "global_svd_tt expects a tensor TT with 3D cores, not an operator TT."
        )

    cores = [core.copy() for core in psi_tt.cores]
    p = len(cores) - 1  # last core is the terminal/sample core

    if p < 1:
        raise ValueError(
            "psi_tt must contain at least one feature core and one final core."
        )

    # Validate final core G^(p): shape (m, m, 1)
    G_last = cores[p]
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

    # Sweep over feature cores except the last feature core:
    # k = 0, ..., p-2
    for k in range(p - 1):
        G = cores[k]  # shape (r_prev, n_k, r_next)

        if G.ndim != 3:
            raise ValueError(f"Core {k} must be 3D, got shape {G.shape}.")

        r_prev, n_k, r_next = G.shape

        # Unfold G^(k) into shape (r_prev * n_k, r_next)
        A = G.reshape(r_prev * n_k, r_next)

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        r_new = _truncate_rank(S, rmax=rmax, tol=tol)

        U = U[:, :r_new]
        S = S[:r_new]
        Vt = Vt[:r_new, :]

        # Replace G^(k) by reshaped U
        cores[k] = U.reshape(r_prev, n_k, r_new)

        # Push Sigma V^T into the next core
        SVt = S[:, None] * Vt
        G_next = cores[k + 1]

        if G_next.ndim != 3:
            raise ValueError(f"Core {k + 1} must be 3D, got shape {G_next.shape}.")
        # G_next has shape (r_next, n_{k+1}, r_{k+1}), and we need to contract the first mode with SVt of shape (r_new, r_next).
        # The result will have shape (r_new, n_{k+1}, r_{k+1}), which is the new shape for core k+1.
        # Also, we can optimize the contraction by directly multiplying SVt with the appropriate slices of G_next without explicitly forming the full tensor product.
        diag_vecs = np.diagonal(G_next, axis1=0, axis2=2).T  # shape (r_next, n_{k+1})
        cores[k + 1] = np.einsum("aj,jk->akj", SVt, diag_vecs, optimize=True)
    
        

    # Final SVD on the last feature core G^(p-1)
    Gp = cores[p - 1]

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

    # Replace G^(p-1) by reshaped U
    cores[p - 1] = U.reshape(r_prev, n_p, r)

    U_cores = cores[:p]
    Sigma = np.diag(S)
    V = Vt.reshape(r,m,1) # shape (r,m,1)

    return U_cores, Sigma, V


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
    U_cores : list[np.ndarray]
        TT cores of the left factor, i.e. updated [G^(0), ..., G^(p-1)].
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

    cores = [core.copy() for core in psi_tt.cores]
    p = len(cores) - 1

    if p < 1:
        raise ValueError(
            "psi_tt must contain at least one feature core and one final core."
        )

    # Validate all cores are 3D
    for k, G in enumerate(cores):
        if G.ndim != 3:
            raise ValueError(f"Core {k} must be 3D, got shape {G.shape}.")

    # Validate final core G^(p): shape (r_p, n_p, 1)
    G_last = cores[p]
    r_last, n_last, one = G_last.shape
    if one != 1:
        raise ValueError(
            f"Final core must have shape (r_p, n_p, 1), got {G_last.shape}."
        )

    # ------------------------------------------------------------------
    # Sweep over feature cores G^(0), ..., G^(p-2)
    # ------------------------------------------------------------------
    for k in range(p - 1):
        G = cores[k]
        r_prev, n_k, r_next = G.shape

        # Unfold G^(k) into shape (r_prev * n_k, r_next)
        A = G.reshape(r_prev * n_k, r_next)

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        r_new = _truncate_rank(S, rmax=rmax, tol=tol)

        U = U[:, :r_new]
        S = S[:r_new]
        Vt = Vt[:r_new, :]

        # Replace G^(k) by reshaped U
        cores[k] = U.reshape(r_prev, n_k, r_new)

        # Push Sigma V^T into the next core
        SVt = S[:, None] * Vt
        G_next = cores[k + 1]

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
                cores[k + 1] = np.einsum("aj,jk->akj", SVt, diag_vecs, optimize=True)
            else:
                cores[k + 1] = np.tensordot(SVt, G_next, axes=(1, 0))
        else:
            cores[k + 1] = np.tensordot(SVt, G_next, axes=(1, 0))
        # result shape in all cases: (r_new, n_{k+1}, r_{k+1})

    # ------------------------------------------------------------------
    # SVD of the final core G^(p)
    # ------------------------------------------------------------------
    G_last = cores[p]
    r_last, n_last, one = G_last.shape

    G_last_mat = G_last.reshape(r_last, n_last)

    U_last, S_last, Vt_last = np.linalg.svd(G_last_mat, full_matrices=False)
    r_last_new = _truncate_rank(S_last, rmax=rmax, tol=tol)

    U_last = U_last[:, :r_last_new]
    S_last = S_last[:r_last_new]
    Vt_last = Vt_last[:r_last_new, :]

    # ------------------------------------------------------------------
    # Push U_last * S_last into the last feature core G^(p-1)
    # ------------------------------------------------------------------
    G_prev = cores[p - 1]
    r_prev, n_prev, r_mid = G_prev.shape

    if r_mid != r_last:
        raise ValueError(
            f"Incompatible TT ranks between cores {p-1} and {p}: "
            f"{r_mid} != {r_last}."
        )

    US_last = U_last * S_last[None, :]
    G_prev_updated = np.tensordot(G_prev, US_last, axes=(2, 0))
    # shape: (r_{p-2}, n_{p-1}, r_last_new)

    # ------------------------------------------------------------------
    # Final SVD on the updated last feature core G^(p-1)
    # ------------------------------------------------------------------
    r_prev, n_prev, r_right = G_prev_updated.shape
    A_prev = G_prev_updated.reshape(r_prev * n_prev, r_right)

    U_prev, S_prev, Vt_prev = np.linalg.svd(A_prev, full_matrices=False)
    r = _truncate_rank(S_prev, rmax=rmax, tol=tol)

    U_prev = U_prev[:, :r]
    S_prev = S_prev[:r]
    Vt_prev = Vt_prev[:r, :]

    # Replace G^(p-1) by reshaped U
    cores[p - 1] = U_prev.reshape(r_prev, n_prev, r)

    # ------------------------------------------------------------------
    # Build outputs
    # ------------------------------------------------------------------
    U_cores = cores[:p]
    Sigma = np.diag(S_prev)

    Vt_combined = Vt_prev @ Vt_last
    V_core = Vt_combined.reshape(r, n_last, 1)

    return U_cores, Sigma, V_core






    