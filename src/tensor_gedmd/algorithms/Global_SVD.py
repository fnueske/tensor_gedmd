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


def global_svd_tt(psi_tt: TT, rmax: int = None, tol: float = 0):
    """
    Global SVD of Psi(X) in TT format.

    Parameters
    ----------
    psi_tt : TT
        Tensor-train object with 3D cores. The last core is the sample core.
    rmax : int, optional
        Maximum allowed TT rank. Default is None.
    tol : float, optional
        Relative tail-energy truncation tolerance. Default is 0.

    Returns
    -------
    U_cores : list[np.ndarray]
        TT cores of the left singular tensor, with open right boundary rank r.
    Sigma : np.ndarray
        Diagonal matrix of retained singular values.
    V_core : np.ndarray
        Single 3D core of shape (r, m, 1) representing V^T.
    """
    if not isinstance(psi_tt, TT):
        raise TypeError("psi_tt must be an instance of TT.")

    if psi_tt.is_operator:
        raise ValueError(
            "global_svd_tt expects a tensor TT with 3D cores, not an operator TT."
        )

    cores = [core.copy() for core in psi_tt.cores]
    p = len(cores) - 1  # last core is the sample core

    if p < 1:
        raise ValueError(
            "psi_tt must contain at least one feature core and one sample core."
        )

    # Sweep over feature cores except the last feature core
    for k in range(p - 1):
        G = cores[k]  # (r_prev, n_k, r_next)
        r_prev, n_k, r_next = G.shape

        A = G.reshape(r_prev * n_k, r_next)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        r_new = _truncate_rank(S, rmax=rmax, tol=tol)

        U = U[:, :r_new]
        S = S[:r_new]
        Vt = Vt[:r_new, :]

        cores[k] = U.reshape(r_prev, n_k, r_new)

        SVt = np.diag(S) @ Vt
        cores[k + 1] = np.tensordot(SVt, cores[k + 1], axes=(1, 0))

    # Final feature core + sample core
    Gp = cores[p - 1]   # (r_{p-2}, n_p, r_{p-1})
    Gs = cores[p]       # (r_{p-1}, m, 1)

    T = np.tensordot(Gp, Gs, axes=(2, 0))   # (r_{p-2}, n_p, m, 1)
    r_prev, n_p, m, _ = T.shape

    A = T.reshape(r_prev * n_p, m)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    r = _truncate_rank(S, rmax=rmax, tol=tol)

    U = U[:, :r]
    S = S[:r]
    Vt = Vt[:r, :]

    cores[p - 1] = U.reshape(r_prev, n_p, r)

    U_cores = cores[:p]
    Sigma = np.diag(S)
    V_core = Vt.reshape(r, m, 1)

    return U_cores, Sigma, V_core







