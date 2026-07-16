import numpy as np

from tensor_gedmd.reps.tensor_train import TT
from tensor_gedmd.algorithms.util import _truncate_rank


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


def _get_tt_cores(psi_tt, num_cores: int = None) -> list[np.ndarray]:
    """
    Materialize the TT cores of ``psi_tt`` as a plain list of NumPy arrays.

    Accepts:
    - a ``TT`` instance (or any object exposing ``.cores``/``.tt_cores``)
    - a plain list/tuple of cores
    - a callable ``psi_tt(k)`` returning the k-th core (0-indexed); in this
      case ``num_cores`` must be supplied. Cores are materialized eagerly
      (rather than fetched lazily during the sweep) so that the existing
      diagonal-coupling shortcut below can still peek at neighboring cores.
    """
    if isinstance(psi_tt, TT):
        if psi_tt.is_operator:
            raise ValueError(
                "global_svd_tt expects a tensor TT with 3D cores, not an operator TT."
            )
        return [core.copy() for core in psi_tt.cores]

    if callable(psi_tt):
        if num_cores is None:
            raise ValueError("When psi_tt is callable, num_cores must be provided.")
        return [np.asarray(psi_tt(k), dtype=float).copy() for k in range(num_cores)]

    if hasattr(psi_tt, "cores"):
        return [np.asarray(core).copy() for core in psi_tt.cores]

    if hasattr(psi_tt, "tt_cores") and psi_tt.tt_cores is not None:
        return [np.asarray(core).copy() for core in psi_tt.tt_cores]

    if isinstance(psi_tt, (list, tuple)):
        return [np.asarray(core).copy() for core in psi_tt]

    raise TypeError(
        "psi_tt must be a TT instance, an object exposing `.cores`/`.tt_cores`, "
        "a list/tuple of cores, or a callable core_getter(k) with num_cores given."
    )


def _final_core_is_identity(core: np.ndarray, atol: float = 1e-12) -> bool:
    """
    Check whether a final core of shape (r, n, 1) is, up to squeezing the
    trailing axis, the identity matrix (r == n and core[:, :, 0] == I_r).

    This holds by construction for Psi(X) tensors built via
    ``Transformed_Data_Tensor_TT`` (its final core is always ``eye(m)``).
    When it holds, SVD-ing that core and folding U*S back into the
    previous core is a mathematical no-op, so it can be skipped entirely.
    """
    r, n, one = core.shape
    if one != 1 or r != n:
        return False
    return bool(np.allclose(core[:, :, 0], np.eye(r), atol=atol))


def global_svd_tt(
    psi_tt, num_cores: int = None, rmax: int = None, tol: float = 0.0
):
    """
    Global SVD-like factorization of Psi(X) in TT format (general case).

    Unlike ``global_svd_data_tensor``, this function does not assume diagonal
    rank coupling across the sample index: it takes a general TT tensor
    (a ``TT`` instance, a plain list of cores, or a callable core getter)
    and performs the full contractions, so it works for any TT structure.

    Assumed TT structure
    --------------------
    Psi(X) = [[G^(0), ..., G^(p)]]

    where
        G^(k) has shape (r_{k-1}, n_k, r_k), for k = 0, ..., p-1
        G^(p) has shape (r_p, n_p, 1)

    Thus:
      - G^(0), ..., G^(p-1) are the feature cores,
      - G^(p) is the final core with trailing TT rank 1.

    Fast path
    ---------
    If the final core G^(p) happens to be the identity matrix (r_p == n_p,
    reshaped to (r_p, n_p, 1)) -- which is always the case for Psi(X)
    tensors built via ``Transformed_Data_Tensor_TT`` -- its SVD is a no-op
    (U = S = V = I), so it is skipped entirely and the result comes
    directly from the SVD of the (carry-folded) last feature core. This
    saves an O(m^3) SVD of an m x m identity matrix and the redundant
    fold-back step. Any other final core shape/content falls back to the
    fully general procedure below.

    General procedure (fallback)
    -----------------------------
    1. Left-orthonormalize cores G^(0), ..., G^(p-2).
    2. Compute an SVD of the final core G^(p).
    3. Push U_last * S_last into G^(p-1).
    4. Compute an SVD of the updated G^(p-1).
    5. Store the retained singular values in Sigma.
    6. Merge the right factors into a single final core.

    Parameters
    ----------
    psi_tt : TT, list of np.ndarray, or callable
        Tensor-train cores, see ``_get_tt_cores``.
    num_cores : int, optional
        Required only when ``psi_tt`` is a callable core getter.
    rmax : int, optional
        Maximum allowed truncation rank. Default is None.
    tol : float, optional
        Relative tail-energy truncation tolerance. Default is 0.

    Returns
    -------
    U_tt : TT
        TT tensor of the left factor, i.e. updated [G^(0), ..., G^(p-1)],
        with open right boundary rank r.
    Sigma : np.ndarray
        Diagonal matrix of retained singular values, shape (r, r).
    V_core : np.ndarray
        Final merged right factor, stored as a TT core of shape (r, n_p, 1).
    """
    cores = _get_tt_cores(psi_tt, num_cores=num_cores)
    total_cores = len(cores)

    if total_cores < 2:
        raise ValueError(
            "psi_tt must contain at least one feature core and one final core."
        )

    p = total_cores - 1

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

    fast_path = _final_core_is_identity(G_last)

    # ------------------------------------------------------------------
    # Sweep over feature cores G^(0), ..., G^(p-2) (unchanged either way)
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

        # If G_next has diagonal rank coupling, use the optimized contraction;
        # otherwise fall back to the fully general contraction.
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
    # Fast path: final core is the identity -> its SVD is a no-op
    # ------------------------------------------------------------------
    if fast_path:
        G_prev = cores[p - 1]
        r_prev, n_prev, r_right = G_prev.shape

        if r_right != r_last:
            raise ValueError(
                f"Incompatible TT ranks between cores {p-1} and {p}: "
                f"{r_right} != {r_last}."
            )

        A_prev = G_prev.reshape(r_prev * n_prev, r_right)

        U_prev, S_prev, Vt_prev = np.linalg.svd(A_prev, full_matrices=False)
        r = _truncate_rank(S_prev, rmax=rmax, tol=tol)

        U_prev = U_prev[:, :r]
        S_prev = S_prev[:r]
        Vt_prev = Vt_prev[:r, :]

        cores[p - 1] = U_prev.reshape(r_prev, n_prev, r)

        U_cores = cores[:p]
        U_tt = TT(U_cores, require_right_rank_one=False)
        Sigma = np.diag(S_prev)
        V_core = Vt_prev.reshape(r, n_last, 1)

        return U_tt, Sigma, V_core

    # ------------------------------------------------------------------
    # General fallback: SVD of the final core G^(p)
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

    # IMPORTANT: U_tt has open right boundary, so do not require last rank = 1
    U_tt = TT(U_cores, require_right_rank_one=False)
    Sigma = np.diag(S_prev)

    Vt_combined = Vt_prev @ Vt_last
    V_core = Vt_combined.reshape(r, n_last, 1)

    return U_tt, Sigma, V_core





