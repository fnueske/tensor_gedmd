import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# TT core truncation
# ============================================================================

def truncate_tt_core(
    Y: np.ndarray,
    rmax: int = 100,
    tol: float = 1e-10,
    sample_rows: Optional[int] = None,
    sample_cols: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncate a TT core by a CUR-based low-rank approximation.

    Parameters
    ----------
    Y : np.ndarray
        TT core of shape (r_{k-1}, n_k, r_k).
    rmax : int, optional
        Maximum retained rank.
    tol : float, optional
        Tolerance used in the pseudoinverse during the CUR step.
    sample_rows : int, optional
        Number of sampled rows of the unfolding. If None, a default value
        based on `rmax` is used.
    sample_cols : int, optional
        Number of sampled columns of the unfolding. If None, a default value
        based on `rmax` is used.
    random_state : int, optional
        Random seed for row/column sampling.

    Returns
    -------
    Y_trunc : np.ndarray
        Truncated TT core of shape (r_{k-1}, n_k, r_k_trunc).
    R : np.ndarray
        Residual factor of shape (r_k_trunc, r_k), to be absorbed into
        the next TT core.
    """
    rng = np.random.default_rng(random_state)

    rkm1, nk, rk = Y.shape
    A = Y.reshape(rkm1 * nk, rk)

    if sample_cols is None:
        sample_cols = min(rmax + 50, rk)
    if sample_rows is None:
        sample_rows = min(rmax + 50, rkm1 * nk)

    col_idx = rng.choice(rk, size=sample_cols, replace=False)
    row_idx = rng.choice(rkm1 * nk, size=sample_rows, replace=False)

    C = A[:, col_idx]
    R_full = A[row_idx, :]
    W = A[np.ix_(row_idx, col_idx)]

    U_cur = np.linalg.pinv(W, rcond=tol)
    CU = C @ U_cur

    if CU.shape[0] >= CU.shape[1]:
        Uc, Sc, Vhc = np.linalg.svd(CU, full_matrices=False)
        r_new = min(rmax, Sc.shape[0])

        Uc = Uc[:, :r_new]
        Sc = Sc[:r_new]
        Vhc = Vhc[:r_new, :]

        R = (Sc[:, None] * Vhc) @ R_full
        Y_trunc = Uc.reshape(rkm1, nk, r_new)
    else:
        Ur, Sr, Vhr = np.linalg.svd(R_full, full_matrices=False)
        r_new = min(rmax, Sr.shape[0])

        Ur = Ur[:, :r_new]
        Sr = Sr[:r_new]
        Vhr = Vhr[:r_new, :]

        Z = C @ (Ur * Sr[None, :])
        Y_trunc = Z.reshape(rkm1, nk, r_new)
        R = Vhr

    return Y_trunc, R


# ============================================================================
# Prepared block data
# ============================================================================

@dataclass(frozen=True)
class PreparedBlocks:
    """
    Container for precomputed block data used in the TT matvec.

    Attributes
    ----------
    BT00 : np.ndarray
        Transposed (0,0) blocks, shape (m, n, n).
    BT11 : np.ndarray
        Transposed (1,1) blocks, shape (m, n, n).
    BT01 : np.ndarray
        Transposed (0,1) blocks, shape (m, n, n).
    idx00 : np.ndarray
        Indices of nonzero (0,0) blocks.
    idx11 : np.ndarray
        Indices of nonzero (1,1) blocks.
    idx01 : np.ndarray
        Indices of nonzero (0,1) blocks.
    m : int
        Number of block positions.
    n : int
        Physical mode size.
    """
    BT00: np.ndarray
    BT11: np.ndarray
    BT01: np.ndarray
    idx00: np.ndarray
    idx11: np.ndarray
    idx01: np.ndarray
    m: int
    n: int


def prepare_blocks(blocks, m: int) -> PreparedBlocks:
    """
    Precompute block stacks, transposes, and nonzero masks.

    Parameters
    ----------
    blocks : sequence
        Block data structure such that:
            blocks[i][0, :, :, 0] is the (0,0) block,
            blocks[i][1, :, :, 1] is the (1,1) block,
            blocks[i][0, :, :, 1] is the (0,1) block.
    m : int
        Number of block positions.

    Returns
    -------
    PreparedBlocks
        Cached block data for repeated TT matrix-vector products.
    """
    BM00 = np.stack([np.asarray(blocks[i][0, :, :, 0]) for i in range(m)], axis=0)
    BM11 = np.stack([np.asarray(blocks[i][1, :, :, 1]) for i in range(m)], axis=0)
    BM01 = np.stack([np.asarray(blocks[i][0, :, :, 1]) for i in range(m)], axis=0)

    mask00 = ~(BM00 == 0).all(axis=(1, 2))
    mask11 = ~(BM11 == 0).all(axis=(1, 2))
    mask01 = ~(BM01 == 0).all(axis=(1, 2))

    idx00 = np.flatnonzero(mask00)
    idx11 = np.flatnonzero(mask11)
    idx01 = np.flatnonzero(mask01)

    BT00 = BM00.transpose(0, 2, 1).copy()
    BT11 = BM11.transpose(0, 2, 1).copy()
    BT01 = BM01.transpose(0, 2, 1).copy()

    n = BM00.shape[1]

    return PreparedBlocks(
        BT00=BT00,
        BT11=BT11,
        BT01=BT01,
        idx00=idx00,
        idx11=idx11,
        idx01=idx01,
        m=m,
        n=n,
    )


# ============================================================================
# TT matrix-vector product with prepared blocks
# ============================================================================

def tt_matrix_vector_product_csr_prepared(
    A_cores: List[np.ndarray],
    x_cores: List[np.ndarray],
    prepared: PreparedBlocks,
    rmax: int = 100,
    tol: float = 1e-10,
    sample_rows: Optional[int] = None,
    sample_cols: Optional[int] = None,
    random_state: Optional[int] = None,
    timing: bool = False,
):
    """
    Compute a TT matrix-vector product using precomputed block data.

    Parameters
    ----------
    A_cores : list[np.ndarray]
        TT-matrix cores.
    x_cores : list[np.ndarray]
        TT-vector cores.
    prepared : PreparedBlocks
        Precomputed block data.
    rmax : int, optional
        Maximum retained TT rank in truncation.
    tol : float, optional
        Truncation tolerance passed to `truncate_tt_core`.
    sample_rows : int, optional
        Number of sampled rows in CUR truncation.
    sample_cols : int, optional
        Number of sampled columns in CUR truncation.
    random_state : int, optional
        Random seed for CUR truncation.
    timing : bool, optional
        If True, also return a timing dictionary.

    Returns
    -------
    y_cores : list[np.ndarray]
        TT cores of the product.
    timings : dict, optional
        Timing breakdown if `timing=True`.
    """
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    m = prepared.m
    n = prepared.n

    BT00 = prepared.BT00
    BT11 = prepared.BT11
    BT01 = prepared.BT01

    idx00 = prepared.idx00
    idx11 = prepared.idx11
    idx01 = prepared.idx01

    d = len(A_cores)
    y_cores: List[np.ndarray] = []
    R: Optional[np.ndarray] = None

    stride = 2 * m
    m_blocks = 2 * m

    for k in range(d):
        xk = x_cores[k]

        # ------------------------------------------------------------------
        # Middle cores
        # ------------------------------------------------------------------
        if 1 <= k < d - 1:
            if R is None:
                raise ValueError("Residual factor must be initialized before middle cores.")

            mkm1 = xk.shape[0]
            sk = xk.shape[2]

            r_prev, dim_mid = R.shape
            expected_dim = m_blocks * mkm1

            if dim_mid != expected_dim:
                raise ValueError(
                    f"Expected residual shape (_, {expected_dim}), got {R.shape}."
                )

            R_view = R.reshape(r_prev, m_blocks, mkm1)
            R0 = R_view[:, 0::2, :]
            R1 = R_view[:, 1::2, :]

            t0 = time.perf_counter()
            H0_all = np.einsum("rmi,ins->rmns", R0, xk, optimize=True)
            H1_all = np.einsum("rmi,ins->rmns", R1, xk, optimize=True)
            if timing:
                timings["einsum_H01"] = timings.get("einsum_H01", 0.0) + (time.perf_counter() - t0)

            r_next = stride * sk
            Yk = np.zeros((r_prev, n, r_next), dtype=xk.dtype)
            Yk_view = Yk.reshape(r_prev, n, m, 2, sk)

            if idx00.size:
                t0 = time.perf_counter()
                H = H0_all[:, idx00, :, :]
                B = BT00[idx00]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                Yk_view[:, :, idx00, 0, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    timings["einsum_00"] = timings.get("einsum_00", 0.0) + (time.perf_counter() - t0)

            if idx11.size:
                t0 = time.perf_counter()
                H = H1_all[:, idx11, :, :]
                B = BT11[idx11]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                Yk_view[:, :, idx11, 1, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    timings["einsum_11"] = timings.get("einsum_11", 0.0) + (time.perf_counter() - t0)

            if idx01.size:
                t0 = time.perf_counter()
                H = H0_all[:, idx01, :, :]
                B = BT01[idx01]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                Yk_view[:, :, idx01, 1, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    timings["einsum_01"] = timings.get("einsum_01", 0.0) + (time.perf_counter() - t0)

            t0 = time.perf_counter()
            Yk, R = truncate_tt_core(
                Yk,
                rmax=rmax,
                tol=tol,
                sample_rows=sample_rows,
                sample_cols=sample_cols,
                random_state=random_state,
            )
            if timing:
                timings["truncate"] = timings.get("truncate", 0.0) + (time.perf_counter() - t0)

            y_cores.append(Yk)
            continue

        # ------------------------------------------------------------------
        # First and last cores
        # ------------------------------------------------------------------
        Ak = A_cores[k]
        rkm1, nk, mk, rk = Ak.shape
        skm1, mk_x, sk = xk.shape

        if mk != mk_x:
            raise ValueError(
                f"Mismatch at core {k}: matrix has mode {mk}, vector has mode {mk_x}."
            )

        t0 = time.perf_counter()
        tmp = np.tensordot(Ak, xk, axes=([2], [1]))
        Yk = tmp.transpose(0, 3, 1, 2, 4).reshape(rkm1 * skm1, nk, rk * sk)
        if timing:
            timings["tensordot_Ax"] = timings.get("tensordot_Ax", 0.0) + (time.perf_counter() - t0)

        if R is not None:
            if R.shape[1] != Yk.shape[0]:
                raise ValueError(
                    f"Residual mismatch: R.shape[1]={R.shape[1]} vs Yk.shape[0]={Yk.shape[0]}."
                )

            t0 = time.perf_counter()
            Yk = np.tensordot(R, Yk, axes=(1, 0))
            if timing:
                timings["tensordot_residual"] = timings.get("tensordot_residual", 0.0) + (
                    time.perf_counter() - t0
                )

        t0 = time.perf_counter()
        Yk, R = truncate_tt_core(
            Yk,
            rmax=rmax,
            tol=tol,
            sample_rows=sample_rows,
            sample_cols=sample_cols,
            random_state=random_state,
        )
        if timing:
            timings["truncate"] = timings.get("truncate", 0.0) + (time.perf_counter() - t0)

        y_cores.append(Yk)

    # ----------------------------------------------------------------------
    # Final residual absorption
    # ----------------------------------------------------------------------
    if R is not None:
        Y_last = y_cores[-1]

        if Y_last.shape[2] != R.shape[0]:
            raise ValueError(
                f"Final residual mismatch: last core right rank {Y_last.shape[2]} "
                f"does not match residual left rank {R.shape[0]}."
            )

        t0 = time.perf_counter()
        y_cores[-1] = np.tensordot(Y_last, R, axes=(2, 0))
        if timing:
            timings["final_tensordot"] = timings.get("final_tensordot", 0.0) + (
                time.perf_counter() - t0
            )

    if timing:
        timings["total"] = time.perf_counter() - t_total
        return y_cores, timings

    return y_cores


# ============================================================================
# Convenience wrapper
# ============================================================================

def make_A_mv(
    generator_op,
    blocks,
    *,
    rmax: int = 500,
    tol: float = 1e-10,
    sample_rows: Optional[int] = None,
    sample_cols: Optional[int] = None,
) -> Callable[[List[np.ndarray]], List[np.ndarray]]:
    """
    Build a reusable TT matvec closure with prepared blocks.

    Parameters
    ----------
    generator_op : object
        Operator object with TT-matrix cores stored in `generator_op.tg_cores`.
    blocks : sequence
        Block data used by `prepare_blocks`.
    rmax : int, optional
        Maximum retained TT rank.
    tol : float, optional
        Truncation tolerance.
    sample_rows : int, optional
        Number of sampled rows in CUR truncation.
    sample_cols : int, optional
        Number of sampled columns in CUR truncation.

    Returns
    -------
    A_mv : callable
        Function that maps TT-vector cores to TT-vector cores.
    """
    m = generator_op.tg_cores[1].shape[0] // 2
    prepared = prepare_blocks(blocks, m)

    def A_mv(x_tt: List[np.ndarray]) -> List[np.ndarray]:
        return tt_matrix_vector_product_csr_prepared(
            generator_op.tg_cores,
            x_tt,
            prepared,
            rmax=rmax,
            tol=tol,
            sample_rows=sample_rows,
            sample_cols=sample_cols,
        )

    return A_mv


# ============================================================================
# Timing helper
# ============================================================================

def time_one_matvec(
    generator_op,
    blocks,
    x_tt,
    rmax: int = 500,
    tol: float = 1e-10,
    sample_rows: Optional[int] = None,
    sample_cols: Optional[int] = None,
):
    """
    Time a single TT matrix-vector product.

    Parameters
    ----------
    generator_op : object
        Operator object with TT-matrix cores stored in `generator_op.tg_cores`.
    blocks : sequence
        Block data used by `prepare_blocks`.
    x_tt : list[np.ndarray]
        Input TT-vector cores.
    rmax : int, optional
        Maximum retained TT rank.
    tol : float, optional
        Truncation tolerance.
    sample_rows : int, optional
        Number of sampled rows in CUR truncation.
    sample_cols : int, optional
        Number of sampled columns in CUR truncation.

    Returns
    -------
    y_tt : list[np.ndarray]
        Output TT-vector cores.
    timings : dict
        Timing breakdown.
    """
    m = generator_op.tg_cores[1].shape[0] // 2
    prepared = prepare_blocks(blocks, m)

    y_tt, timings = tt_matrix_vector_product_csr_prepared(
        generator_op.tg_cores,
        x_tt,
        prepared,
        rmax=rmax,
        tol=tol,
        sample_rows=sample_rows,
        sample_cols=sample_cols,
        timing=True,
    )

    return y_tt, timings


# ============================================================================
# TT column extraction
# ============================================================================

def extract_tt_column(U_cores: List[np.ndarray], i: int) -> List[np.ndarray]:
    """
    Extract the i-th column of a TT-matrix as a TT-vector.

    Parameters
    ----------
    U_cores : list[np.ndarray]
        TT-matrix represented by 3D cores, with the last core containing
        the column index in its last mode.
    i : int
        Column index to extract.

    Returns
    -------
    x_tt : list[np.ndarray]
        TT-vector cores corresponding to the i-th column.
    """
    x_tt = [G.copy() for G in U_cores]
    G_last = x_tt[-1]
    x_tt[-1] = G_last[:, :, i].reshape(G_last.shape[0], G_last.shape[1], 1)
    return x_tt