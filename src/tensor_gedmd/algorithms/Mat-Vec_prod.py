import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tensor_gedmd.reps.Tensor_Train import TT
from tensor_gedmd.algorithms.Util import truncate_tt_core

# ---------------------------
# 2) Precompute blocks ONCE
# ---------------------------
@dataclass(frozen=True)
class PreparedBlocks:
    # transposed blocks for right-multiplication (m, n, n)
    BT00: np.ndarray
    BT11: np.ndarray
    BT01: np.ndarray
    # indices of nonzero blocks
    idx00: np.ndarray
    idx11: np.ndarray
    idx01: np.ndarray
    # cached sizes
    m: int
    n: int


def prepare_blocks(blocks, m: int) -> PreparedBlocks:
    """
    Precompute stacked blocks, nonzero masks, and transposes once.
    """
    BM00 = np.stack([np.asarray(blocks[i][0, :, :, 0]) for i in range(m)], axis=0)  # (m,n,n)
    BM11 = np.stack([np.asarray(blocks[i][1, :, :, 1]) for i in range(m)], axis=0)  # (m,n,n)
    BM01 = np.stack([np.asarray(blocks[i][0, :, :, 1]) for i in range(m)], axis=0)  # (m,n,n)

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
    return PreparedBlocks(BT00=BT00, BT11=BT11, BT01=BT01, idx00=idx00, idx11=idx11, idx01=idx01, m=m, n=n)





def tt_matrix_vector_product_csr_prepared(
    M_cores: List[np.ndarray],
    x_cores: List[np.ndarray],
    prepared: Optional[PreparedBlocks] = None,
    max_rank: int = 100,
    tolerance: float = 1e-10,
    random_state: Optional[int] = None,
    sample_rows: Optional[int] = None,
    sample_cols: Optional[int] = None,
    timing: bool = False,
):
    """
    Matrix-vector product between a TT-matrix and a TT-vector
    with block-sparse structure, using precomputed blocks when needed.

    If d == 2, the middle-core branch is skipped automatically and the code
    uses only the first/last core computation path.
    """
    if len(M_cores) != len(x_cores):
        raise ValueError(
            f"M_cores and x_cores must have the same length, got "
            f"{len(M_cores)} and {len(x_cores)}."
        )

    d = len(M_cores)
    if d < 2:
        raise ValueError("Expected at least two TT cores (first and last).")

    t = {}
    t0_all = time.perf_counter()

    y_cores = []
    residual = None

    # Only needed if middle cores actually exist
    if d > 2:
        if prepared is None:
            raise ValueError("prepared blocks must be provided when d > 2.")

        m = prepared.m
        n = prepared.n
        BT00, BT11, BT01 = prepared.BT00, prepared.BT11, prepared.BT01
        idx00, idx11, idx01 = prepared.idx00, prepared.idx11, prepared.idx01

        stride = 2 * m
        m_blocks = 2 * m

    for k in range(d):
        x_core = x_cores[k]
        m1_k = x_core.shape[0]
        s2_k = x_core.shape[2]

        # ===== Middle cores =====
        if d > 2 and 1 <= k < d - 1:
            if residual is None:
                raise ValueError("Residual must be initialized before middle core.")

            r_prev, mid_dim = residual.shape
            expected_dim = m_blocks * m1_k
            if mid_dim != expected_dim:
                raise ValueError(
                    f"Expected residual shape (_, {expected_dim}), got {residual.shape}."
                )

            residual_reshaped = residual.reshape(r_prev, m_blocks, m1_k)
            R0 = residual_reshaped[:, 0::2, :]
            R1 = residual_reshaped[:, 1::2, :]

            t0 = time.perf_counter()
            H0_all = np.einsum("rmi,ins->rmns", R0, x_core, optimize=True)
            H1_all = np.einsum("rmi,ins->rmns", R1, x_core, optimize=True)
            if timing:
                t.setdefault("einsum_H01", 0.0)
                t["einsum_H01"] += time.perf_counter() - t0

            r_next = stride * s2_k
            y_updated = np.zeros((r_prev, n, r_next), dtype=x_core.dtype)
            yu = y_updated.reshape(r_prev, n, m, 2, s2_k)

            if idx00.size:
                t0 = time.perf_counter()
                H = H0_all[:, idx00, :, :]
                B = BT00[idx00]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                yu[:, :, idx00, 0, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    t.setdefault("einsum_00", 0.0)
                    t["einsum_00"] += time.perf_counter() - t0

            if idx11.size:
                t0 = time.perf_counter()
                H = H1_all[:, idx11, :, :]
                B = BT11[idx11]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                yu[:, :, idx11, 1, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    t.setdefault("einsum_11", 0.0)
                    t["einsum_11"] += time.perf_counter() - t0

            if idx01.size:
                t0 = time.perf_counter()
                H = H0_all[:, idx01, :, :]
                B = BT01[idx01]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                yu[:, :, idx01, 1, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    t.setdefault("einsum_01", 0.0)
                    t["einsum_01"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            y_updated, residual = truncate_tt_core(
                y_updated,
                max_rank=max_rank,
                tol=tolerance,
                sample_rows=sample_rows,
                sample_cols=sample_cols,
                random_state=random_state,
            )
            if timing:
                t.setdefault("truncate", 0.0)
                t["truncate"] += time.perf_counter() - t0

            y_cores.append(y_updated)
            continue

        # ===== First and last cores =====
        M_core = M_cores[k]
        r1, n_k, m_k, r2 = M_core.shape
        s1, m1_k_check, s2 = x_core.shape

        if m_k != m1_k_check:
            raise ValueError(
                f"Mismatch at core {k}: matrix has mode {m_k}, vector has mode {m1_k_check}."
            )

        t0 = time.perf_counter()
        tmp = np.tensordot(M_core, x_core, axes=([2], [1]))
        y_core = tmp.transpose(0, 3, 1, 2, 4).reshape(r1 * s1, n_k, r2 * s2)
        if timing:
            t.setdefault("tensordot_Mx", 0.0)
            t["tensordot_Mx"] += time.perf_counter() - t0

        if residual is not None:
            if residual.shape[1] != y_core.shape[0]:
                raise ValueError(
                    f"Residual mismatch: residual.shape[1]={residual.shape[1]} "
                    f"vs y_core.shape[0]={y_core.shape[0]}"
                )
            t0 = time.perf_counter()
            y_core = np.tensordot(residual, y_core, axes=(1, 0))
            if timing:
                t.setdefault("tensordot_residual", 0.0)
                t["tensordot_residual"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        y_core, residual = truncate_tt_core(
            y_core,
            max_rank=max_rank,
            tol=tolerance,
            sample_rows=sample_rows,
            sample_cols=sample_cols,
            random_state=random_state,
        )
        if timing:
            t.setdefault("truncate", 0.0)
            t["truncate"] += time.perf_counter() - t0

        y_cores.append(y_core)

    # ===== Final contraction =====
    if residual is not None:
        last_core = y_cores[-1]
        if last_core.shape[2] != residual.shape[0]:
            raise ValueError(
                f"Final residual mismatch: last_core.shape[2]={last_core.shape[2]} "
                f"vs residual.shape[0]={residual.shape[0]}"
            )
        t0 = time.perf_counter()
        y_cores[-1] = np.tensordot(last_core, residual, axes=(2, 0))
        if timing:
            t.setdefault("final_tensordot", 0.0)
            t["final_tensordot"] += time.perf_counter() - t0

    if timing:
        t["total"] = time.perf_counter() - t0_all
        return y_cores, t

    return y_cores




def make_A_mv(
    generator_op,
    blocks=None,
    *,
    max_rank=500,
    tolerance=1e-10,
    sample_rows=None,
    sample_cols=None,
):
    """
    Return A_mv(x_tt) that applies the operator.
    """
    tg_cores = generator_op.tg_cores
    d = len(tg_cores)

    if d < 2:
        raise ValueError("generator_op.tg_cores must contain at least two cores.")

    prepared = None
    if d > 2:
        if blocks is None:
            raise ValueError("blocks must be provided when the operator has middle cores.")
        m = tg_cores[1].shape[0] // 2
        prepared = prepare_blocks(blocks, m)

    def A_mv(x_tt):
        return tt_matrix_vector_product_csr_prepared(
            tg_cores,
            x_tt,
            prepared=prepared,
            max_rank=max_rank,
            tolerance=tolerance,
            sample_rows=sample_rows,
            sample_cols=sample_cols,
        )

    return A_mv
# ---------------------------
# 5) (Optional) quick timing harness for ONE matvec
# ---------------------------
def time_one_matvec(generator_op, blocks, x_tt, max_rank=500, tolerance=1e-10, sample_rows=None, sample_cols=None):
    m = generator_op.tg_cores[1].shape[0] // 2
    prepared = prepare_blocks(blocks, m)

    y, t = tt_matrix_vector_product_csr_prepared(
        generator_op.tg_cores,
        x_tt,
        prepared,
        max_rank=max_rank,
        tolerance=tolerance,
        sample_rows=sample_rows,
        sample_cols=sample_cols,
        timing=True,
    )
    return y, t


# ---------------------------
# 6) Your extract_tt_column unchanged
# ---------------------------
def extract_tt_column(U_cores, i: int):
    """
    Extract i-th column of a TT-matrix as a TT-vector.
    """
    x_tt = [G.copy() for G in U_cores]
    last = x_tt[-1]  # (r_prev, n, r)
    x_tt[-1] = last[:, :, i].reshape(last.shape[0], last.shape[1], 1)
    return x_tt
