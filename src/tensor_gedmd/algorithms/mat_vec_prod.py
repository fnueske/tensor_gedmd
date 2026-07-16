from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tensor_gedmd.algorithms.util import truncate_tt_core_cur
from tensor_gedmd.reps.tensor_train import TT
from tensor_gedmd.operations import (
    TTInnerProductMixin,
    extract_tt_column,
    tt_inner_product,
    tt_matrix_to_dense,
    tt_norm,
    tt_vector_to_dense,
)
from tensor_gedmd.operations.tt_operations import (
    TTLike,
    _as_tt_cores,
    _validate_tt_matrix_cores,
    _validate_tt_vector_cores,
)

# Re-exported here for backwards compatibility with existing imports from
# tensor_gedmd.algorithms.mat_vec_prod; the generic implementations now live
# in tensor_gedmd.operations.
__all__ = [
    "PreparedBlocks",
    "prepare_blocks",
    "tt_matrix_vector_product_general",
    "tt_matrix_vector_product_csr_prepared",
    "make_A_mv",
    "compute_A_r",
    "TTInnerProductMixin",
    "extract_tt_column",
    "tt_inner_product",
    "tt_matrix_to_dense",
    "tt_norm",
    "tt_vector_to_dense",
]


# ======================================================================================
# Data containers
# ======================================================================================

@dataclass(frozen=True)
class PreparedBlocks:
    """
    Cache for block-sparse middle-core contractions.

    The arrays ``BT00``, ``BT11``, and ``BT01`` store transposed block matrices
    so that right-multiplication can be performed efficiently inside the middle-core
    contraction routine.

    Attributes
    ----------
    BT00, BT11, BT01 : np.ndarray
        Stacked and transposed blocks with shape ``(m, n, n)``.
    idx00, idx11, idx01 : np.ndarray
        Indices of blocks that are not identically zero.
    m : int
        Number of block rows / block columns used by the special middle-core format.
    n : int
        Physical output mode size of each block.
    """

    BT00: np.ndarray
    BT11: np.ndarray
    BT01: np.ndarray
    idx00: np.ndarray
    idx11: np.ndarray
    idx01: np.ndarray
    m: int
    n: int


# ======================================================================================
# Internal helpers
# ======================================================================================
# _as_tt_cores, _validate_tt_vector_cores, and _validate_tt_matrix_cores now
# live in tensor_gedmd.operations.tt_operations and are imported above.


def _validate_matrix_vector_compatibility(
    M_cores: Sequence[np.ndarray],
    x_cores: Sequence[np.ndarray],
) -> None:
    """
    Check that TT-matrix and TT-vector are dimension-compatible.
    """
    if len(M_cores) != len(x_cores):
        raise ValueError(
            "TT-matrix and TT-vector must have the same number of cores; "
            f"got {len(M_cores)} and {len(x_cores)}."
        )

    _validate_tt_matrix_cores(M_cores, name="M_cores")
    _validate_tt_vector_cores(x_cores, name="x_cores")

    for k, (M_core, x_core) in enumerate(zip(M_cores, x_cores)):
        _, _, m_k, _ = M_core.shape
        _, x_mode, _ = x_core.shape
        if m_k != x_mode:
            raise ValueError(
                f"Physical dimension mismatch at core {k}: "
                f"matrix input mode is {m_k}, vector mode is {x_mode}."
            )


# ======================================================================================
# Block preparation
# ======================================================================================

def prepare_blocks(blocks: Sequence[np.ndarray], m: int) -> PreparedBlocks:
    """
    Precompute transposed block matrices and nonzero block indices.
    """
    if m <= 0:
        raise ValueError(f"`m` must be positive; got {m}.")

    if len(blocks) < m:
        raise ValueError(f"`blocks` must contain at least {m} entries; got {len(blocks)}.")

    try:
        BM00 = np.stack([np.asarray(blocks[i][0, :, :, 0]) for i in range(m)], axis=0)
        BM11 = np.stack([np.asarray(blocks[i][1, :, :, 1]) for i in range(m)], axis=0)
        BM01 = np.stack([np.asarray(blocks[i][0, :, :, 1]) for i in range(m)], axis=0)
    except Exception as exc:
        raise ValueError(
            "Failed to extract block slices. Expected each block to support "
            "indexing [0,:,:,0], [1,:,:,1], and [0,:,:,1]."
        ) from exc

    if BM00.ndim != 3 or BM11.ndim != 3 or BM01.ndim != 3:
        raise ValueError("Prepared block stacks must all be 3D arrays of shape (m, n, n).")

    if BM00.shape != BM11.shape or BM00.shape != BM01.shape:
        raise ValueError(
            "All extracted block stacks must have the same shape; got "
            f"{BM00.shape}, {BM11.shape}, and {BM01.shape}."
        )

    if BM00.shape[1] != BM00.shape[2]:
        raise ValueError(f"Prepared block matrices must be square; got shape {BM00.shape}.")

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


# ======================================================================================
# General TT matrix-by-vector product
# ======================================================================================

def tt_matrix_vector_product_general(
    M_cores: TTLike,
    x_cores: TTLike,
) -> List[np.ndarray]:
    """
    Compute the TT matrix-by-vector product using the standard per-core rule.
    """
    M_list = _as_tt_cores(M_cores)
    x_list = _as_tt_cores(x_cores)
    _validate_matrix_vector_compatibility(M_list, x_list)

    y_cores: List[np.ndarray] = []

    for M_core, x_core in zip(M_list, x_list):
        tmp = np.tensordot(M_core, x_core, axes=([2], [1]))
        y_core = tmp.transpose(0, 3, 1, 2, 4).reshape(
            M_core.shape[0] * x_core.shape[0],
            M_core.shape[1],
            M_core.shape[3] * x_core.shape[2],
        )
        y_cores.append(y_core)

    return y_cores


# ======================================================================================
# Specialized TT matrix-by-vector product with precomputed sparse blocks
# ======================================================================================

def tt_matrix_vector_product_csr_prepared(
    M_cores: TTLike,
    x_cores: TTLike,
    prepared: Optional[PreparedBlocks] = None,
    max_rank: int = 100,
    tolerance: float = 1e-10,
    random_state: Optional[int] = None,
    sample_rows: Optional[int] = None,
    sample_cols: Optional[int] = None,
    timing: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict[str, float]]]:
    """
    Compute a TT matrix-vector product using a specialized block-sparse middle-core routine.

    Rules:
    - if user gives 1 core -> raise clear error asking for at least 2 cores
    - if user gives 2 cores -> compute first and last directly
    - if user gives > 2 cores -> first and last direct, all others treated as middle cores
    """
    M_list = _as_tt_cores(M_cores)
    x_list = _as_tt_cores(x_cores)
    _validate_matrix_vector_compatibility(M_list, x_list)

    d = len(M_list)

    if d == 1:
        raise ValueError(
            "At least 2 TT cores are required for this routine. Got only 1 core."
        )

    timing_info: Dict[str, float] = {}
    t0_all = time.perf_counter()

    y_cores: List[np.ndarray] = []
    residual: Optional[np.ndarray] = None

    has_middle_cores = d > 2

    if has_middle_cores:
        if prepared is None:
            raise ValueError(
                "`prepared` must be provided when the operator has more than 2 cores."
            )

        m = prepared.m
        n = prepared.n
        BT00, BT11, BT01 = prepared.BT00, prepared.BT11, prepared.BT01
        idx00, idx11, idx01 = prepared.idx00, prepared.idx11, prepared.idx01

        stride = 2 * m
        m_blocks = 2 * m

    for k in range(d):
        M_core = M_list[k]
        x_core = x_list[k]

        x_rank_left = x_core.shape[0]
        x_rank_right = x_core.shape[2]

        is_first = k == 0
        is_last = k == d - 1
        is_middle = (not is_first) and (not is_last)

        # ------------------------------------------------------------------
        # Middle cores
        # ------------------------------------------------------------------
        if has_middle_cores and is_middle:
            if residual is None:
                raise ValueError(
                    f"Residual must be initialized before middle-core contraction at core {k}."
                )

            r_prev, mid_dim = residual.shape
            expected_dim = m_blocks * x_rank_left
            if mid_dim != expected_dim:
                raise ValueError(
                    f"Residual shape mismatch at core {k}: expected (_, {expected_dim}), "
                    f"got {residual.shape}."
                )

            residual_reshaped = residual.reshape(r_prev, m_blocks, x_rank_left)
            R0 = residual_reshaped[:, 0::2, :]
            R1 = residual_reshaped[:, 1::2, :]

            t0 = time.perf_counter()
            H0_all = np.einsum("rmi,ins->rmns", R0, x_core, optimize=True)
            H1_all = np.einsum("rmi,ins->rmns", R1, x_core, optimize=True)
            if timing:
                timing_info["einsum_H01"] = timing_info.get("einsum_H01", 0.0) + (
                    time.perf_counter() - t0
                )

            r_next = stride * x_rank_right
            y_updated = np.zeros(
                (r_prev, n, r_next),
                dtype=np.result_type(residual.dtype, x_core.dtype),
            )
            yu = y_updated.reshape(r_prev, n, m, 2, x_rank_right)

            if idx00.size:
                t0 = time.perf_counter()
                H = H0_all[:, idx00, :, :]
                B = BT00[idx00]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                yu[:, :, idx00, 0, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    timing_info["einsum_00"] = timing_info.get("einsum_00", 0.0) + (
                        time.perf_counter() - t0
                    )

            if idx11.size:
                t0 = time.perf_counter()
                H = H1_all[:, idx11, :, :]
                B = BT11[idx11]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                yu[:, :, idx11, 1, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    timing_info["einsum_11"] = timing_info.get("einsum_11", 0.0) + (
                        time.perf_counter() - t0
                    )

            if idx01.size:
                t0 = time.perf_counter()
                H = H0_all[:, idx01, :, :]
                B = BT01[idx01]
                Y = np.einsum("rmns,mtn->rmts", H, B, optimize=True)
                yu[:, :, idx01, 1, :] += Y.transpose(0, 2, 1, 3)
                if timing:
                    timing_info["einsum_01"] = timing_info.get("einsum_01", 0.0) + (
                        time.perf_counter() - t0
                    )

            t0 = time.perf_counter()
            y_updated, residual = truncate_tt_core_cur(
                y_updated,
                max_rank=max_rank,
                tol=tolerance,
                sample_rows=sample_rows,
                sample_cols=sample_cols,
                random_state=random_state,
            )
            if timing:
                timing_info["truncate"] = timing_info.get("truncate", 0.0) + (
                    time.perf_counter() - t0
                )

            y_cores.append(y_updated)
            continue

        # ------------------------------------------------------------------
        # First and last cores
        # Also this branch handles both cores when d == 2
        # ------------------------------------------------------------------
        r1, n_k, m_k, r2 = M_core.shape
        s1, m1_k_check, s2 = x_core.shape

        if m_k != m1_k_check:
            raise ValueError(
                f"Mode mismatch at core {k}: matrix input mode={m_k}, vector mode={m1_k_check}."
            )

        t0 = time.perf_counter()
        tmp = np.tensordot(M_core, x_core, axes=([2], [1]))
        y_core = tmp.transpose(0, 3, 1, 2, 4).reshape(r1 * s1, n_k, r2 * s2)
        if timing:
            timing_info["tensordot_Mx"] = timing_info.get("tensordot_Mx", 0.0) + (
                time.perf_counter() - t0
            )

        if residual is not None:
            if residual.shape[1] != y_core.shape[0]:
                raise ValueError(
                    f"Residual mismatch at core {k}: residual.shape[1]={residual.shape[1]} "
                    f"but y_core.shape[0]={y_core.shape[0]}."
                )
            t0 = time.perf_counter()
            y_core = np.tensordot(residual, y_core, axes=(1, 0))
            if timing:
                timing_info["tensordot_residual"] = timing_info.get(
                    "tensordot_residual", 0.0
                ) + (time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_core, residual = truncate_tt_core_cur(
            y_core,
            max_rank=max_rank,
            tol=tolerance,
            sample_rows=sample_rows,
            sample_cols=sample_cols,
            random_state=random_state,
        )
        if timing:
            timing_info["truncate"] = timing_info.get("truncate", 0.0) + (
                time.perf_counter() - t0
            )

        y_cores.append(y_core)

    # ----------------------------------------------------------------------
    # Final residual contraction
    # ----------------------------------------------------------------------
    if residual is not None:
        last_core = y_cores[-1]
        if last_core.shape[2] != residual.shape[0]:
            raise ValueError(
                "Final residual mismatch: "
                f"last_core.shape[2]={last_core.shape[2]} vs residual.shape[0]={residual.shape[0]}."
            )

        t0 = time.perf_counter()
        y_cores[-1] = np.tensordot(last_core, residual, axes=(2, 0))
        if timing:
            timing_info["final_tensordot"] = timing_info.get("final_tensordot", 0.0) + (
                time.perf_counter() - t0
            )

    if timing:
        timing_info["total"] = time.perf_counter() - t0_all
        return y_cores, timing_info

    return y_cores


# ======================================================================================
# Factory helper for repeated operator application
# ======================================================================================

def make_A_mv(
    generator_op: Any,
    blocks: Optional[Sequence[np.ndarray]] = None,
    *,
    max_rank: int = 500,
    tolerance: float = 1e-10,
    sample_rows: Optional[int] = None,
    sample_cols: Optional[int] = None,
    random_state: Optional[int] = None,
    use_general: bool = False,
) -> Callable[[TTLike], List[np.ndarray]]:
    """
    Build a callable ``A_mv(x_tt)`` that applies a TT operator to a TT vector.

    Rules:
    - 1 core  -> raise clear error
    - 2 cores -> run directly without middle-core preparation
    - >2 cores -> prepare middle-core blocks
    """
    if not hasattr(generator_op, "tg_cores"):
        raise ValueError("`generator_op` must expose a `tg_cores` attribute.")

    tg_cores = [np.asarray(core) for core in generator_op.tg_cores]
    _validate_tt_matrix_cores(tg_cores, name="generator_op.tg_cores")

    d = len(tg_cores)
    prepared: Optional[PreparedBlocks] = None

    if not use_general:
        if d == 1:
            raise ValueError(
                "At least 2 TT cores are required for this routine. Got only 1 core."
            )
        elif d > 2:
            if blocks is None:
                raise ValueError(
                    "`blocks` must be provided when using the specialized operator with more than 2 cores."
                )
            m = tg_cores[1].shape[0] // 2
            prepared = prepare_blocks(blocks, m)

    def A_mv(x_tt: TTLike) -> List[np.ndarray]:
        if use_general:
            return tt_matrix_vector_product_general(tg_cores, x_tt)

        return tt_matrix_vector_product_csr_prepared(
            tg_cores,
            x_tt,
            prepared=prepared,
            max_rank=max_rank,
            tolerance=tolerance,
            sample_rows=sample_rows,
            sample_cols=sample_cols,
            random_state=random_state,
            timing=False,
        )

    return A_mv


# ======================================================================================
# TT inner product, norm, column extraction, and dense conversions now live in
# tensor_gedmd.operations.tt_operations (imported above for convenience /
# backwards compatibility).
# ======================================================================================











# ======================================================================================
# Reduced generator matrix A_r, computed directly from a stiffness operator
# and a reduced TT basis U (from global_svd_tt), without ever materializing
# the full stiffness TT operator.
#
# Ported from a supplied reference implementation and adapted to
# tensor_gedmd.reps.stiffness_tt.TgStiffnessOperator's actual conventions:
# 0-indexed lists (self.psi, self.dpsi, self.local_dims) rather than
# 1-indexed dict keys, dpsi already normalized to plain 2D arrays, and
# `sigma_mode` / `Sigma_prepared` for the None / constant / samplewise
# diffusion tensor.
#
# Two fixes relative to the reference version:
#   - the reference version reused a single basis size `n` for every
#     dimension when reshaping cores; this breaks whenever dimensions have
#     different basis sizes (e.g. dims=[4, 5, 6] as used elsewhere in this
#     package). Here each core uses its own local dimension.
#   - the reference version assumed op.Sigma is always a real array in the
#     "constant" mode; TgStiffnessOperator's constant mode also covers the
#     Sigma=None (implicit identity) case, which is handled explicitly below.
# ======================================================================================

def _fmt(sec: float) -> str:
    return f"{sec * 1000:.1f} ms" if sec < 1 else f"{sec:.3f} s"


def _truncate_core(core: np.ndarray, cap: int) -> np.ndarray:
    """
    Hard-cap a TT core's bond dimensions to at most `cap` (simple slicing,
    not an SVD-based truncation -- use this only as a cheap/approximate cap).
    """
    rL, n, rR = core.shape
    return core[:min(rL, cap), :, :min(rR, cap)]


def _process_chunk(
    l_start: int,
    l_end: int,
    op: Any,
    U_cores: List[np.ndarray],
    p: int,
    bonds: List[int],
    r: int,
) -> Tuple[int, int, np.ndarray]:
    """
    Compute W_chunk[l, c, j] = the derivative of reduced basis function j
    (from U_cores) with respect to physical dimension c, evaluated at each
    sample l in [l_start, l_end), by carrying both the plain (undifferentiated)
    TT contraction and, for every dimension c, the contraction with dpsi
    substituted at site c, through U's cores in a single left-to-right sweep.
    """
    L = l_end - l_start

    # ---- dimension 0 ----
    U0 = U_cores[0]
    psi0 = op.psi[0][:, l_start:l_end]
    dpsi0 = op.dpsi[0][:, l_start:l_end]

    carry_base = np.einsum('isb,sl->lb', U0, psi0, optimize=True)
    d0 = np.einsum('isb,sl->lb', U0, dpsi0, optimize=True)
    del psi0, dpsi0

    carry_diff = np.empty((L, p, carry_base.shape[1]), dtype=np.float64)
    for c in range(p):
        carry_diff[:, c, :] = d0 if c == 0 else carry_base
    del d0

    # ---- middle dimensions 1, ..., p-2 ----
    for k in range(1, p - 1):
        Uk = U_cores[k]
        rL_k = bonds[k]
        rR_k = bonds[k + 1]
        n_k = op.local_dims[k]
        c_k = k

        psi_k = op.psi[k][:, l_start:l_end]
        dpsi_k = op.dpsi[k][:, l_start:l_end]

        Uk_flat = Uk.reshape(rL_k, n_k * rR_k)

        tmp_base = (carry_base @ Uk_flat).reshape(L, n_k, rR_k)
        new_base = np.einsum('lsb,sl->lb', tmp_base, psi_k, optimize=True)

        tmp_diff = (carry_diff.reshape(L * p, rL_k) @ Uk_flat).reshape(L, p, n_k, rR_k)
        new_diff = np.einsum('lcnb,nl->lcb', tmp_diff, psi_k, optimize=True)
        new_diff[:, c_k, :] = np.einsum('lsb,sl->lb', tmp_base, dpsi_k, optimize=True)

        del psi_k, dpsi_k, Uk_flat, tmp_base, tmp_diff
        carry_base = new_base
        carry_diff = new_diff

    # ---- last dimension p-1 ----
    Up = U_cores[-1]
    rL_last = bonds[p - 1]
    n_last = op.local_dims[p - 1]
    Up_flat = Up.reshape(rL_last, n_last * r)

    psi_last = op.psi[p - 1][:, l_start:l_end]
    dpsi_last = op.dpsi[p - 1][:, l_start:l_end]

    carryUp = (carry_diff.reshape(L * p, rL_last) @ Up_flat).reshape(L, p, n_last, r)

    W_chunk = np.empty((L, p, r), dtype=np.float64)
    if p > 1:
        W_chunk[:, :p - 1, :] = np.einsum(
            'nl,lcnj->lcj', psi_last, carryUp[:, :p - 1, :, :], optimize=True
        )
    W_chunk[:, p - 1, :] = np.einsum(
        'nl,lnj->lj', dpsi_last, carryUp[:, p - 1, :, :], optimize=True
    )

    del carry_diff, carry_base, carryUp, psi_last, dpsi_last
    return l_start, l_end, W_chunk


def compute_A_r(
    op: Any,
    U_cores: TTLike,
    r: int,
    chunk_size: int = 100,
    n_workers: Optional[int] = None,
    r_cap: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the reduced (Galerkin-projected) generator matrix

        A_r[i, j] = -1/(2m) * sum_l sum_{c,c'} Sigma_{c,c'}(l) *
                    d/dx_c u_i(x_l) * d/dx_{c'} u_j(x_l)

    directly from a stiffness operator ``op`` (a
    ``tensor_gedmd.reps.stiffness_tt.TgStiffnessOperator``) and a reduced TT
    basis ``U_cores`` (typically from ``global_svd_tt``), without ever
    materializing the full stiffness TT operator. Sample chunks are processed
    in parallel via a thread pool.

    Parameters
    ----------
    op : TgStiffnessOperator
        Provides ``p``, ``m``, ``psi``, ``dpsi``, ``local_dims``,
        ``sigma_mode``, and ``Sigma_prepared``.
    U_cores : TT or list of np.ndarray
        The reduced basis, with exactly ``op.p`` cores.
    r : int
        Retained rank of ``U_cores`` (the last core's right bond dimension).
    chunk_size : int, default=100
        Number of samples processed per chunk/task.
    n_workers : int, optional
        Number of worker threads. Defaults to ``min(4, cpu_count() // 2)``.
    r_cap : int, optional
        If given and smaller than ``r``, hard-caps every core's bond
        dimensions to at most ``r_cap`` before computing (cheap/approximate).

    Returns
    -------
    A_r : np.ndarray
        Symmetrized reduced generator matrix of shape (r, r).
    """
    t0 = time.perf_counter()
    p, m = op.p, op.m

    if p < 2:
        raise ValueError("compute_A_r requires at least 2 physical dimensions.")

    U_cores = _as_tt_cores(U_cores)
    if len(U_cores) != p:
        raise ValueError(f"Expected {p} U_cores, got {len(U_cores)}")

    if r_cap is not None and r_cap < r:
        U_cores = [_truncate_core(c, r_cap) for c in U_cores]
        r = U_cores[-1].shape[2]

    if n_workers is None:
        n_workers = min(4, max(1, (os.cpu_count() or 2) // 2))

    bonds = [c.shape[0] for c in U_cores] + [r]

    print(f"  [compute_A_r] p={p}  m={m}  r={r}  chunk={chunk_size}  workers={n_workers}")

    W_full = np.zeros((m, p, r), dtype=np.float64)
    chunks = [(s, min(s + chunk_size, m)) for s in range(0, m, chunk_size)]
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_chunk, s, e, op, U_cores, p, bonds, r): (s, e)
            for s, e in chunks
        }
        for fut in as_completed(futures):
            l_start, l_end, W_chunk = fut.result()
            W_full[l_start:l_end] = W_chunk
            completed += 1
            if completed % 5 == 0 or completed == len(chunks):
                print(
                    f"    chunks {completed}/{len(chunks)} ({100 * l_end / m:.0f}%)  "
                    f"elapsed {_fmt(time.perf_counter() - t0)}",
                    flush=True,
                )

    if op.sigma_mode == "constant":
        Sigma = op.Sigma_prepared
        if Sigma is None:
            # Implicit identity Sigma: Sigma @ W == W.
            SW = W_full
        else:
            SW = np.tensordot(Sigma, W_full, axes=([1], [1]))
            SW = np.moveaxis(SW, 0, 1)
    else:
        SW = np.einsum('abl,lbj->laj', op.Sigma_prepared, W_full, optimize=True)

    W2 = W_full.reshape(-1, r)
    SW2 = SW.reshape(-1, r)
    A_r = W2.T @ SW2
    A_r *= -(1.0 / (2.0 * m))
    A_r = 0.5 * (A_r + A_r.T)

    print(f"  [compute_A_r] TOTAL {_fmt(time.perf_counter() - t0)}")
    return A_r
