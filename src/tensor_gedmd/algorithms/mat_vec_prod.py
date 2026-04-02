from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tensor_gedmd.algorithms.util import truncate_tt_core_cur
from tensor_gedmd.reps.tensor_train import TT


# ======================================================================================
# Types
# ======================================================================================

TTLike = Union[TT, Sequence[np.ndarray]]


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

def _as_tt_cores(tt_obj: TTLike) -> List[np.ndarray]:
    """
    Return TT cores as a plain list of NumPy arrays.
    """
    if isinstance(tt_obj, TT):
        if hasattr(tt_obj, "tt_cores"):
            return [np.asarray(core) for core in tt_obj.tt_cores]
        if hasattr(tt_obj, "cores"):
            return [np.asarray(core) for core in tt_obj.cores]

    if hasattr(tt_obj, "tt_cores"):
        return [np.asarray(core) for core in tt_obj.tt_cores]

    if hasattr(tt_obj, "cores"):
        return [np.asarray(core) for core in tt_obj.cores]

    if isinstance(tt_obj, (list, tuple)):
        return [np.asarray(core) for core in tt_obj]

    raise TypeError(
        "Expected a TT object, an object with `.tt_cores`/`.cores`, "
        f"or a sequence of NumPy arrays; got {type(tt_obj)!r}."
    )


def _validate_tt_vector_cores(x_cores: Sequence[np.ndarray], *, name: str = "x_cores") -> None:
    """
    Validate TT-vector cores with shape ``(r_{k-1}, n_k, r_k)``.
    """
    if len(x_cores) == 0:
        raise ValueError(f"{name} must contain at least one core.")

    prev_rank_right = None
    for k, core in enumerate(x_cores):
        if core.ndim != 3:
            raise ValueError(
                f"{name}[{k}] must have 3 dimensions (r_left, mode, r_right); "
                f"got shape {core.shape}."
            )

        r_left, mode, r_right = core.shape
        if r_left <= 0 or mode <= 0 or r_right <= 0:
            raise ValueError(f"{name}[{k}] has invalid non-positive shape {core.shape}.")

        if k == 0 and r_left != 1:
            raise ValueError(f"{name}[0] must have left TT-rank 1; got {r_left}.")

        if k > 0 and prev_rank_right != r_left:
            raise ValueError(
                f"Inconsistent TT ranks between {name}[{k - 1}] and {name}[{k}]: "
                f"{prev_rank_right} != {r_left}."
            )

        prev_rank_right = r_right

    if x_cores[-1].shape[2] != 1:
        raise ValueError(f"{name}[-1] must have right TT-rank 1; got {x_cores[-1].shape[2]}.")


def _validate_tt_matrix_cores(M_cores: Sequence[np.ndarray], *, name: str = "M_cores") -> None:
    """
    Validate TT-matrix cores with shape ``(r_{k-1}, n_k, m_k, r_k)``.
    """
    if len(M_cores) == 0:
        raise ValueError(f"{name} must contain at least one core.")

    prev_rank_right = None
    for k, core in enumerate(M_cores):
        if core.ndim != 4:
            raise ValueError(
                f"{name}[{k}] must have 4 dimensions (r_left, n, m, r_right); "
                f"got shape {core.shape}."
            )

        r_left, n_k, m_k, r_right = core.shape
        if r_left <= 0 or n_k <= 0 or m_k <= 0 or r_right <= 0:
            raise ValueError(f"{name}[{k}] has invalid non-positive shape {core.shape}.")

        if k == 0 and r_left != 1:
            raise ValueError(f"{name}[0] must have left TT-rank 1; got {r_left}.")

        if k > 0 and prev_rank_right != r_left:
            raise ValueError(
                f"Inconsistent TT ranks between {name}[{k - 1}] and {name}[{k}]: "
                f"{prev_rank_right} != {r_left}."
            )

        prev_rank_right = r_right

    if M_cores[-1].shape[3] != 1:
        raise ValueError(f"{name}[-1] must have right TT-rank 1; got {M_cores[-1].shape[3]}.")


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
# TT inner product
# ======================================================================================

def tt_inner_product(A: TTLike, B: TTLike) -> float:
    """
    Compute the TT inner product ``<A, B>`` between two TT vectors.
    """
    A_cores = _as_tt_cores(A)
    B_cores = _as_tt_cores(B)

    _validate_tt_vector_cores(A_cores, name="A_cores")
    _validate_tt_vector_cores(B_cores, name="B_cores")

    if len(A_cores) != len(B_cores):
        raise ValueError(
            f"TT operands must have the same number of cores; got {len(A_cores)} and {len(B_cores)}."
        )

    for k, (A_core, B_core) in enumerate(zip(A_cores, B_cores)):
        if A_core.shape[1] != B_core.shape[1]:
            raise ValueError(
                f"Mode mismatch at core {k}: A has mode {A_core.shape[1]}, "
                f"B has mode {B_core.shape[1]}."
            )

    if len(A_cores) == 0:
        return 0.0

    A1 = A_cores[0]
    B1 = B_cores[0]

    rA0, n1, rA1 = A1.shape
    rB0, _, rB1 = B1.shape

    v = np.zeros((rA0 * rB0, rA1 * rB1), dtype=np.result_type(A1.dtype, B1.dtype))
    for i in range(n1):
        v += np.kron(A1[:, i, :], B1[:, i, :])

    for k in range(1, len(A_cores)):
        Ak = A_cores[k]
        Bk = B_cores[k]

        _, n_k, rA_next = Ak.shape
        _, _, rB_next = Bk.shape

        new_v = np.zeros((v.shape[0], rA_next * rB_next), dtype=np.result_type(Ak.dtype, Bk.dtype))
        for ik in range(n_k):
            kron_prod = np.kron(Ak[:, ik, :], Bk[:, ik, :])
            new_v += v @ kron_prod
        v = new_v

    v = np.squeeze(v)
    if np.ndim(v) == 0:
        return float(v.item())
    if v.size == 1:
        return float(v.reshape(-1)[0])

    return float(np.linalg.norm(v))


def tt_norm(x: TTLike) -> float:
    """
    Compute the Euclidean norm of a TT vector.
    """
    return float(np.sqrt(max(tt_inner_product(x, x), 0.0)))


class TTInnerProductMixin:
    """
    Small mixin adding the `@` operator to TT-vector-like classes.
    """

    tt_cores: List[np.ndarray]

    def __matmul__(self, other: TTLike) -> float:
        return tt_inner_product(self, other)


# ======================================================================================
# Utility helpers
# ======================================================================================

def extract_tt_column(U_cores: TTLike, i: int) -> List[np.ndarray]:
    """
    Extract the `i`-th column of a TT-matrix-like last core as a TT vector.
    """
    cores = _as_tt_cores(U_cores)
    if len(cores) == 0:
        raise ValueError("`U_cores` must contain at least one core.")

    x_tt = [G.copy() for G in cores]
    last = x_tt[-1]

    if last.ndim != 3:
        raise ValueError(
            "The last core must have shape (r_prev, n, r) for column extraction; "
            f"got {last.shape}."
        )

    if not (0 <= i < last.shape[2]):
        raise IndexError(f"Column index {i} out of bounds for last core with shape {last.shape}.")

    x_tt[-1] = last[:, :, i].reshape(last.shape[0], last.shape[1], 1)
    return x_tt


# ======================================================================================
# Optional dense-reference utilities for testing / debugging
# ======================================================================================

def tt_vector_to_dense(x: TTLike) -> np.ndarray:
    """
    Convert a TT vector to a dense NumPy array.
    """
    cores = _as_tt_cores(x)
    _validate_tt_vector_cores(cores)

    result = cores[0][0, :, :]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=([-1], [0]))
    return np.squeeze(result, axis=-1)


def tt_matrix_to_dense(M: TTLike) -> np.ndarray:
    """
    Convert a TT matrix to a dense NumPy array with interleaved input/output modes.
    """
    cores = _as_tt_cores(M)
    _validate_tt_matrix_cores(cores)

    result = cores[0][0, :, :, :]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=([-1], [0]))

    result = np.squeeze(result, axis=-1)

    d = len(cores)
    n_dims = [core.shape[1] for core in cores]
    m_dims = [core.shape[2] for core in cores]

    perm_n = list(range(0, 2 * d, 2))
    perm_m = list(range(1, 2 * d, 2))
    result = result.transpose(*(perm_n + perm_m))
    return result.reshape(*n_dims, *m_dims)









