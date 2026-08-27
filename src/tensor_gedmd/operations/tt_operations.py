"""
Generic operations on TT (tensor-train) objects.

These helpers are agnostic to any particular algorithm (global SVD, gEDMD,
stiffness operators, ...) and are used across the package and in examples,
so they are collected here in one place instead of being duplicated.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np

from tensor_gedmd.reps.tensor_train import TT

TTLike = Union[TT, Sequence[np.ndarray]]


# ======================================================================================
# Core-list helpers / validation
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


# ======================================================================================
# TT <-> dense conversions
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


def tt_to_dense(tt: TT) -> np.ndarray:
    """
    Convert a tensor-form TT object (3D cores) to a dense NumPy array.

    This is the general-purpose counterpart of ``tt_vector_to_dense`` that
    works directly on a ``TT`` instance (rather than a raw list of cores)
    and does not require the trailing TT rank to be 1 -- useful for e.g.
    the left factor returned by ``global_svd_tt``/``global_svd_data_tensor``,
    which intentionally has an open right boundary.
    """
    if tt.is_operator:
        raise ValueError("tt_to_dense expects a tensor TT, not an operator TT.")

    X = tt.get_core(0)
    if X.ndim != 3:
        raise ValueError(f"Core 0 must be 3D, got shape {X.shape}.")

    for k in range(1, len(tt)):
        G = tt.get_core(k)
        if G.ndim != 3:
            raise ValueError(f"Core {k} must be 3D, got shape {G.shape}.")
        X = np.tensordot(X, G, axes=([-1], [0]))

    if X.shape[0] != 1:
        raise ValueError(f"Expected left boundary rank 1, got shape {X.shape}.")

    return np.squeeze(X, axis=0)


def tt_operator_to_dense(tt: TT) -> np.ndarray:
    """
    Convert an operator-form TT object (4D cores) to a dense matrix.

    Cores have shape ``(r_{k-1}, n_k, n_k, r_k)``; the result has shape
    ``(prod_k n_k, prod_k n_k)``.
    """
    cores = tt.cores if isinstance(tt, TT) else _as_tt_cores(tt)
    if len(cores) == 0:
        raise ValueError("TT has no cores.")

    first = cores[0]
    if first.ndim != 4:
        raise ValueError("Expected TT operator cores with 4 dimensions.")
    if first.shape[0] != 1:
        raise ValueError("First TT rank must be 1.")

    tensor = first[0]  # shape: (n1, n1, r1)

    for k in range(1, len(cores)):
        core = cores[k]
        if core.ndim != 4:
            raise ValueError(f"Core {k} is not a 4D TT operator core.")
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))

    if tensor.shape[-1] != 1:
        raise ValueError("Last TT rank must be 1.")

    tensor = tensor[..., 0]

    p = len(cores)
    perm = list(range(0, 2 * p, 2)) + list(range(1, 2 * p, 2))
    tensor = np.transpose(tensor, axes=perm)

    row_dims = [core.shape[1] for core in cores]
    col_dims = [core.shape[2] for core in cores]

    return tensor.reshape(int(np.prod(row_dims)), int(np.prod(col_dims)))


# ======================================================================================
# TT inner product / norm
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
# Misc utility helpers
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
