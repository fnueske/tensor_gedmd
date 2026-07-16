from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensor_gedmd.algorithms.mat_vec_prod import (
    make_A_mv,
    tt_inner_product,
    tt_matrix_to_dense,
    tt_matrix_vector_product_csr_prepared,
    tt_matrix_vector_product_general,
    tt_norm,
    tt_vector_to_dense,
)


# =========================================================
# Helpers
# =========================================================
@dataclass
class DummyGeneratorOp:
    tg_cores: list[np.ndarray]


def random_tt_vector(
    dims: list[int],
    ranks: list[int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Build a random TT vector with cores of shape (r_{k-1}, n_k, r_k).
    """
    if len(ranks) != len(dims) + 1:
        raise ValueError("ranks must have length len(dims) + 1.")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("TT boundary ranks must be 1.")

    return [
        rng.normal(size=(ranks[k], dims[k], ranks[k + 1]))
        for k in range(len(dims))
    ]


def random_tt_matrix(
    out_dims: list[int],
    in_dims: list[int],
    ranks: list[int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Build a random TT matrix with cores of shape (r_{k-1}, n_k, m_k, r_k).
    """
    if len(out_dims) != len(in_dims):
        raise ValueError("out_dims and in_dims must have the same length.")
    if len(ranks) != len(out_dims) + 1:
        raise ValueError("ranks must have length len(out_dims) + 1.")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("TT boundary ranks must be 1.")

    return [
        rng.normal(size=(ranks[k], out_dims[k], in_dims[k], ranks[k + 1]))
        for k in range(len(out_dims))
    ]


def relative_error_norm(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute relative error norm ||a-b|| / max(||b||, eps).
    """
    num = np.linalg.norm(a.reshape(-1) - b.reshape(-1))
    den = max(np.linalg.norm(b.reshape(-1)), 1e-15)
    return float(num / den)


def absolute_error_norm(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute absolute error norm ||a-b||.
    """
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)))


# =========================================================
# Demo 1: 4-core general TT mat-vec vs dense
# =========================================================
def demo_general_vs_dense_4cores() -> None:
    print("\n" + "=" * 70)
    print("DEMO 1: 4-CORE GENERAL TT MAT-VEC vs DENSE")
    print("=" * 70)

    rng = np.random.default_rng(0)

    # at least 4 cores
    out_dims = [2, 3, 2, 2]
    in_dims = [2, 2, 3, 2]

    M = random_tt_matrix(out_dims, in_dims, [1, 2, 3, 2, 1], rng)
    x = random_tt_vector(in_dims, [1, 2, 2, 2, 1], rng)

    y_tt = tt_matrix_vector_product_general(M, x)

    y_tt_dense = tt_vector_to_dense(y_tt).reshape(*out_dims)
    M_dense = tt_matrix_to_dense(M).reshape(np.prod(out_dims), np.prod(in_dims))
    x_dense = tt_vector_to_dense(x).reshape(-1)
    y_dense_ref = (M_dense @ x_dense).reshape(*out_dims)

    abs_err = absolute_error_norm(y_tt_dense, y_dense_ref)
    rel_err = relative_error_norm(y_tt_dense, y_dense_ref)

    print("\nOutput dims:", out_dims)
    print("Input dims :", in_dims)

    print("\nDense reference norm      :", np.linalg.norm(y_dense_ref.reshape(-1)))
    print("TT result norm            :", np.linalg.norm(y_tt_dense.reshape(-1)))
    print("Absolute error norm       :", abs_err)
    print("Relative error norm       :", rel_err)
    print("allclose                  :", np.allclose(y_tt_dense, y_dense_ref, atol=1e-10, rtol=1e-10))

    print("\nTT inner product / norm checks:")
    print("<y_tt, y_tt>              :", tt_inner_product(y_tt, y_tt))
    print("tt_norm(y_tt)             :", tt_norm(y_tt))
    print("dense ||y_tt||            :", np.linalg.norm(y_tt_dense.reshape(-1)))


# =========================================================
# Demo 2: 4-core via make_A_mv(use_general=True)
# =========================================================
def demo_make_A_mv_general_4cores() -> None:
    print("\n" + "=" * 70)
    print("DEMO 2: make_A_mv(use_general=True) WITH 4 CORES")
    print("=" * 70)

    rng = np.random.default_rng(1)

    out_dims = [2, 2, 3, 2]
    in_dims = [2, 3, 2, 2]

    M = random_tt_matrix(out_dims, in_dims, [1, 2, 2, 2, 1], rng)
    x = random_tt_vector(in_dims, [1, 2, 2, 2, 1], rng)

    op = DummyGeneratorOp(tg_cores=M)
    A_mv = make_A_mv(op, use_general=True)

    y_tt = A_mv(x)

    y_tt_dense = tt_vector_to_dense(y_tt).reshape(*out_dims)
    M_dense = tt_matrix_to_dense(M).reshape(np.prod(out_dims), np.prod(in_dims))
    x_dense = tt_vector_to_dense(x).reshape(-1)
    y_dense_ref = (M_dense @ x_dense).reshape(*out_dims)

    abs_err = absolute_error_norm(y_tt_dense, y_dense_ref)
    rel_err = relative_error_norm(y_tt_dense, y_dense_ref)

    print("\nDense reference norm      :", np.linalg.norm(y_dense_ref.reshape(-1)))
    print("TT result norm            :", np.linalg.norm(y_tt_dense.reshape(-1)))
    print("Absolute error norm       :", abs_err)
    print("Relative error norm       :", rel_err)
    print("allclose                  :", np.allclose(y_tt_dense, y_dense_ref, atol=1e-10, rtol=1e-10))


# =========================================================
# Demo 3: specialized routine for d=2 still works
# =========================================================
def demo_specialized_d2_error_norm() -> None:
    print("\n" + "=" * 70)
    print("DEMO 3: SPECIALIZED ROUTINE FOR d=2")
    print("=" * 70)

    rng = np.random.default_rng(2)

    out_dims = [2, 3]
    in_dims = [2, 2]

    M = random_tt_matrix(out_dims, in_dims, [1, 2, 1], rng)
    x = random_tt_vector(in_dims, [1, 2, 1], rng)

    y_general = tt_matrix_vector_product_general(M, x)
    y_special = tt_matrix_vector_product_csr_prepared(M, x)

    y_general_dense = tt_vector_to_dense(y_general).reshape(*out_dims)
    y_special_dense = tt_vector_to_dense(y_special).reshape(*out_dims)

    abs_err = absolute_error_norm(y_special_dense, y_general_dense)
    rel_err = relative_error_norm(y_special_dense, y_general_dense)

    print("\nGeneral norm              :", np.linalg.norm(y_general_dense.reshape(-1)))
    print("Specialized norm          :", np.linalg.norm(y_special_dense.reshape(-1)))
    print("Absolute error norm       :", abs_err)
    print("Relative error norm       :", rel_err)
    print("allclose                  :", np.allclose(y_special_dense, y_general_dense, atol=1e-10, rtol=1e-10))


# =========================================================
# Main runner for demos
# =========================================================
def main() -> None:
    print("script started")
    demo_general_vs_dense_4cores()
    demo_make_A_mv_general_4cores()
    demo_specialized_d2_error_norm()
    print("\nDone.")


if __name__ == "__main__":
    main()
