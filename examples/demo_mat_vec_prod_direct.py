"""
Demo: tensor_gedmd.algorithms.mat_vec_prod_direct.compute_A_r

A faster drop-in alternative to mat_vec_prod.compute_A_r that classifies
Sigma's structure (identity / scalar*I / diagonal / dense, each either
constant or samplewise) and skips the general (p, p) contraction whenever
a cheaper equivalent is available. Verifies exact agreement with the
original compute_A_r across every Sigma structure, then times both.
"""

from __future__ import annotations

import time

import numpy as np

from tensor_gedmd.algorithms.mat_vec_prod_direct import classify_sigma
from tensor_gedmd.algorithms.mat_vec_prod_direct import compute_A_r as compute_A_r_fast
from tensor_gedmd.algorithms.global_svd import global_svd_tt
from tensor_gedmd.algorithms.mat_vec_prod import compute_A_r as compute_A_r_orig
from tensor_gedmd.reps.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.reps.transformed_data_tensor import Transformed_Data_Tensor_TT


def build_reduced_basis(psi, tol=1e-8):
    """Same construction the full gedmd.py pipeline uses to get U_cores."""
    builder = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)
    U_tt, Sigma_svd, V_core = global_svd_tt(builder.to_tt(), rmax=None, tol=tol)
    return U_tt.cores


def compare_one_case(label, psi, dpsi, Sigma, U_cores, chunk_size=50, n_workers=2):
    print("\n" + "=" * 70)
    print(f"CASE: {label}")
    print("=" * 70)

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
    r = U_cores[-1].shape[2]

    kind, _ = classify_sigma(op)
    print(f"Sigma classified as: {kind}")

    t0 = time.perf_counter()
    A_r_orig = compute_A_r_orig(op, U_cores, r, chunk_size=chunk_size, n_workers=n_workers)
    t_orig = time.perf_counter() - t0

    t0 = time.perf_counter()
    A_r_fast = compute_A_r_fast(op, U_cores, r, chunk_size=chunk_size, n_workers=n_workers)
    t_fast = time.perf_counter() - t0

    max_diff = np.max(np.abs(A_r_orig - A_r_fast))
    print(f"\nmat_vec_prod.compute_A_r        : {t_orig * 1000:.2f} ms")
    print(f"mat_vec_prod_direct.compute_A_r    : {t_fast * 1000:.2f} ms")
    print(f"max |A_r_orig - A_r_fast|       : {max_diff:.3e}")
    print(f"outputs identical               : {np.allclose(A_r_orig, A_r_fast, atol=1e-10)}")


def main() -> None:
    rng = np.random.default_rng(0)
    p, m, dims = 3, 300, [4, 5, 4]

    psi = [rng.normal(size=(n, m)) for n in dims]
    dpsi = [rng.normal(size=(n, m)) for n in dims]

    print("Building reduced TT basis (shared across all cases below) ...")
    U_cores = build_reduced_basis(psi)
    print("U_cores shapes:", [c.shape for c in U_cores])

    # Case 1: no diffusion tensor at all (implicit identity)
    compare_one_case("Sigma = None (identity)", psi, dpsi, None, U_cores)

    # Case 2: scalar-multiple-of-identity, isotropic constant diffusion
    # (this is the specific case the "compute_A_r_fast" reference started from)
    compare_one_case("Sigma = 2.5 * I  (scalar)", psi, dpsi, 2.5 * np.eye(p), U_cores)

    # Case 3: diagonal but non-isotropic constant diffusion
    compare_one_case(
        "Sigma = diag([0.5, 2.0, 1.2])  (diagonal, constant)",
        psi, dpsi, np.diag(np.array([0.5, 2.0, 1.2])), U_cores,
    )

    # Case 4: general dense constant diffusion (with real cross-terms)
    A = rng.normal(size=(p, p))
    Sigma_dense_const = A @ A.T + np.eye(p)
    compare_one_case(
        "Sigma = dense constant (with cross-terms)",
        psi, dpsi, Sigma_dense_const, U_cores,
    )

    # Case 5: diagonal but samplewise (varies per data point) diffusion
    Sigma_diag_var = np.zeros((p, p, m))
    for l in range(m):
        Sigma_diag_var[:, :, l] = np.diag(rng.uniform(0.5, 3.0, size=p))
    compare_one_case(
        "Sigma = diagonal, samplewise",
        psi, dpsi, Sigma_diag_var, U_cores,
    )

    # Case 6: fully general samplewise diffusion tensor (like a real MD/TICA run)
    Sigma_dense_var = np.zeros((p, p, m))
    for l in range(m):
        A = rng.normal(size=(p, p))
        Sigma_dense_var[:, :, l] = A @ A.T + np.eye(p)
    compare_one_case(
        "Sigma = dense, samplewise (with cross-terms)",
        psi, dpsi, Sigma_dense_var, U_cores,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
