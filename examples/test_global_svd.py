import numpy as np
import matplotlib.pyplot as plt

from tensor_gedmd.reps.Tensor_Train import TT
from tensor_gedmd.algorithms.Global_SVD import global_svd_tt_general


def tt_to_dense(tt: TT) -> np.ndarray:
    if tt.is_operator:
        raise ValueError("tt_to_dense expects a tensor TT, not an operator TT.")

    X = tt.get_core(0)

    for k in range(1, len(tt)):
        G = tt.get_core(k)
        X = np.tensordot(X, G, axes=([-1], [0]))

    X = np.squeeze(X, axis=0)

    # Only squeeze last axis if TT is closed on right
    if getattr(tt, "require_right_rank_one", True) and X.shape[-1] == 1:
        X = np.squeeze(X, axis=-1)

    return X


def reconstruction_from_factors(U_tt, Sigma, V_core):
    U_dense = tt_to_dense(U_tt)
    r = U_dense.shape[-1]

    V_mat = V_core[:, :, 0]

    US = np.tensordot(U_dense, Sigma, axes=([-1], [0]))
    A_rec = np.tensordot(US, V_mat, axes=([-1], [0]))

    return A_rec


def relative_error(A, B):
    return np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-15)


if __name__ == "__main__":
    np.random.seed(7)

    # Build TT
    G0 = np.random.randn(1, 4, 6)
    G1 = np.random.randn(6, 5, 7)
    G2 = np.random.randn(7, 8, 1)

    psi_tt = TT([G0, G1, G2])

    print("Input TT:", psi_tt)

    A_original = tt_to_dense(psi_tt)

    # ==========================================================
    # No truncation
    # ==========================================================
    U_tt_full, Sigma_full, V_core_full = global_svd_tt_general(
        psi_tt,
        rmax=None,
        tol=0.0,
    )

    A_rec_full = reconstruction_from_factors(U_tt_full, Sigma_full, V_core_full)

    err_full = relative_error(A_original, A_rec_full)

    print("No truncation relative error:", err_full)

    # ==========================================================
    # 1) Truncation by rank
    # ==========================================================
    full_rank = Sigma_full.shape[0]
    rmax_values = list(range(1, full_rank + 1))

    rank_errors = []

    for rmax in rmax_values:
        U_tt, Sigma, V_core = global_svd_tt_general(
            psi_tt,
            rmax=rmax,
            tol=0.0,
        )

        A_rec = reconstruction_from_factors(U_tt, Sigma, V_core)
        err = relative_error(A_original, A_rec)
        rank_errors.append(err)

        print(f"rmax={rmax}, error={err}")

    # Plot error vs truncation rank
    plt.figure()
    plt.plot(rmax_values, rank_errors, marker="o")
    plt.yscale("log")
    plt.xlabel("Truncation rank (rmax)")
    plt.ylabel("Relative reconstruction error")
    plt.title("Error vs truncation rank")
    plt.grid(True)
    plt.show()

    # ==========================================================
    # 2) Truncation by tolerance
    # ==========================================================
    tol_values = np.logspace(-12, -1, 12)
    tol_errors = []
    tol_ranks = []

    for tol in tol_values:
        U_tt, Sigma, V_core = global_svd_tt_general(
            psi_tt,
            rmax=None,
            tol=tol,
        )

        A_rec = reconstruction_from_factors(U_tt, Sigma, V_core)
        err = relative_error(A_original, A_rec)

        tol_errors.append(err)
        tol_ranks.append(Sigma.shape[0])

        print(f"tol={tol:.1e}, rank={Sigma.shape[0]}, error={err}")

    # Plot error vs tolerance
    plt.figure()
    plt.plot(tol_values, tol_errors, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Tolerance")
    plt.ylabel("Relative reconstruction error")
    plt.title("Error vs truncation tolerance")
    plt.grid(True)
    plt.show()

    # Plot retained rank vs tolerance
    plt.figure()
    plt.plot(tol_values, tol_ranks, marker="o")
    plt.xscale("log")
    plt.xlabel("Tolerance")
    plt.ylabel("Retained rank")
    plt.title("Retained rank vs tolerance")
    plt.grid(True)
    plt.show()

