import numpy as np
import matplotlib.pyplot as plt

from tensor_gedmd.reps.tensor_train import TT
from tensor_gedmd.algorithms.global_svd import global_svd_tt, global_svd_data_tensor


def tt_to_dense(tt: TT) -> np.ndarray:
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

    X = np.squeeze(X, axis=0)
    return X


def reconstruction_from_factors(U_tt: TT, Sigma: np.ndarray, V_core: np.ndarray) -> np.ndarray:
    """
    Generic reconstruction:
        U_dense @ Sigma @ V

    If
        U_dense has shape (..., r),
        Sigma has shape (r, r),
        V_core has shape (r, q, 1),
    then the reconstruction has shape (..., q).
    """
    U_dense = tt_to_dense(U_tt)

    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be square, got shape {Sigma.shape}.")

    if V_core.ndim != 3 or V_core.shape[2] != 1:
        raise ValueError(f"V_core must have shape (r, q, 1), got {V_core.shape}.")

    r = Sigma.shape[0]
    if U_dense.shape[-1] != r:
        raise ValueError(
            f"Last mode of U_dense must match Sigma rank: {U_dense.shape[-1]} != {r}."
        )
    if V_core.shape[0] != r:
        raise ValueError(
            f"First mode of V_core must match Sigma rank: {V_core.shape[0]} != {r}."
        )

    V_mat = V_core[:, :, 0]
    US = np.tensordot(U_dense, Sigma, axes=([-1], [0]))
    A_rec = np.tensordot(US, V_mat, axes=([-1], [0]))
    return A_rec


def relative_error(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch in relative_error: A.shape={A.shape}, B.shape={B.shape}"
        )
    denom = max(np.linalg.norm(A), 1e-15)
    return np.linalg.norm(A - B) / denom


def dense_tensor_from_basis_evals(basis_evals: list[np.ndarray]) -> np.ndarray:
    """
    Build dense reference tensor

        A[i0, i1, ..., i_{p-1}, s] = prod_k basis_evals[k][s, i_k]

    from basis evaluations basis_evals[k] of shape (m, n_k).
    """
    if not isinstance(basis_evals, list) or len(basis_evals) == 0:
        raise ValueError("basis_evals must be a non-empty list.")

    m = basis_evals[0].shape[0]
    for k, B in enumerate(basis_evals):
        if not isinstance(B, np.ndarray):
            raise TypeError(f"basis_evals[{k}] must be a numpy.ndarray.")
        if B.ndim != 2:
            raise ValueError(f"basis_evals[{k}] must be 2D, got shape {B.shape}.")
        if B.shape[0] != m:
            raise ValueError(
                f"All basis evaluations must have the same sample dimension m. "
                f"basis_evals[0].shape[0]={m}, basis_evals[{k}].shape[0]={B.shape[0]}."
            )

    A = basis_evals[0].T
    for B in basis_evals[1:]:
        A = np.einsum("...s,sj->...js", A, B, optimize=True)

    return A


def run_single_test(
    method_name: str,
    method,
    basis_evals: list[np.ndarray],
    A_ref: np.ndarray | None = None,
    rmax=None,
    tol: float = 0.0,
):
    U_tt, Sigma, V_core = method(basis_evals, rmax=rmax, tol=tol)
    A_rec = reconstruction_from_factors(U_tt, Sigma, V_core)

    print(
        f"[{method_name}] "
        f"U_dense shape={tt_to_dense(U_tt).shape}, "
        f"Sigma shape={Sigma.shape}, "
        f"V_core shape={V_core.shape}, "
        f"A_rec shape={A_rec.shape}"
    )

    err = None
    if A_ref is not None:
        if A_ref.shape == A_rec.shape:
            err = relative_error(A_ref, A_rec)
            print(
                f"[{method_name}] rmax={rmax}, tol={tol:.1e}, "
                f"retained_rank={Sigma.shape[0]}, rel_error={err:.6e}"
            )
        else:
            print(
                f"[{method_name}] reference comparison skipped: "
                f"A_ref shape={A_ref.shape} != A_rec shape={A_rec.shape}"
            )

    return U_tt, Sigma, V_core, A_rec, err


if __name__ == "__main__":
    np.random.seed(7)

    m = 8
    basis_evals = [
        np.random.randn(m, 4),
        np.random.randn(m, 5),
        np.random.randn(m, 6),
    ]

    print("Basis evaluation shapes:", [B.shape for B in basis_evals])

    # Reference tensor for the data-tensor formulation:
    # shape = (n0, n1, ..., n_{p-1}, m)
    A_ref_data = dense_tensor_from_basis_evals(basis_evals)
    print("A_ref_data shape:", A_ref_data.shape)

    # ==========================================================
    # global_svd_data_tensor
    # ==========================================================
    U_dt, S_dt, V_dt, A_rec_dt, err_dt = run_single_test(
        "global_svd_data_tensor",
        global_svd_data_tensor,
        basis_evals,
        A_ref=A_ref_data,
        rmax=None,
        tol=0.0,
    )

    # ==========================================================
    # global_svd_tt
    # ==========================================================
    U_tt, S_tt, V_tt, A_rec_tt, err_tt = run_single_test(
        "global_svd_tt",
        global_svd_tt,
        basis_evals,
        A_ref=A_ref_data,
        rmax=None,
        tol=0.0,
    )

    # ==========================================================
    # Compare the two reconstructions only if shapes match
    # ==========================================================
    if A_rec_dt.shape == A_rec_tt.shape:
        diff = relative_error(A_rec_dt, A_rec_tt)
        print("Relative difference between both reconstructions:", diff)
    else:
        print(
            "Cannot compare both reconstructions directly: "
            f"A_rec_dt.shape={A_rec_dt.shape}, A_rec_tt.shape={A_rec_tt.shape}"
        )

    # ==========================================================
    # Rank truncation study: data tensor
    # ==========================================================
    if err_dt is not None:
        full_rank_dt = S_dt.shape[0]
        rmax_values = list(range(1, full_rank_dt + 1))
        rank_errors_dt = []

        for rmax in rmax_values:
            _, _, _, A_rec, err = run_single_test(
                "global_svd_data_tensor",
                global_svd_data_tensor,
                basis_evals,
                A_ref=A_ref_data,
                rmax=rmax,
                tol=0.0,
            )
            rank_errors_dt.append(err)

        plt.figure()
        plt.plot(rmax_values, rank_errors_dt, marker="o")
        plt.yscale("log")
        plt.xlabel("Truncation rank (rmax)")
        plt.ylabel("Relative reconstruction error")
        plt.title("global_svd_data_tensor: Error vs truncation rank")
        plt.grid(True)
        plt.show()

    # ==========================================================
    # Tolerance truncation study: data tensor
    # ==========================================================
    if err_dt is not None:
        tol_values = np.logspace(-12, -1, 12)
        tol_errors_dt = []
        tol_ranks_dt = []

        for tol in tol_values:
            _, Sigma, _, _, err = run_single_test(
                "global_svd_data_tensor",
                global_svd_data_tensor,
                basis_evals,
                A_ref=A_ref_data,
                rmax=None,
                tol=tol,
            )
            tol_errors_dt.append(err)
            tol_ranks_dt.append(Sigma.shape[0])

        plt.figure()
        plt.plot(tol_values, tol_errors_dt, marker="o")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Tolerance")
        plt.ylabel("Relative reconstruction error")
        plt.title("global_svd_data_tensor: Error vs truncation tolerance")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(tol_values, tol_ranks_dt, marker="o")
        plt.xscale("log")
        plt.xlabel("Tolerance")
        plt.ylabel("Retained rank")
        plt.title("global_svd_data_tensor: Retained rank vs tolerance")
        plt.grid(True)
        plt.show()