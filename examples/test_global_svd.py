import numpy as np

from tensor_gedmd.reps.Tensor_Train import TT
from tensor_gedmd.algorithms.Global_SVD import global_svd_tt


np.random.seed(0)

cores = [
    np.random.rand(1, 4, 8),
    np.random.rand(8, 5, 8),
    np.random.rand(8, 8, 1),
]

psi_tt = TT(cores)

print("Input TT:")
print(psi_tt)
print("Ranks:", psi_tt.tt_ranks())
print("Mode sizes:", psi_tt.mode_sizes())
print("Is operator:", psi_tt.is_operator)
print()


def summarize_result(name, Sigma, Sigma_full=None):
    Sigma = np.asarray(Sigma).ravel()
    rank = len(Sigma)

    print(f"=== {name} ===")
    print("Retained rank:", rank)
    print("Singular values:", Sigma)

    if rank > 0:
        print("Largest singular value:", Sigma[0])
        print("Smallest retained singular value:", Sigma[-1])
        print("Condition number (retained):", Sigma[0] / Sigma[-1] if Sigma[-1] > 0 else np.inf)

    retained_energy = np.sum(Sigma**2)
    print("Retained energy ||Sigma||_F^2:", retained_energy)

    if Sigma_full is not None:
        Sigma_full = np.asarray(Sigma_full).ravel()
        total_energy = np.sum(Sigma_full**2)
        discarded_energy = total_energy - retained_energy
        rel_energy = retained_energy / total_energy if total_energy > 0 else 0.0
        rel_error = np.sqrt(discarded_energy / total_energy) if total_energy > 0 else 0.0

        print("Total energy ||Sigma_full||_F^2:", total_energy)
        print("Discarded energy:", discarded_energy)
        print("Retained energy fraction:", rel_energy)
        print("Relative truncation error:", rel_error)

    print()


# Full decomposition: use as reference
U_cores_full, Sigma_full, V_core_full = global_svd_tt(psi_tt)

print("=== No truncation ===")
print("Number of U cores:", len(U_cores_full))
print("U core shapes:", [core.shape for core in U_cores_full])
print("Sigma shape:", np.shape(Sigma_full))
print("V core shape:", np.shape(V_core_full))
print()

summarize_result("No truncation", Sigma_full)


# rmax only
U_cores_rmax, Sigma_rmax, V_core_rmax = global_svd_tt(psi_tt, rmax=2)

print("U core shapes:", [core.shape for core in U_cores_rmax])
print("Sigma shape:", np.shape(Sigma_rmax))
print("V core shape:", np.shape(V_core_rmax))
print()

summarize_result("Only rmax=2", Sigma_rmax, Sigma_full)


# tol only
U_cores_tol, Sigma_tol, V_core_tol = global_svd_tt(psi_tt, tol=1e-6)

print("U core shapes:", [core.shape for core in U_cores_tol])
print("Sigma shape:", np.shape(Sigma_tol))
print("V core shape:", np.shape(V_core_tol))
print()

summarize_result("Only tol=1e-6", Sigma_tol, Sigma_full)


# both rmax and tol
U_cores_both, Sigma_both, V_core_both = global_svd_tt(psi_tt, rmax=2, tol=1e-6)

print("U core shapes:", [core.shape for core in U_cores_both])
print("Sigma shape:", np.shape(Sigma_both))
print("V core shape:", np.shape(V_core_both))
print()

summarize_result("Both rmax=2 and tol=1e-6", Sigma_both, Sigma_full)