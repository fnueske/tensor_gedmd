import numpy as np

from tensor_gedmd.reps.Tensor_Train import TT
from tensor_gedmd.algorithms.Global_SVD import global_svd_tt


cores = [
    np.random.rand(1, 4, 7),
    np.random.rand(7, 5, 8),
    np.random.rand(8, 10, 1),
]

psi_tt = TT(cores)

print("Input TT:")
print(psi_tt)
print("Ranks:", psi_tt.tt_ranks())
print("Mode sizes:", psi_tt.mode_sizes())
print("Is operator:", psi_tt.is_operator)
print()

U_cores, Sigma, V_core = global_svd_tt(psi_tt)

print("=== No truncation ===")
print("Number of U cores:", len(U_cores))
print("U core shapes:", [core.shape for core in U_cores])
print("Sigma shape:", Sigma.shape)
print("V core shape:", V_core.shape)
print()

U_cores, Sigma, V_core = global_svd_tt(psi_tt, rmax=2)

print("=== Only rmax=2 ===")
print("U core shapes:", [core.shape for core in U_cores])
print("Sigma shape:", Sigma.shape)
print("V core shape:", V_core.shape)
print()

U_cores, Sigma, V_core = global_svd_tt(psi_tt, tol=1e-6)

print("=== Only tol=1e-6 ===")
print("U core shapes:", [core.shape for core in U_cores])
print("Sigma shape:", Sigma.shape)
print("V core shape:", V_core.shape)
print()

U_cores, Sigma, V_core = global_svd_tt(psi_tt, rmax=2, tol=1e-6)

print("=== Both rmax=2 and tol=1e-6 ===")
print("U core shapes:", [core.shape for core in U_cores])
print("Sigma shape:", Sigma.shape)
print("V core shape:", V_core.shape)
print()


