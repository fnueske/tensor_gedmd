"""
Lemon Slice 3D -- lemon_slice_3d_run_calculation_base.py

Base file: the actual idea being tested here, with everything needed to
run it once, made explicit and runnable standalone -- separate from the
"repeat 10 times per data size m for error bars" orchestration in
run_calculation.py.

Two ideas live here, both reused by the sweeps in run_calculation.py:

  Eigenvalue idea: for one random subsample of the Lemon-Slice SDE
  trajectory, compute the leading eigenvalues both the dense/"Matrix"
  reference way and the tensor-train way, and see that they agree.

  Rank idea: for one random subsample, build the TT data tensor and see
  what TT rank the global SVD retains at a given truncation tolerance.

run_calculation.py imports build_simulator / simulate /
spectral_analysis_gedmd_dense / tensor_spectral_analysis / get_max_tt_rank
from here and wraps them in loops over data sizes (M_VALUES_EIG /
M_VALUES_RANK) x 10 independent experiments each, to produce
lemon_slice_3d_results.npz (the actual paper figure's data). It also uses
build_simulator/simulate for its own Section A (a single, non-repeated
PCCA run) and Section B.3 (a runtime benchmark that's already
single-experiment).

Run this file directly for a quick, single-experiment demo of both ideas
at one fixed m -- no repeats, no sweep, no plot. Confirms the pipeline
runs correctly before committing to the full sweep (the heaviest script in
publication/).

Requirements
------------
- The `tensor_gedmd` package (this repository) importable -- includes the
  Lemon-Slice `Decoupled_3d` SDE simulator (tensor_gedmd.systems), no
  external simulator package needed.

Usage
-----
    python lemon_slice_3d_run_calculation_base.py
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.linalg import eigh

from tensor_gedmd.algorithms.gedmd import build_deterministic_rff_basis, evaluate_basis
from tensor_gedmd.algorithms.global_svd import global_svd_tt
from tensor_gedmd.algorithms.mat_vec_prod_direct import compute_A_r
from tensor_gedmd.algorithms.util import filter_ev, spectral_analysis_gedmd_rev
from tensor_gedmd.reps.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.reps.transformed_data_tensor import Transformed_Data_Tensor_TT
from tensor_gedmd.systems.decoupled_potential_3d import Decoupled_3d

# ----------------------------------------------------------------------
# Physical/pipeline constants (verbatim from the original notebook) -- the
# same regardless of which data size m or repeat is being tested, so they
# live here rather than in the sweep-orchestration file.
# ----------------------------------------------------------------------
KK, LL, Z_PARAM = 1.0, 1.0, 5.0
BETA = 1.0                      # inverse temperature
DIFFUSION_CONST = 2.0 / BETA    # scalar isotropic Sigma = DIFFUSION_CONST * I_p

NTEST = 1
DT = 1e-3
DSAVE = 20

N_FEATURES = 10                 # RFF features per dimension
LENGTH_SCALE = 4.0               # RFF length scale ("sigma" in the notebook)
P_DIMS = 3

R_TRUNC = 400                    # global_svd_tt rmax


# ============================================================
# Simulator
# ============================================================

def build_simulator():
    return Decoupled_3d(BETA, KK, LL, Z_PARAM)


def simulate(LS, n_snapshots: int) -> np.ndarray:
    """Simulate and return (p, n_snapshots) trajectory."""
    x0 = np.ones((NTEST, 3))
    Xfull = LS.simulate(x0, n_snapshots * DSAVE, DT)[:, ::DSAVE]
    return Xfull[:, :-1]


# ============================================================
# Dense/"Matrix" reference method.
#
# Uses this project's mean-free spectral_analysis_gedmd_rev
# (tensor_gedmd.algorithms.util) -- see that function's docstring for why
# this is deliberately mean-free, not the plain dmp_methods.gEDMD.gEDMD
# library behavior.
# ============================================================

def spectral_analysis_gedmd_dense(X, phi, nev, diffusion_const, tol=1e-4, eps_ev=0.0):
    """Dense/"Matrix" reference-method eigendecomposition (thin wrapper
    kept for this file's naming/argument convention)."""
    d, W, Wdata = spectral_analysis_gedmd_rev(X, phi, nev, a=diffusion_const, tol=tol, eps_ev=eps_ev)
    return d[::-1], W[:, ::-1], Wdata[::-1]


# ============================================================
# Tensor (TT) method: reduced generator -> whitened eigenproblem
# ============================================================

def tensor_spectral_analysis(Xlist, basis_list, r_trunc, svd_tol, nev, eps_ev=0.0):
    psi, dpsi = evaluate_basis(Xlist, basis_list)
    m = Xlist.shape[1]

    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k):
        return data_tensor.build_core(k + 1)

    U_tt, Sigma_svd, V_core = global_svd_tt(core_getter, num_cores=len(psi) + 1,
                                             rmax=r_trunc, tol=svd_tol)
    U_cores = U_tt.cores
    r = U_cores[-1].shape[2]

    Z = np.sqrt(m) * V_core[:, :, 0]
    mean_z = np.mean(Z, axis=1)
    G1 = np.eye(r) - np.outer(mean_z, mean_z)
    d_G, W_G = eigh(G1)
    d_G, W_G = d_G[1:], W_G[:, 1:]

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=DIFFUSION_CONST * np.eye(len(psi)))
    A_r = compute_A_r(op, U_cores, r, chunk_size=200, n_workers=None, r_cap=r)

    Sigma_inv = np.linalg.inv(Sigma_svd)
    reduced_matrix = Sigma_inv @ A_r @ Sigma_inv

    Dg_inv_sqrt = np.diag(d_G ** (-0.5))
    R_white = Dg_inv_sqrt @ W_G.T @ reduced_matrix @ W_G @ Dg_inv_sqrt

    d_tt, W_tt = eigh(R_white)
    d_tt, W_tt = filter_ev(d_tt, W_tt, eps2=-eps_ev)
    d_tt = d_tt[-nev:][::-1]
    W_tt = W_tt[:, -nev:][:, ::-1]

    Z_white = Dg_inv_sqrt @ (W_G.T @ Z)
    return d_tt, W_tt, Z_white, U_cores, Sigma_svd


def get_max_tt_rank(U_cores: List[np.ndarray]) -> int:
    ranks = []
    for core in U_cores:
        ranks.append(core.shape[0])
        ranks.append(core.shape[2])
    return max(ranks)


# ============================================================
# Demo: one fixed m, no repeats -- both ideas (eigenvalues, rank)
# ============================================================

if __name__ == "__main__":
    from tensor_gedmd.basis_sets.product_basis import ProductBasis

    DEMO_M = 2000
    DEMO_TOL_RANK = 1e-4

    LS = build_simulator()
    Xlist = simulate(LS, DEMO_M)
    basis_list = build_deterministic_rff_basis(P_DIMS, N_FEATURES, LENGTH_SCALE)

    print(f"\n--- Eigenvalue idea: Matrix vs Tensor, m={DEMO_M} ---")
    phi = ProductBasis(basis_list)
    d_mat, *_ = spectral_analysis_gedmd_dense(
        Xlist, phi, nev=200, diffusion_const=DIFFUSION_CONST, tol=1e-4, eps_ev=0.0
    )
    d_tt, *_ = tensor_spectral_analysis(Xlist, basis_list, R_TRUNC, 1e-9, nev=40)
    print(f"  Matrix leading eigenvalues: {np.real(d_mat[:4])}")
    print(f"  Tensor leading eigenvalues: {np.real(d_tt[:4])}")

    print(f"\n--- Rank idea: TT rank at tol={DEMO_TOL_RANK:.0e}, m={DEMO_M} ---")
    psi, _ = evaluate_basis(Xlist, basis_list)
    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k):
        return data_tensor.build_core(k + 1)

    U_cores, _, _ = global_svd_tt(core_getter, num_cores=P_DIMS + 1, rmax=R_TRUNC, tol=DEMO_TOL_RANK)
    print(f"  Max TT rank: {get_max_tt_rank(U_cores)}")
