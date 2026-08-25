"""
Chignolin -- chignolin_supplementary_base.py

Base file: the actual idea being tested here, with everything needed to
run it once, made explicit and runnable standalone -- separate from the
"repeat it 10 times per (dim, sigma, tol) combination for error bars"
orchestration in chignolin_supplementary.py.

The idea: at a given TICA embedding dimension, build an RFF/TT-gEDMD basis
at some bandwidth sigma_omega, truncate the global SVD at some tolerance,
and see what leading eigenvalues (kappa_1, kappa_2) and TT rank come out.
chignolin_supplementary.py imports fit_tica_and_diffusion /
subsample_one_repeat / solve_one from here, and wraps solve_one in a loop
over TICA_DIMS x N_REPEATS x SIGMA_LIST x SVD_TOLS to produce
chignolin_supplementary_results.npz (the actual paper figure's data).

Run this file directly for a quick, single-experiment demo -- one fixed
(dim, sigma, tol) combination, no repeats, no sweep, no plot. Confirms the
pipeline runs correctly on your data before committing to the full sweep.

Data: this loads the precomputed cln_tica.npy / cln_diff.npy (TICA fit
once at dim=10, diffusion tensor projected onto all 10 directions --
validated to give bit-for-bit identical results to fitting/projecting
separately at a smaller dimension and slicing). If you only have the raw
data (cln_tica.pkl, cln_atomic.pkl, chi_vac.pdb -- the last of which is
large, a real jax Jacobian computation), run build_precomputed_data.py
once first to generate these two files; every calculation script here
uses the precomputed files directly and never touches the raw data or
needs jax/mdtraj/deeptime.

Requirements
------------
Just `numpy`/`scipy` plus `tensor_gedmd` itself -- no jax/mdtraj/deeptime
needed, since TICA and the diffusion tensor are already precomputed. See
data/README.md.

Usage
-----
    python chignolin_supplementary_base.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import eigh

from tensor_gedmd.algorithms.global_svd import global_svd_tt
from tensor_gedmd.algorithms.mat_vec_prod_direct import compute_A_r
from tensor_gedmd.algorithms.util import filter_ev
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures
from tensor_gedmd.reps.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.reps.transformed_data_tensor import Transformed_Data_Tensor_TT

# Default: this case study's own data/ folder (see data/README.md for the
# Zenodo download link and expected filenames). Override with the
# CHIGNOLIN_DATA_DIR environment variable if your data lives elsewhere --
# no need to edit this file.
DATA_DIR = Path(os.environ.get(
    "CHIGNOLIN_DATA_DIR", str(Path(__file__).resolve().parent / "data")
))
CG_TICA_NPY_PATH = Path(os.environ.get("CHIGNOLIN_TICA_NPY", str(DATA_DIR / "cln_tica.npy")))
CG_DIFF_NPY_PATH = Path(os.environ.get("CHIGNOLIN_DIFF_NPY", str(DATA_DIR / "cln_diff.npy")))

# ----------------------------------------------------------------------
# Physical/pipeline constants -- these are the same regardless of which
# (dim, sigma, tol) combination is being tested, so they live here rather
# than in the sweep-orchestration file. MAX_DIM_AVAILABLE must match
# whatever dimension build_precomputed_data.py fit TICA at (10 by default).
# ----------------------------------------------------------------------
MAX_DIM_AVAILABLE = 10
R_TRUN_MAP = {3: 300, 4: 500, 6: 2000, 8: 99999, 10: 99999}
TARGET_M = 6000
N_BASIS = 10
NEV = 20


def rff_omega(n: int, sigma: float) -> np.ndarray:
    return (np.arange(1, n + 1) / sigma).reshape(-1, 1)


# ============================================================
# Data loading (precomputed -- no jax/mdtraj/deeptime needed)
# ============================================================

def fit_tica_and_diffusion(tica_dim: int):
    """
    Load the precomputed TICA coordinates and diffusion tensor, sliced to
    tica_dim. Call this ONCE per tica_dim, then subsample_one_repeat(...)
    as many times as you like against the same pool (cheap).
    """
    if tica_dim > MAX_DIM_AVAILABLE:
        raise ValueError(
            f"tica_dim={tica_dim} exceeds MAX_DIM_AVAILABLE={MAX_DIM_AVAILABLE} -- "
            f"re-run build_precomputed_data.py with a higher MAX_DIM if you need this."
        )

    tica_data_full = np.load(CG_TICA_NPY_PATH, allow_pickle=True)
    diff_full = np.load(CG_DIFF_NPY_PATH, allow_pickle=True)

    tica_data = tica_data_full[:tica_dim, :]
    dmat_full = diff_full[:tica_dim, :tica_dim, :]
    N_total = tica_data.shape[1]
    r_trun = R_TRUN_MAP[tica_dim]

    print(f"  dim={tica_dim} loaded  pool={N_total} frames")
    return tica_data, dmat_full, N_total, r_trun


def subsample_one_repeat(
    tica_data: np.ndarray, dmat_full: np.ndarray, N_total: int, seed: int,
    target_m: int = TARGET_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The cheap, per-repeat part: draw one random subsample of target_m
    frames from the already-loaded TICA pool.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N_total, size=target_m, replace=False))
    Xlist = tica_data[:, idx]
    dmat = dmat_full[:, :, idx]
    return Xlist, dmat


# ============================================================
# The actual idea: one (dim, sigma, tol) solve
# ============================================================

def solve_one(Xlist, dmat, sigma_rff, r_trun, tol) -> Dict[str, float]:
    p, m = Xlist.shape
    omega = rff_omega(N_BASIS, sigma_rff)
    basis_list = [RandomFourierFeatures(omega=omega) for _ in range(p)]
    psi = [basis_list[i](Xlist[i:i + 1, :]) for i in range(p)]
    dpsi = [basis_list[i].gradient(Xlist[i:i + 1, :])[:, 0, :] for i in range(p)]

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat)
    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k):
        return data_tensor.build_core(k + 1)

    U_tt, Sigma_svd, V_core = global_svd_tt(core_getter, num_cores=p + 1, rmax=r_trun, tol=tol)
    U_cores = U_tt.cores
    rank_got = V_core.shape[0]

    Z = np.sqrt(m) * V_core[:, :, 0]
    A_r = compute_A_r(op=op, U_cores=U_cores, r=U_cores[-1].shape[2],
                      chunk_size=100, n_workers=None, r_cap=U_cores[-1].shape[2])
    reduced_matrix = np.linalg.inv(Sigma_svd) @ A_r @ np.linalg.inv(Sigma_svd)

    mean_z = np.mean(Z, axis=1)
    G1 = np.eye(rank_got) - np.outer(mean_z, mean_z)
    d_G, W_G = eigh(G1)
    d_G, W_G = d_G[1:], W_G[:, 1:]
    Dmh = np.diag(d_G ** -0.5)
    R_white = Dmh @ (W_G.T @ reduced_matrix @ W_G) @ Dmh

    d_tt, W_tt = eigh(R_white)
    d_tt, W_tt = filter_ev(d_tt, W_tt, eps2=0.0)
    nev_take = min(NEV, len(d_tt))
    d_tt = d_tt[-nev_take:][::-1]

    ev0 = float(d_tt[0].real) if len(d_tt) > 0 else float("nan")
    ev1 = float(d_tt[1].real) if len(d_tt) > 1 else float("nan")

    data_tensor.clear_cache()
    return {"rank": rank_got, "ev0": ev0, "ev1": ev1}


# ============================================================
# Demo: one fixed (dim, sigma, tol) combination, no repeats, no sweep
# ============================================================

if __name__ == "__main__":
    DEMO_TICA_DIM = 3
    DEMO_SIGMA_RFF = 7
    DEMO_TOL = 1e-5
    DEMO_SEED = 42

    tica_data, dmat_full, N_total, r_trun = fit_tica_and_diffusion(DEMO_TICA_DIM)
    Xlist, dmat = subsample_one_repeat(tica_data, dmat_full, N_total, DEMO_SEED)

    res = solve_one(Xlist, dmat, DEMO_SIGMA_RFF, r_trun, DEMO_TOL)

    print(f"\nDemo result -- dim={DEMO_TICA_DIM}, sigma_omega={DEMO_SIGMA_RFF}, tol={DEMO_TOL:.0e}:")
    print(f"  TT rank = {res['rank']}")
    print(f"  kappa_1 = {res['ev0']:+.6f}")
    print(f"  kappa_2 = {res['ev1']:+.6f}")