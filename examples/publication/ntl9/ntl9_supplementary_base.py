"""
NTL9 -- ntl9_supplementary_base.py

Base file: the actual idea being tested here, with everything needed to run
it once, made explicit and runnable standalone -- separate from the
"repeat for N_SUBSAMPLES independent subsamples, swept over dims and
sigmas" orchestration in ntl9_supplementary.py.

The idea: at a given TICA embedding dimension, build an RFF/TT-gEDMD basis
at some bandwidth sigma_omega, truncate the global SVD at some tolerance,
and see what leading eigenvalues and TT rank come out -- exactly the same
idea as chignolin_supplementary_base.py, adapted for NTL9's precomputed
.npy data (no jax/mdtraj needed here).

ntl9_supplementary.py imports load_ntl9_data / prepare_ntl9_subset /
compute_eigs_one_subsample from here and wraps compute_eigs_one_subsample
in loops over SWEEP_TICA_DIMS x N_SUBSAMPLES x SIGMA_RFF_SWEEP to produce
ntl9_supplementary_results.npz (the actual paper figure's data).

Run this file directly for a quick, single-subsample demo -- one fixed
(dim, sigma) combination, one subsample, sweeping the actual SVD_TOLS list
(that sweep is intrinsic to the idea, not part of the repeat
orchestration), no plot.

Requirements
------------
Data files at NTL9_DATA_PATH / NTL9_DIFFUSION_PATH below (see data/README.md).

Usage
-----
    python ntl9_supplementary_base.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

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
# NTL9_DATA_DIR environment variable if your data lives elsewhere -- no
# need to edit this file.
DATA_DIR = Path(os.environ.get(
    "NTL9_DATA_DIR", str(Path(__file__).resolve().parent / "data")
))
NTL9_DATA_PATH = DATA_DIR / "ntl9_tica.npy"
NTL9_DIFFUSION_PATH = DATA_DIR / "ntl9_diff.npy"

# ----------------------------------------------------------------------
# Physical/pipeline constants (verbatim from the source notebook) -- the
# same regardless of which (dim, sigma, subsample) combination is being
# tested, so they live here rather than in the sweep-orchestration file.
# ----------------------------------------------------------------------
SVD_TOLS = [1e-3, 1e-5, 1e-7, 1e-10, 1e-12, 1e-15]
DEFAULT_R_TRUN = 99999
R_TRUN_MAP = {4: 99999, 5: 99999, 6: 99999, 8: 99999}

N_BASIS = 10
NEV = 20
N_EIGS_KEEP = 5  # ev0..ev4 computed and saved (only ev0..ev2 are plotted)

START_IDX = 109267
END_IDX = 128720


def rff_omega(n: int, sigma: float) -> np.ndarray:
    return (np.arange(1, n + 1) / sigma).reshape(-1, 1)


def get_r_trun(tica_dim: int) -> int:
    return R_TRUN_MAP.get(tica_dim, DEFAULT_R_TRUN)


# ============================================================
# Data loading
# ============================================================

def load_ntl9_data():
    tica_data = np.load(NTL9_DATA_PATH, allow_pickle=True)
    diffusion = np.load(NTL9_DIFFUSION_PATH, allow_pickle=True)
    print("NTL9 TICA shape      :", tica_data.shape)
    print("NTL9 diffusion shape :", diffusion.shape)
    return tica_data, diffusion


def prepare_ntl9_subset(tica_data, diffusion, tica_dim, start, end, stride,
                        offset=0, standardize=True):
    """
    Select TICA dimensions/indices, then standardize to unit variance.
    If X_i -> X_i / s_i, the diffusion matrix transforms as
    a_ij -> a_ij / (s_i s_j), so the generator stays consistent.
    """
    Xlist = tica_data[:tica_dim, start + offset:end:stride]
    diff_mat = diffusion[:tica_dim, :tica_dim, start + offset:end:stride]

    if standardize:
        s = Xlist.std(axis=1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        Xlist = Xlist / s
        s_col = s[:, 0]
        diff_mat = diff_mat / (s_col[:, None, None] * s_col[None, :, None])

    return Xlist, diff_mat


# ============================================================
# The actual idea: one (dim, sigma, subsample), swept over SVD_TOLS
# ============================================================

def compute_eigs_one_subsample(Xlist_dim, dmat_dim, sigma_rff, r_trun) -> List[Dict[str, float]]:
    """Full RFF -> TT-SVD -> whitening -> eigenvalue pipeline for a single
    subsample and a single sigma, sweeping all SVD_TOLS."""
    p, m = Xlist_dim.shape

    omega = rff_omega(N_BASIS, sigma_rff)
    basis_list = [RandomFourierFeatures(omega=omega) for _ in range(p)]
    psi = [basis_list[i](Xlist_dim[i:i + 1, :]) for i in range(p)]
    dpsi = [basis_list[i].gradient(Xlist_dim[i:i + 1, :])[:, 0, :] for i in range(p)]

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat_dim)
    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k):
        return data_tensor.build_core(k + 1)

    tol_results = []
    for tol in SVD_TOLS:
        t0 = time.perf_counter()

        U_tt, Sigma_svd, V_core = global_svd_tt(core_getter, num_cores=p + 1, rmax=r_trun, tol=tol)
        U_cores = U_tt.cores
        rank_got = V_core.shape[0]

        Z = np.sqrt(m) * V_core[:, :, 0]
        A_r = compute_A_r(op=op, U_cores=U_cores, r=U_cores[-1].shape[2],
                          chunk_size=100, n_workers=None, r_cap=U_cores[-1].shape[2])
        Sigma_inv = np.linalg.inv(Sigma_svd)
        reduced_matrix = Sigma_inv @ A_r @ Sigma_inv

        mean_z = np.mean(Z, axis=1)
        G1 = np.eye(rank_got) - np.outer(mean_z, mean_z)
        d_G, W_G = eigh(G1)
        d_G, W_G = d_G[1:], W_G[:, 1:]
        valid = d_G > 1e-14
        d_G, W_G = d_G[valid], W_G[:, valid]
        Dmh = np.diag(d_G ** -0.5)
        R_white = Dmh @ (W_G.T @ reduced_matrix @ W_G) @ Dmh

        d_tt, W_tt = eigh(R_white)
        d_tt, W_tt = filter_ev(d_tt, W_tt, eps2=0.0)
        nev_take = min(max(NEV, N_EIGS_KEEP), len(d_tt))
        d_tt = d_tt[-nev_take:][::-1]

        eig_values = [
            float(np.real(d_tt[k])) if k < len(d_tt) else float("nan")
            for k in range(N_EIGS_KEEP)
        ]

        elapsed = time.perf_counter() - t0
        ev_print = "  ".join(f"{v:>+13.6f}" for v in eig_values)
        print(f"      tol={tol:>10.2e}  rank={rank_got:>7d}  {ev_print}  ({elapsed:.2f}s)")

        result = {"tol": tol, "rank": rank_got}
        for k, val in enumerate(eig_values):
            result[f"ev{k}"] = val
        tol_results.append(result)

    data_tensor.clear_cache()
    return tol_results


# ============================================================
# Demo: one fixed (dim, sigma), one subsample, sweeping SVD_TOLS
# ============================================================

if __name__ == "__main__":
    DEMO_TICA_DIM = 3
    DEMO_SIGMA_RFF = 7.0
    DEMO_SUBSAMPLE_M = 6485
    DEMO_SEED = 43

    tica_data, diffusion = load_ntl9_data()
    X_pool, D_pool = prepare_ntl9_subset(
        tica_data, diffusion, tica_dim=DEMO_TICA_DIM,
        start=START_IDX, end=END_IDX, stride=1, offset=0, standardize=True,
    )
    N_pool = X_pool.shape[1]

    rng = np.random.default_rng(DEMO_SEED)
    idx = np.sort(rng.choice(N_pool, size=DEMO_SUBSAMPLE_M, replace=False))
    Xs, Ds = X_pool[:, idx], D_pool[:, :, idx]

    r_trun = get_r_trun(DEMO_TICA_DIM)
    print(f"\nDemo -- dim={DEMO_TICA_DIM}, sigma={DEMO_SIGMA_RFF}, m={Xs.shape[1]}, r_trun={r_trun}")
    results = compute_eigs_one_subsample(Xs, Ds, DEMO_SIGMA_RFF, r_trun)

    print(f"\nDemo result -- dim={DEMO_TICA_DIM}, sigma={DEMO_SIGMA_RFF}:")
    for row in results:
        print(f"  tol={row['tol']:.0e}: rank={row['rank']}  ev0={row['ev0']:+.6f}  ev1={row['ev1']:+.6f}")
