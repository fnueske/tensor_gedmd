"""
Chignolin -- run_calculation_threshold_and_dim_sweep_base.py

Base file: the actual idea being tested here, with everything needed to run
it once, made explicit and runnable standalone -- separate from the "sweep
TICA dimension and SVD tolerance, repeat N_RUNS times for error bars"
orchestration in run_calculation_threshold_and_dim_sweep.py.

The idea: at a given TICA embedding dimension, fit TICA and project the
diffusion tensor once (expensive), then for a given random subsample
(cheap, repeatable with different seeds), build an RFF/TT-gEDMD basis at
some bandwidth and see what leading eigenvalues (kappa_1, kappa_2) and TT
rank come out, across a list of SVD truncation tolerances.

run_calculation_threshold_and_dim_sweep.py imports load_dim_pool /
run_subsample from here and wraps run_subsample in a loop over dims x
N_RUNS x (per-dim sigma/r_trun/tol-list configurations) to produce
chignolin_results.npz (the actual paper figure's data, both the
threshold-sweep and dimension-sweep panels).

Run this file directly for a quick, single-experiment demo -- one fixed
dimension, one fixed subsample (run_id), a small representative list of
tolerances, no repeats, no plot. Confirms the pipeline runs correctly on
your data before committing to the full sweep.

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
data/README.md for the precomputed files (cln_tica.npy / cln_diff.npy) or
the raw data if you'd rather regenerate them yourself.

Usage
-----
    python run_calculation_threshold_and_dim_sweep_base.py
"""

from __future__ import annotations

import gc
import os
import time
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
# dim/sigma/tol combination is being tested, so they live here rather than
# in the sweep-orchestration file. MAX_DIM_AVAILABLE must match whatever
# dimension build_precomputed_data.py fit TICA at (10 by default).
# ----------------------------------------------------------------------
MAX_DIM_AVAILABLE = 10
BASE_SEED = 42
TARGET_M = 6000
N_BASIS = 10
NEV = 20


def rff_omega(n: int, sigma: float) -> np.ndarray:
    """Deterministic RFF frequency vector, shape (n, 1)."""
    return (np.arange(1, n + 1) / sigma).reshape(-1, 1)


# ============================================================
# Data loading (precomputed -- no jax/mdtraj/deeptime needed)
# ============================================================

def load_dim_pool(dim: int):
    """
    Load the precomputed TICA coordinates and diffusion tensor, sliced to
    the requested dimension (dim <= MAX_DIM_AVAILABLE). Returns the full
    pool (not yet subsampled) -- run_subsample() draws a random subsample
    from this pool per (run_id, tol-list) call.
    """
    if dim > MAX_DIM_AVAILABLE:
        raise ValueError(
            f"dim={dim} exceeds MAX_DIM_AVAILABLE={MAX_DIM_AVAILABLE} -- "
            f"re-run build_precomputed_data.py with a higher MAX_DIM if you need this."
        )

    t0 = time.perf_counter()
    tica_data_full = np.load(CG_TICA_NPY_PATH, allow_pickle=True)
    diff_full = np.load(CG_DIFF_NPY_PATH, allow_pickle=True)

    tica_data = tica_data_full[:dim, :]
    dmat_all = diff_full[:dim, :dim, :]
    N_total = tica_data.shape[1]

    print(f"  dim={dim} loaded ({time.perf_counter() - t0:.2f}s)  pool={N_total} frames  "
          f"dmat_all: {dmat_all.shape}")
    assert N_total > TARGET_M, f"Pool {N_total} <= TARGET_M {TARGET_M}"

    return tica_data, dmat_all, N_total


# ============================================================
# The actual idea: one subsample (run_id), swept over a list of tols
# ============================================================

def run_subsample(dim, run_id, tica_data, dmat_all, N_total,
                   sigma_rff, r_trun, tols) -> Dict[float, Tuple[float, float, int]]:
    """Returns {tol: (kappa_1, kappa_2, retained_rank)} for each tol in tols."""
    seed = BASE_SEED + run_id
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N_total, size=TARGET_M, replace=False))
    Xlist = tica_data[:, idx]
    dmat = dmat_all[:, :, idx]
    p, m = Xlist.shape
    del idx, rng

    basis_list = [RandomFourierFeatures(omega=rff_omega(N_BASIS, sigma_rff)) for _ in range(p)]
    psi = [basis_list[i](Xlist[i:i + 1, :]) for i in range(p)]
    dpsi = [basis_list[i].gradient(Xlist[i:i + 1, :])[:, 0, :] for i in range(p)]
    del Xlist

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat)
    data_tensor = Transformed_Data_Tensor_TT(psi=psi)
    del dmat

    def core_getter(k):
        return data_tensor.build_core(k + 1)

    out: Dict[float, Tuple[float, float, int]] = {}
    for tol in tols:
        t0 = time.perf_counter()
        U_tt, Sigma_svd, V_core = global_svd_tt(core_getter, num_cores=p + 1, rmax=r_trun, tol=tol)
        U_cores = U_tt.cores
        rank_got = V_core.shape[0]
        Z = np.sqrt(m) * V_core[:, :, 0]
        del V_core

        A_r = compute_A_r(op=op, U_cores=U_cores, r=U_cores[-1].shape[2],
                          chunk_size=100, n_workers=None, r_cap=U_cores[-1].shape[2])
        del U_cores
        reduced_matrix = np.linalg.inv(Sigma_svd) @ A_r @ np.linalg.inv(Sigma_svd)
        del A_r, Sigma_svd

        mean_z = np.mean(Z, axis=1)
        G1 = np.eye(rank_got) - np.outer(mean_z, mean_z)
        del mean_z
        d_G, W_G = eigh(G1)
        del G1
        d_G, W_G = d_G[1:], W_G[:, 1:]
        Dmh = np.diag(d_G ** -0.5)
        del d_G
        R_white = Dmh @ (W_G.T @ reduced_matrix @ W_G) @ Dmh
        del Dmh, W_G, reduced_matrix, Z

        d_tt, W_tt = eigh(R_white)
        del R_white
        d_tt, W_tt = filter_ev(d_tt, W_tt, eps2=0.0)
        del W_tt
        nev_take = min(NEV, len(d_tt))
        d_tt = d_tt[-nev_take:][::-1]

        k1 = float(d_tt[0].real) if len(d_tt) > 0 else float("nan")
        k2 = float(d_tt[1].real) if len(d_tt) > 1 else float("nan")
        del d_tt
        gc.collect()

        out[tol] = (k1, k2, rank_got)
        print(f"    dim={dim:2d} run={run_id:02d} sigma={sigma_rff:<4} tol={tol:.0e} "
              f"r={rank_got:5d}  k1={k1:+.6f} k2={k2:+.6f} ({time.perf_counter() - t0:.1f}s)",
              flush=True)

    data_tensor.clear_cache()
    del op, data_tensor, psi, dpsi
    gc.collect()
    return out


# ============================================================
# Demo: one fixed dim/sigma/r_trun/run_id, small representative tol list
# ============================================================

if __name__ == "__main__":
    DEMO_DIM = 3
    DEMO_SIGMA = 25.0
    DEMO_RTRUN = 99999
    DEMO_RUN_ID = 1
    DEMO_TOLS = [1e-6, 1e-10, 1e-14]

    tica_data, dmat_all, N_total = load_dim_pool(DEMO_DIM)

    res = run_subsample(DEMO_DIM, DEMO_RUN_ID, tica_data, dmat_all, N_total,
                        DEMO_SIGMA, DEMO_RTRUN, DEMO_TOLS)

    print(f"\nDemo result -- dim={DEMO_DIM}, sigma={DEMO_SIGMA}, run_id={DEMO_RUN_ID}:")
    for tol, (k1, k2, rank) in res.items():
        print(f"  tol={tol:.0e}: rank={rank}  kappa_1={k1:+.6f}  kappa_2={k2:+.6f}")