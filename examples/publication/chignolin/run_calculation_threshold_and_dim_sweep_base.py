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

run_calculation_threshold_and_dim_sweep.py imports load_raw_data /
tica_and_diffusion / run_subsample from here and wraps run_subsample in a
loop over dims x N_RUNS x (per-dim sigma/r_trun/tol-list configurations) to
produce chignolin_results.npz (the actual paper figure's data, both the
threshold-sweep and dimension-sweep panels).

Run this file directly for a quick, single-experiment demo -- one fixed
dimension, one fixed subsample (run_id), a small representative list of
tolerances, no repeats, no plot. Confirms the pipeline runs correctly on
your data before committing to the full sweep.

Requirements
------------
- `deeptime` (TICA), `mdtraj` (topology/CA selection), `jax` (Jacobian of
  pairwise distances) all importable.
- Data files at CG_PICKLE_PATH / CG_ATOMIC_NUMBERS_PATH / PDB_PATH below
  (see data/README.md).

Usage
-----
    python run_calculation_threshold_and_dim_sweep_base.py
"""

from __future__ import annotations

import gc
import itertools
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
CG_PICKLE_PATH = DATA_DIR / "cln_tica.pkl"
CG_ATOMIC_NUMBERS_PATH = DATA_DIR / "cln_atomic.pkl"
PDB_PATH = DATA_DIR / "chi_vac.pdb"

# ----------------------------------------------------------------------
# Physical/pipeline constants (verbatim from the source notebooks) -- these
# are the same regardless of which dim/sigma/tol combination is being
# tested, so they live here rather than in the sweep-orchestration file.
# ----------------------------------------------------------------------
BASE_SEED = 42
LAGTIME = 15
TARGET_M = 6000
N_BASIS = 10
NEV = 20

M_MASS = 12.0
T_K = 300.0
GAMMA = 0.1
K_B = 8.314462e-3
BETA = 1.0 / (K_B * T_K)


def rff_omega(n: int, sigma: float) -> np.ndarray:
    """Deterministic RFF frequency vector, shape (n, 1)."""
    return (np.arange(1, n + 1) / sigma).reshape(-1, 1)


# ============================================================
# Raw data + diffusion tensor
# ============================================================

def load_raw_data():
    import mdtraj as md
    import jax.numpy as jnp
    from jax import jacrev, vmap

    print("Loading raw data ...")
    trajectories = []
    with open(CG_PICKLE_PATH, "rb") as f:
        for traj in pickle.load(f)["x"]:
            trajectories.append(traj)
    print(f"  {len(trajectories)} trajectories (first: {trajectories[0].shape})")

    top = md.load_topology(PDB_PATH)
    CA_atoms = top.select("name CA")
    n_ca = len(CA_atoms)

    with open(CG_ATOMIC_NUMBERS_PATH, "rb") as f:
        x_atomic = np.concatenate(
            [d.xyz[:, CA_atoms] for d in pickle.load(f)["x"]], axis=0
        )
    print(f"  x_atomic: {x_atomic.shape}")

    CA_comb = np.array(list(itertools.combinations(np.arange(n_ca), 2)))

    def distances(t):
        return jnp.linalg.norm(t[CA_comb[:, 1]] - t[CA_comb[:, 0]], axis=1)

    def jac(x):
        return jacrev(distances, argnums=0)(x)

    print("  Jacobian (vmap) ...", end="", flush=True)
    t0 = time.perf_counter()
    diff_full = vmap(jac, in_axes=(0,), out_axes=0)(x_atomic).transpose(1, 2, 3, 0)
    print(f" done ({time.perf_counter() - t0:.1f} s)  shape: {diff_full.shape}")
    del x_atomic
    gc.collect()
    return trajectories, np.asarray(diff_full)


def tica_and_diffusion(trajectories, diff_full, dim: int):
    """Run TICA at a given dimension and project the atomic diffusion
    tensor onto the TICA directions -- both needed once per dimension."""
    from deeptime import decomposition

    t0 = time.perf_counter()
    tica = decomposition.TICA(lagtime=LAGTIME, dim=dim)
    tica_model = tica.fit(trajectories).fetch_model()
    tica_data = tica_model.transform(trajectories).reshape(-1, dim).T
    eigs_sv = tica_model.singular_values
    eigvec = tica_model.singular_vectors_left[:, np.argsort(eigs_sv)[::-1]]
    N_total = tica_data.shape[1]
    print(f"  TICA dim={dim} done ({time.perf_counter() - t0:.2f}s)  pool={N_total} frames")

    t0 = time.perf_counter()
    dtr = np.einsum("ijlr,ik->kjlr", diff_full, eigvec[:, :dim])
    dmat_all = (2.0 / BETA / M_MASS / GAMMA) * np.einsum("gikl,hikl->ghl", dtr, dtr)
    print(f"  Diffusion matrix done ({time.perf_counter() - t0:.2f}s)  dmat_all: {dmat_all.shape}")
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

    trajectories, diff_full = load_raw_data()
    tica_data, dmat_all, N_total = tica_and_diffusion(trajectories, diff_full, DEMO_DIM)

    res = run_subsample(DEMO_DIM, DEMO_RUN_ID, tica_data, dmat_all, N_total,
                        DEMO_SIGMA, DEMO_RTRUN, DEMO_TOLS)

    print(f"\nDemo result -- dim={DEMO_DIM}, sigma={DEMO_SIGMA}, run_id={DEMO_RUN_ID}:")
    for tol, (k1, k2, rank) in res.items():
        print(f"  tol={tol:.0e}: rank={rank}  kappa_1={k1:+.6f}  kappa_2={k2:+.6f}")
