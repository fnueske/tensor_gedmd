"""
Chignolin -- chignolin_supplementary_base.py

Base file: the actual idea being tested here, with everything needed to
run it once, made explicit and runnable standalone -- separate from the
"repeat it 10 times per (dim, sigma, tol) combination for error bars"
orchestration in chignolin_supplementary.py.

The idea: at a given TICA embedding dimension, build an RFF/TT-gEDMD basis
at some bandwidth sigma_omega, truncate the global SVD at some tolerance,
and see what leading eigenvalues (kappa_1, kappa_2) and TT rank come out.
chignolin_supplementary.py imports load_raw_data / fit_tica_and_diffusion /
subsample_one_repeat / solve_one from here, and wraps solve_one in a loop
over TICA_DIMS x N_REPEATS x SIGMA_LIST x SVD_TOLS to produce
chignolin_supplementary_results.npz (the actual paper figure's data).

Run this file directly for a quick, single-experiment demo -- one fixed
(dim, sigma, tol) combination, no repeats, no sweep, no plot. Confirms the
pipeline runs correctly on your data before committing to the full sweep.

Diffusion tensor: estimated from the underlying atomic coordinates via a
Jacobian of pairwise CA-CA distances (jax), projected onto the TICA
directions -- same real, data-dependent computation as
chignolin/run_calculation_threshold_and_dim_sweep.py, kept verbatim since
it has no tensor_gedmd equivalent.

Requirements
------------
- `deeptime`, `mdtraj`, `jax` importable.
- Data files at CG_PICKLE_PATH / CG_ATOMIC_NUMBERS_PATH / PDB_PATH below
  (see data/README.md).

Usage
-----
    python chignolin_supplementary_base.py
"""

from __future__ import annotations

import gc
import itertools
import os
import pickle
import sys
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
CG_PICKLE_PATH = DATA_DIR / "cln_tica.pkl"
CG_ATOMIC_NUMBERS_PATH = DATA_DIR / "cln_atomic.pkl"
PDB_PATH = DATA_DIR / "chi_vac.pdb"

# ----------------------------------------------------------------------
# Physical/pipeline constants (verbatim from the source notebooks) -- these
# are the same regardless of which (dim, sigma, tol) combination is being
# tested, so they live here rather than in the sweep-orchestration file.
# ----------------------------------------------------------------------
R_TRUN_MAP = {3: 300, 4: 500, 6: 2000, 8: 99999, 10: 99999}
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

    with open(CG_ATOMIC_NUMBERS_PATH, "rb") as f:
        x_atomic = np.concatenate(
            [d.xyz[:, CA_atoms] for d in pickle.load(f)["x"]], axis=0
        )

    CA_comb = np.array(list(itertools.combinations(np.arange(len(CA_atoms)), 2)))

    def distances(t):
        return jnp.linalg.norm(t[CA_comb[:, 1]] - t[CA_comb[:, 0]], axis=1)

    def jac(x):
        return jacrev(distances, argnums=0)(x)

    print("  Jacobian (vmap) ...", end="", flush=True)
    t0 = time.perf_counter()
    diff_full = vmap(jac, in_axes=(0,), out_axes=0)(x_atomic).transpose(1, 2, 3, 0)
    print(f" done ({time.perf_counter() - t0:.1f}s)  shape: {diff_full.shape}")
    del x_atomic
    gc.collect()
    return trajectories, np.asarray(diff_full)


def fit_tica_and_diffusion(trajectories, diff_full, tica_dim: int):
    """
    The expensive, per-dimension-only-once part: fit TICA and project the
    diffusion tensor onto the TICA directions for the full data pool.
    Call this ONCE per tica_dim, then subsample_one_repeat(...) as many
    times as you like against the same fit (cheap).
    """
    from deeptime import decomposition

    t0 = time.perf_counter()
    tica = decomposition.TICA(lagtime=LAGTIME, dim=tica_dim)
    tica_model = tica.fit(trajectories).fetch_model()
    tica_data = tica_model.transform(trajectories).reshape(-1, tica_dim).T
    eigs_sv = tica_model.singular_values
    eigvec = tica_model.singular_vectors_left[:, np.argsort(eigs_sv)[::-1]]
    N_total = tica_data.shape[1]
    print(f"  TICA done ({time.perf_counter() - t0:.2f}s)  pool={N_total} frames")

    dtr = np.einsum("ijlr,ik->kjlr", diff_full, eigvec[:, :tica_dim])
    r_trun = R_TRUN_MAP[tica_dim]
    return tica_data, dtr, N_total, r_trun


def subsample_one_repeat(
    tica_data: np.ndarray, dtr: np.ndarray, N_total: int, seed: int,
    target_m: int = TARGET_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The cheap, per-repeat part: draw one random subsample of target_m
    frames from the already-fit TICA pool, and build its diffusion tensor.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N_total, size=target_m, replace=False))
    Xlist = tica_data[:, idx]

    dtr_idx = dtr[:, :, :, idx]
    dmat = (2.0 / BETA / M_MASS / GAMMA) * np.einsum("gikl,hikl->ghl", dtr_idx, dtr_idx)
    del dtr_idx
    gc.collect()
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

    trajectories, diff_full = load_raw_data()
    tica_data, dtr, N_total, r_trun = fit_tica_and_diffusion(trajectories, diff_full, DEMO_TICA_DIM)
    Xlist, dmat = subsample_one_repeat(tica_data, dtr, N_total, DEMO_SEED)

    res = solve_one(Xlist, dmat, DEMO_SIGMA_RFF, r_trun, DEMO_TOL)

    print(f"\nDemo result -- dim={DEMO_TICA_DIM}, sigma_omega={DEMO_SIGMA_RFF}, tol={DEMO_TOL:.0e}:")
    print(f"  TT rank = {res['rank']}")
    print(f"  kappa_1 = {res['ev0']:+.6f}")
    print(f"  kappa_2 = {res['ev1']:+.6f}")
