"""
Chignolin -- run_calculation_scratch_vs_incremental_base.py

Base file: the actual idea being tested here, with everything needed to run
it once, made explicit and runnable standalone -- separate from the
"repeat for N_RUNS independent subsamples" orchestration in
run_calculation_scratch_vs_incremental.py.

The idea: for one random subsample (one seed), compute the leading
generator eigenvalues as the TICA embedding dimension grows from 3 to 10,
two different ways:

  "scratch": recompute the whole TT-gEDMD pipeline from dimension 1 up,
    independently, at every dimension.

  "incremental": build the TT basis for dim=3 once, then for each further
    dimension only re-does the *local* SVD steps affected by the newly
    added physical dimension, carrying forward the unaffected earlier
    cores instead of recomputing them.

Both should converge to the same eigenvalues; the interesting content of
the actual paper figure is (a) that they actually agree, and (b) the
wall-clock time difference -- which is exactly what this base file's demo
run shows for one subsample, before the full file repeats it N_RUNS times
for statistics.

Scope note on the incremental path
-----------------------------------
The original notebook's incremental pipeline caches partial per-sample sums
*inside* compute_A_r itself, reused across dimension increments
(`W_prefix_cache` / `save_prefix_depth`) -- an additional optimization layer
that tensor_gedmd's `compute_A_r` doesn't currently implement (it would be a
new capability, not a reuse of something already tested elsewhere in this
package). This script replicates the *other*, primary source of the
incremental speedup -- reusing the TT-SVD's already-computed cores between
dimensions instead of resweeping from dimension 1 each time -- but calls
`compute_A_r` fresh (uncached) at each dimension. The eigenvalues will match
the original faithfully; the measured "incremental" runtime here will be
somewhat conservative (slower) than the original notebook's, since it's
missing that extra caching layer.

run_calculation_scratch_vs_incremental.py imports tica_pool /
run_scratch_one_dim / run_incremental_all_dims from here and wraps them
in a loop over N_RUNS independent seeds to produce
chignolin_scratch_vs_incremental_results.npz (the actual paper figure's
data).

Run this file directly for a quick, single-subsample demo -- one fixed
seed, scratch vs incremental across all DIMS, no repeats, no plot. Confirms
scratch and incremental actually agree on your data before committing to
the full N_RUNS repeats.

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
    python run_calculation_scratch_vs_incremental_base.py
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.linalg import eigh

from tensor_gedmd.algorithms.global_svd import global_svd_tt
from tensor_gedmd.algorithms.mat_vec_prod_direct import compute_A_r
from tensor_gedmd.algorithms.util import _truncate_rank, filter_ev
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
# Physical/pipeline constants -- the same regardless of which seed is
# being tested, so they live here rather than in the repeat-orchestration
# file. DIMS in particular is the fixed sequence of dimensions to sweep
# for EVERY seed, not something that varies per repeat.
# MAX_DIM_AVAILABLE must match whatever dimension build_precomputed_data.py
# fit TICA at (10 by default).
# ----------------------------------------------------------------------
MAX_DIM_AVAILABLE = 10
N_BASIS = 10
SIGMA_RFF = 25
TARGET_M = 6000
NEV = 40
EPS_EV = 0.0

# Truncation held constant across every dimension -- keeps the incremental
# chain from crossing a tol/rank regime boundary (see original notebook's
# note: this is what makes scratch and incremental agree cleanly).
R_TRUNC = 99999
TOL = 1e-12
DIMS = [3, 4, 5, 6, 7, 8, 9, 10]

BASE_SEED = 42


def rff_omega(n: int, sigma: float) -> np.ndarray:
    return (np.arange(1, n + 1) / sigma).reshape(-1, 1)


# ============================================================
# Data loading (precomputed -- no jax/mdtraj/deeptime needed)
# ============================================================

_TICA_POOL: dict = {}


def tica_pool(tica_dim: int):
    """Loads (and caches) the precomputed TICA coordinates and diffusion
    tensor, sliced to tica_dim -- shared across every seed/repeat, since
    the data itself doesn't depend on which subsample is drawn."""
    if tica_dim > MAX_DIM_AVAILABLE:
        raise ValueError(
            f"tica_dim={tica_dim} exceeds MAX_DIM_AVAILABLE={MAX_DIM_AVAILABLE} -- "
            f"re-run build_precomputed_data.py with a higher MAX_DIM if you need this."
        )
    if tica_dim not in _TICA_POOL:
        tica_data_full = np.load(CG_TICA_NPY_PATH, allow_pickle=True)
        diff_full = np.load(CG_DIFF_NPY_PATH, allow_pickle=True)
        tica_data = tica_data_full[:tica_dim, :]
        dmat_full = diff_full[:tica_dim, :tica_dim, :]
        N_total = tica_data.shape[1]
        _TICA_POOL[tica_dim] = (tica_data, dmat_full, N_total)
    return _TICA_POOL[tica_dim]


def get_tica_data(tica_dim: int, seed: int):
    tica_data, dmat_full, N_total = tica_pool(tica_dim)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N_total, size=TARGET_M, replace=False))
    Xlist = tica_data[:, idx]
    dmat = dmat_full[:, :, idx]
    return Xlist, dmat


def whiten_and_eig(Z, reduced_matrix, nev=NEV, eps_ev=EPS_EV):
    r = Z.shape[0]
    mean_z = np.mean(Z, axis=1)
    G1 = np.eye(r) - np.outer(mean_z, mean_z)
    dG, WG = eigh(G1)
    dG, WG = dG[1:], WG[:, 1:]
    Dmh = np.diag(dG ** -0.5)
    R_wh = Dmh @ (WG.T @ reduced_matrix @ WG) @ Dmh
    d_tt, W_tt = eigh(R_wh)
    d_tt, W_tt = filter_ev(d_tt, W_tt, eps2=-eps_ev)
    return d_tt[-nev:][::-1]


# ============================================================
# Scratch (from-scratch) pipeline: one dimension, independent of others.
# ============================================================

def run_scratch_one_dim(tica_dim, seed):
    print(f"\n[SCRATCH] tica_dim={tica_dim}")
    t0 = time.perf_counter()

    Xlist, dmat = get_tica_data(tica_dim, seed)
    p, m = Xlist.shape

    omega = rff_omega(N_BASIS, SIGMA_RFF)
    basis_list = [RandomFourierFeatures(omega=omega) for _ in range(p)]
    psi = [basis_list[i](Xlist[i:i + 1, :]) for i in range(p)]
    dpsi = [basis_list[i].gradient(Xlist[i:i + 1, :])[:, 0, :] for i in range(p)]
    del Xlist

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat)
    del dmat

    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k):
        return data_tensor.build_core(k + 1)

    U_tt, Sigma_svd, V_core = global_svd_tt(core_getter, num_cores=p + 1, rmax=R_TRUNC, tol=TOL)
    U_cores = U_tt.cores
    r = U_cores[-1].shape[2]
    Z = np.sqrt(m) * V_core[:, :, 0]
    del V_core

    A_r = compute_A_r(op=op, U_cores=U_cores, r=r, chunk_size=100, n_workers=None,
                      r_cap=U_cores[-1].shape[2])
    del op, U_cores

    s = np.diag(Sigma_svd)
    reduced = A_r / s[:, None] / s[None, :]
    del A_r, Sigma_svd, s

    ev = whiten_and_eig(Z, reduced)
    del Z, reduced
    gc.collect()

    elapsed = time.perf_counter() - t0
    print(f"  ev[0]={ev[0]:+.6f}  ev[1]={ev[1]:+.6f}  r={r}  elapsed={elapsed:.1f}s")
    return float(ev[0]), float(ev[1]), elapsed


# ============================================================
# Incremental pipeline: builds dim=DIMS[0] once, then grows one physical
# dimension at a time, reusing already-computed TT-SVD cores. See the
# module docstring's "Scope note" for what this does and doesn't replicate
# from the original.
# ============================================================

def _global_svd_tt_with_carries(core_getter, num_cores, rmax=None, tol=0.0):
    """
    Same fast path as tensor_gedmd.algorithms.global_svd.global_svd_tt when
    the final core is the identity (true for Transformed_Data_Tensor_TT),
    but also returns the intermediate SVD carries at every site -- needed to
    extend the TT chain incrementally when a new physical dimension is added
    later, instead of resweeping from site 1. This is a local, verbatim port
    of the original notebook's own global_svd_tt (confirmed against its
    source), which itself returns these carries.

    core_getter is 0-indexed over the *feature* cores only (k = 0, ..., p-1);
    the trailing identity core is never requested.
    """
    p = num_cores - 1
    carry = None
    U_cores = []
    carries = {}

    for j in range(p - 1):
        G = core_getter(j)
        if carry is not None:
            G = np.tensordot(carry, G, axes=(1, 0))
        rp, nj, rn = G.shape
        U, S, Vt = np.linalg.svd(G.reshape(rp * nj, rn), full_matrices=False)
        r_new = _truncate_rank(S, rmax=rmax, tol=tol)
        U_cores.append(U[:, :r_new].reshape(rp, nj, r_new))
        carry = S[:r_new, None] * Vt[:r_new, :]
        carries[j] = carry.copy()

    Gp = core_getter(p - 1)
    if carry is not None:
        Gp = np.tensordot(carry, Gp, axes=(1, 0))
    rp, n_last, r_final = Gp.shape
    U, S, Vt = np.linalg.svd(Gp.reshape(rp * n_last, r_final), full_matrices=False)
    r = _truncate_rank(S, rmax=rmax, tol=tol)
    U_cores.append(U[:, :r].reshape(rp, n_last, r))
    Sigma = np.diag(S[:r])
    V = Vt[:r].reshape(r, r_final, 1)
    return U_cores, Sigma, V, carries


def run_incremental_all_dims(seed):
    print(f"\n########## INCREMENTAL -- seed={seed} ##########")
    inc_ev0: Dict[int, float] = {}
    inc_ev1: Dict[int, float] = {}
    inc_time: Dict[int, float] = {}

    omega_base = rff_omega(N_BASIS, SIGMA_RFF)

    psi: List[np.ndarray] = []
    dpsi: List[np.ndarray] = []
    U_cores: List[np.ndarray] = []
    extend_carry: np.ndarray = None  # fold ready to extend the chain by one more core

    for dim in DIMS:
        print(f"[INCREMENTAL] dim={dim}")
        t0 = time.perf_counter()

        Xlist, dmat = get_tica_data(dim, seed)
        m = Xlist.shape[1]

        if dim == DIMS[0]:
            for i in range(dim):
                basis_i = RandomFourierFeatures(omega=omega_base)
                Xi = Xlist[i:i + 1, :]
                psi.append(basis_i(Xi))
                dpsi.append(basis_i.gradient(Xi)[:, 0, :])
        else:
            basis_new = RandomFourierFeatures(omega=omega_base)
            X_new = Xlist[dim - 1:dim, :]
            psi.append(basis_new(X_new))
            dpsi.append(basis_new.gradient(X_new)[:, 0, :])
        del Xlist

        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat)
        del dmat
        gc.collect()

        data_tensor = Transformed_Data_Tensor_TT(psi=psi)

        def core_getter(k, dt=data_tensor):
            return dt.build_core(k + 1)

        if dim == DIMS[0]:
            U_cores, Sigma_svd, V_core, carries = _global_svd_tt_with_carries(
                core_getter, num_cores=dim + 1, rmax=R_TRUNC, tol=TOL
            )
            r = U_cores[-1].shape[2]
            Z = np.sqrt(m) * V_core[:, :, 0]
            extend_carry = carries[dim - 2]
            del V_core
        else:
            prev_last_core = core_getter(dim - 2)
            G_mid = np.tensordot(extend_carry, prev_last_core, axes=(1, 0))
            rp, n_mid, rn = G_mid.shape
            U_, S_, Vt_ = np.linalg.svd(G_mid.reshape(rp * n_mid, rn), full_matrices=False)
            rr = _truncate_rank(S_, rmax=R_TRUNC, tol=TOL)
            U_mid = U_[:, :rr].reshape(rp, n_mid, rr)
            new_carry = S_[:rr, None] * Vt_[:rr, :]

            new_last_core = core_getter(dim - 1)
            G_last = np.tensordot(new_carry, new_last_core, axes=(1, 0))
            rp2, n_last, rn2 = G_last.shape
            U_last_raw, S_last, Vt_last = np.linalg.svd(
                G_last.reshape(rp2 * n_last, rn2), full_matrices=False
            )
            r = _truncate_rank(S_last, rmax=R_TRUNC, tol=TOL)
            U_last = U_last_raw[:, :r].reshape(rp2, n_last, r)
            Sigma_svd = np.diag(S_last[:r])
            Z = np.sqrt(m) * Vt_last[:r, :]

            U_cores = U_cores[:-1] + [U_mid, U_last]
            extend_carry = new_carry

        data_tensor.clear_cache()

        A_r = compute_A_r(op=op, U_cores=U_cores, r=r, chunk_size=100, n_workers=None,
                          r_cap=U_cores[-1].shape[2])
        del op

        s = np.diag(Sigma_svd)
        reduced = A_r / s[:, None] / s[None, :]
        del A_r, Sigma_svd, s

        ev = whiten_and_eig(Z, reduced)
        del Z, reduced
        gc.collect()

        elapsed = time.perf_counter() - t0
        inc_ev0[dim] = float(ev[0])
        inc_ev1[dim] = float(ev[1])
        inc_time[dim] = elapsed
        print(f"  dim={dim}  ev0={inc_ev0[dim]:+.6f}  ev1={inc_ev1[dim]:+.6f}  "
              f"r={r}  time={elapsed:.1f}s")

    return inc_ev0, inc_ev1, inc_time


# ============================================================
# Demo: one fixed seed, scratch vs incremental across all DIMS
# ============================================================

if __name__ == "__main__":
    DEMO_SEED = BASE_SEED + 1

    print("Pre-loading TICA pools (once per dim) ...")
    for d in DIMS:
        tica_pool(d)

    print(f"\n########## FULL (scratch) -- seed={DEMO_SEED} ##########")
    scratch_ev0, scratch_ev1 = {}, {}
    for d in DIMS:
        e0, e1, t = run_scratch_one_dim(d, DEMO_SEED)
        scratch_ev0[d], scratch_ev1[d] = e0, e1

    inc_ev0, inc_ev1, inc_time = run_incremental_all_dims(DEMO_SEED)

    print(f"\nDemo result -- seed={DEMO_SEED} -- scratch vs incremental:")
    for d in DIMS:
        print(f"  dim={d:2d}: scratch=({scratch_ev0[d]:+.6f},{scratch_ev1[d]:+.6f})  "
              f"incremental=({inc_ev0[d]:+.6f},{inc_ev1[d]:+.6f})")