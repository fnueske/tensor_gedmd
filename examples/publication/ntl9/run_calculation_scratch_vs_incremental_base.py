"""
NTL9 -- run_calculation_scratch_vs_incremental_base.py

Base file: the actual idea being tested here, with everything needed to run
it once, made explicit and runnable standalone -- separate from the
"repeat for N_RUNS independent subsamples" orchestration in
run_calculation_scratch_vs_incremental.py.

Same idea as chignolin/run_calculation_scratch_vs_incremental_base.py: for
one random subsample (one seed), compute the leading generator eigenvalues
two ways ("scratch" vs "incremental") as the TICA embedding dimension
grows, and see that they agree. Two things are specific to NTL9:

  - Data comes from precomputed .npy files (TICA + diffusion tensor), not a
    live jax/mdtraj computation -- much simpler loading.
  - DIMS = [3, 4, 5, 6, 8] is NOT contiguous (skips 7), so the incremental
    path needs to "bridge" through dim 7 -- extend the TT-SVD chain by one
    site without evaluating eigenvalues there -- before continuing to dim 8.
  - Tracks the top 4 eigenvalues (kappa_1..kappa_4), not 2.

See chignolin/run_calculation_scratch_vs_incremental_base.py's module
docstring for the same scope note: this reuses the TT-SVD's cores
incrementally (the primary structural idea, validated there to match
"scratch" exactly) but calls compute_A_r fresh at each dimension rather
than also caching compute_A_r's own per-sample partial sums across
dimensions.

run_calculation_scratch_vs_incremental.py imports load_pool / make_get_dim
/ run_scratch_one_dim / run_incremental_all_dims from here and repeats them
for N_RUNS independent seeds to produce
ntl9_scratch_vs_incremental_results.npz.

Run this file directly for a quick, single-subsample demo -- one fixed
seed, scratch vs incremental across all DIMS (with bridging through dim
7), no repeats, no plot.

Requirements
------------
Data files at NTL9_DATA_PATH / NTL9_DIFFUSION_PATH below (see data/README.md).

Usage
-----
    python run_calculation_scratch_vs_incremental_base.py
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.linalg import eigh

from tensor_gedmd.algorithms.mat_vec_prod_direct import compute_A_r
from tensor_gedmd.algorithms.util import _truncate_rank
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
# same regardless of which seed is being tested, so they live here rather
# than in the repeat-orchestration file. DIMS in particular is the fixed
# sequence of dimensions to sweep for EVERY seed, not something that varies
# per repeat.
# ----------------------------------------------------------------------
N_BASIS = 10
SIGMA_RFF = 25
NEV = 40
EPS_EV = 0.0
N_SHOW = 4  # track kappa_1..kappa_4

R_TRUNC = 99999
TOL = 1e-12
DIMS = [3, 4, 5, 6, 8]  # non-contiguous: dim 7 is a "bridge" step only

BASE_SEED = 42
M_SUB = 6485
WINDOW_START = 109267
WINDOW_END = 128720


def rff_omega(n: int, sigma: float) -> np.ndarray:
    return (np.arange(1, n + 1) / sigma).reshape(-1, 1)


# ============================================================
# Data loading (precomputed .npy -- no jax/mdtraj needed for NTL9)
# ============================================================

def load_pool():
    tica = np.load(NTL9_DATA_PATH, allow_pickle=True)
    diff = np.load(NTL9_DIFFUSION_PATH, allow_pickle=True)
    print("NTL9 TICA shape:", tica.shape, " diffusion shape:", diff.shape)

    Xpool = np.asarray(tica[:, WINDOW_START:WINDOW_END], dtype=float)
    Dpool = np.asarray(diff[:, :, WINDOW_START:WINDOW_END], dtype=float)
    N_pool = Xpool.shape[1]

    scale = Xpool.std(axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    Xpool = Xpool / scale
    sc = scale[:, 0]
    Dpool = Dpool / (sc[:, None, None] * sc[None, :, None])

    print(f"pool = {N_pool} frames;  M_SUB = {M_SUB}")
    if M_SUB > N_pool:
        raise ValueError(f"M_SUB={M_SUB} exceeds pool size {N_pool}")
    return Xpool, Dpool, N_pool


def make_get_dim(Xpool, Dpool, idx):
    def get_dim(dim):
        X = Xpool[:dim, idx].copy()
        D = Dpool[:dim, :dim, idx].copy()
        return X, D
    return get_dim


def mean_free_whiten_and_eig(Z, reduced, nev=NEV, eps=EPS_EV, gram_tol=1e-12):
    r, m = Z.shape
    mean_basis = np.mean(Z, axis=1)
    G1 = np.eye(r) - np.outer(mean_basis, mean_basis)
    G1 = 0.5 * (G1 + G1.T)

    d_G, W_G = eigh(G1)
    keep = d_G > gram_tol
    if not np.any(keep):
        raise ValueError("Mean-free Gram matrix has no positive directions after dropping the mean mode.")
    d_G, W_G = d_G[keep], W_G[:, keep]
    Dmh = np.diag(d_G ** -0.5)

    R_white = Dmh @ W_G.T @ reduced @ W_G @ Dmh
    R_white = 0.5 * (R_white + R_white.T)

    d_tt, W_tt = eigh(R_white)
    if eps is not None:
        mask = np.real(d_tt) < -eps
        d_tt = d_tt[mask]

    order = np.argsort(np.real(d_tt))[-nev:][::-1]
    return d_tt[order]


# ============================================================
# Scratch pipeline (one dimension, independent)
# ============================================================

def run_scratch_one_dim(get_dim, tica_dim, m):
    Xlist, dmat = get_dim(tica_dim)
    p = Xlist.shape[0]

    omega = rff_omega(N_BASIS, SIGMA_RFF)
    psi, dpsi = [], []
    for i in range(p):
        basis_i = RandomFourierFeatures(omega=omega)
        Xi = Xlist[i:i + 1, :]
        psi.append(basis_i(Xi))
        dpsi.append(basis_i.gradient(Xi)[:, 0, :])
    del omega, Xlist

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat)
    del dmat

    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k, dt=data_tensor):
        return dt.build_core(k + 1)

    U_cores, Sigma_svd, V_core, _ = _global_svd_tt_with_carries(
        core_getter, num_cores=p + 1, rmax=R_TRUNC, tol=TOL
    )
    Z = np.sqrt(m) * V_core[:, :, 0]
    r = U_cores[-1].shape[2]
    del V_core

    A_r = compute_A_r(op=op, U_cores=U_cores, r=r, chunk_size=100, n_workers=None, r_cap=r)
    del op, U_cores

    s = np.diag(Sigma_svd)
    reduced = A_r / s[:, None] / s[None, :]
    del A_r, Sigma_svd, s

    ev = mean_free_whiten_and_eig(Z, reduced, nev=NEV, eps=EPS_EV)
    del Z, reduced
    gc.collect()
    return np.asarray([float(x) for x in ev[:N_SHOW]])


# ============================================================
# Incremental pipeline, with bridging through non-evaluated dims.
# ============================================================

def _global_svd_tt_with_carries(core_getter, num_cores, rmax=None, tol=0.0):
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


def run_incremental_all_dims(get_dim, dims, m):
    dims_sorted = sorted(dims)
    D_base, D_max = dims_sorted[0], dims_sorted[-1]
    eval_dims = set(dims)
    omega_base = rff_omega(N_BASIS, SIGMA_RFF)
    out: Dict[int, np.ndarray] = {}

    Xb, dmat_b = get_dim(D_base)
    p = Xb.shape[0]
    psi: List[np.ndarray] = []
    dpsi: List[np.ndarray] = []
    for i in range(p):
        basis_i = RandomFourierFeatures(omega=omega_base)
        Xi = Xb[i:i + 1, :]
        psi.append(basis_i(Xi))
        dpsi.append(basis_i.gradient(Xi)[:, 0, :])
    del Xb

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat_b)
    del dmat_b

    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k, dt=data_tensor):
        return dt.build_core(k + 1)

    U_cores, Sigma_svd, V_core, carries = _global_svd_tt_with_carries(
        core_getter, num_cores=p + 1, rmax=R_TRUNC, tol=TOL
    )
    Z = np.sqrt(m) * V_core[:, :, 0]
    r = U_cores[-1].shape[2]
    extend_carry = carries[p - 2] if p >= 2 else carries[0]
    del V_core, carries

    A_r = compute_A_r(op=op, U_cores=U_cores, r=r, chunk_size=100, n_workers=None, r_cap=r)
    del op

    s = np.diag(Sigma_svd)
    reduced = A_r / s[:, None] / s[None, :]
    del A_r, Sigma_svd, s

    ev = mean_free_whiten_and_eig(Z, reduced, nev=NEV, eps=EPS_EV)
    if D_base in eval_dims:
        out[D_base] = np.asarray([float(x) for x in ev[:N_SHOW]])
    del Z, reduced
    data_tensor.clear_cache()
    gc.collect()

    for D in range(D_base + 1, D_max + 1):
        is_eval = D in eval_dims

        XD, dmat_D = get_dim(D)
        basis_new = RandomFourierFeatures(omega=omega_base)
        X_new = XD[D - 1:D, :]
        psi.append(basis_new(X_new))
        dpsi.append(basis_new.gradient(X_new)[:, 0, :])
        del XD, X_new

        op_D = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat_D)
        del dmat_D
        gc.collect()

        data_tensor = Transformed_Data_Tensor_TT(psi=psi)

        def core_getter(k, dt=data_tensor):
            return dt.build_core(k + 1)

        prev_last_core = core_getter(D - 2)
        G_mid = np.tensordot(extend_carry, prev_last_core, axes=(1, 0))
        rp, n_mid, rn = G_mid.shape
        U_, S_, Vt_ = np.linalg.svd(G_mid.reshape(rp * n_mid, rn), full_matrices=False)
        rr = _truncate_rank(S_, rmax=R_TRUNC, tol=TOL)
        U_mid = U_[:, :rr].reshape(rp, n_mid, rr)
        new_carry = S_[:rr, None] * Vt_[:rr, :]

        new_last_core = core_getter(D - 1)
        G_last = np.tensordot(new_carry, new_last_core, axes=(1, 0))
        rp2, n_last, rn2 = G_last.shape
        U_last_raw, S_last, Vt_last = np.linalg.svd(
            G_last.reshape(rp2 * n_last, rn2), full_matrices=False
        )
        r_D = _truncate_rank(S_last, rmax=R_TRUNC, tol=TOL)
        U_last = U_last_raw[:, :r_D].reshape(rp2, n_last, r_D)
        Sigma_D = np.diag(S_last[:r_D])
        Z_D = np.sqrt(m) * Vt_last[:r_D, :]

        U_cores = U_cores[:-1] + [U_mid, U_last]
        extend_carry = new_carry
        data_tensor.clear_cache()

        if is_eval:
            A_r_D = compute_A_r(op=op_D, U_cores=U_cores, r=r_D, chunk_size=100,
                                n_workers=None, r_cap=r_D)
            del op_D
            s_D = np.diag(Sigma_D)
            reduced_D = A_r_D / s_D[:, None] / s_D[None, :]
            del A_r_D, s_D
            ev = mean_free_whiten_and_eig(Z_D, reduced_D, nev=NEV, eps=EPS_EV)
            out[D] = np.asarray([float(x) for x in ev[:N_SHOW]])
            del reduced_D
        else:
            del op_D
        del Sigma_D, Z_D
        gc.collect()

    return out


# ============================================================
# Demo: one fixed seed, scratch vs incremental across all DIMS
# ============================================================

if __name__ == "__main__":
    DEMO_SEED = BASE_SEED + 1

    Xpool, Dpool, N_pool = load_pool()
    rng = np.random.default_rng(DEMO_SEED)
    idx = np.sort(rng.choice(N_pool, size=M_SUB, replace=False))
    get_dim = make_get_dim(Xpool, Dpool, idx)

    print(f"\n########## SCRATCH -- seed={DEMO_SEED} ##########")
    scratch = {d: run_scratch_one_dim(get_dim, d, M_SUB) for d in DIMS}

    print(f"\n########## INCREMENTAL -- seed={DEMO_SEED} ##########")
    inc = run_incremental_all_dims(get_dim, DIMS, M_SUB)

    print(f"\nDemo result -- seed={DEMO_SEED} -- scratch vs incremental:")
    for d in DIMS:
        print(f"  dim={d}: scratch={scratch[d]}  incremental={inc[d]}")