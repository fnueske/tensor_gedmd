"""
Chignolin -- run_calculation_pcca_base.py

Base file: the actual idea being tested here, with everything needed to
run it once, made explicit and runnable standalone -- separate from the
"do this at every TICA dimension" orchestration in
run_calculation_pcca_multi_dim.py.

The idea: at TICA dim=3, compute PCCA (soft-membership) state assignment
both the Tensor (TT-gEDMD) way and the dense "Matrix" reference way, so
the two can be shown side by side. Dim=3 is small enough for the dense
method to still be affordable -- at higher dimensions (6, 10) the dense
method becomes prohibitively expensive (n_basis^dim grows fast), so those
use the Tensor method only.

run_calculation_pcca_multi_dim.py imports load_raw_data /
tica_and_diffusion_evenly_spaced / spectral_analysis_gedmd_dense /
tensor_reduced_matrix from here and loops over DIMS = [3, 6, 10], using
both methods at dim=3 and Tensor-only at dim=6, 10, to produce
chignolin_pcca_multi_dim_results.npz (the actual paper figure's data).

Unlike chignolin/run_calculation_threshold_and_dim_sweep.py (which
averages over N_RUNS subsamples for error bars), this figure uses one
deterministic, evenly-spaced subsample (np.linspace over the pool) per
dimension -- matching the original notebook, which reports single point
estimates here, not error-barred sweeps.

The diffusion tensor is estimated the same way as in chignolin/
run_calculation_threshold_and_dim_sweep.py (jax Jacobian of pairwise CA-CA
distances, projected onto the TICA directions) -- kept verbatim since it's
real, data-dependent computation with no tensor_gedmd equivalent.

Run this file directly for a quick, single-dimension demo -- dim=3, both
methods, no plot. Confirms the pipeline runs correctly on your data before
committing to the full dim=3/6/10 sweep.

Requirements
------------
`deeptime`, `mdtraj`, `jax` importable, and the same data files (see
data/README.md).

Usage
-----
    python run_calculation_pcca_base.py
"""

from __future__ import annotations

import gc
import itertools
import os
import pickle
import time
from pathlib import Path

import numpy as np

from tensor_gedmd.algorithms.global_svd import global_svd_tt
from tensor_gedmd.algorithms.mat_vec_prod_direct import compute_A_r
from tensor_gedmd.algorithms.util import spectral_analysis_gedmd_rev
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
# Physical/pipeline constants (verbatim from the source notebook) -- the
# same regardless of which dimension is being computed, so they live here
# rather than in the multi-dim orchestration file. DIMS in particular is
# the fixed dimension list, not something that varies per call.
# ----------------------------------------------------------------------
DIMS = [3, 6, 10]

N_BASIS = 10
SIGMA_RFF = 25.0
LAGTIME = 10
TARGET_M = 6000
R_TRUNC = 99999
SVD_TOL = 1e-12
NEV = 40
K_CLUSTERS = 3

M_MASS = 12.0
T_K = 300.0
GAMMA = 0.1
K_B = 8.314462618e-3
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


def tica_and_diffusion_evenly_spaced(trajectories, diff_full, dim: int):
    """
    Run TICA at a given dimension, project the diffusion tensor onto the
    TICA directions, then take one deterministic, evenly-spaced subsample
    of TARGET_M frames (matching the original notebook's np.linspace
    subsampling -- not a random draw, and not repeated/averaged).
    """
    from deeptime import decomposition

    tica = decomposition.TICA(lagtime=LAGTIME, dim=dim)
    tica_model = tica.fit(trajectories).fetch_model()
    tica_data = tica_model.transform(trajectories).reshape(-1, dim).T
    eigs = tica_model.singular_values
    ind_sort = np.argsort(eigs)[::-1]
    eigvec = tica_model.singular_vectors_left[:, ind_sort]

    dtr = np.einsum("ijlr,ik->kjlr", diff_full, eigvec[:, :dim])
    dmat_full = (2.0 / BETA / M_MASS / GAMMA) * np.einsum("gikl,hikl->ghl", dtr, dtr)
    del dtr

    sample_idx = np.linspace(0, tica_data.shape[1] - 1, TARGET_M, dtype=int)
    Xlist = tica_data[:, sample_idx]
    dmat = dmat_full[:, :, sample_idx]
    del dmat_full

    print(f"  dim={dim}: Xlist {Xlist.shape}, dmat {dmat.shape}")
    return Xlist, dmat


# ============================================================
# Dense/"Matrix" reference method.
#
# Uses this project's mean-free spectral_analysis_gedmd_rev
# (tensor_gedmd.algorithms.util), which mean-centers PhiX before whitening
# by design -- see that function's docstring.
# ============================================================

def spectral_analysis_gedmd_dense(X, phi, nev, Sigma, tol=1e-5, eps_ev=0.0):
    d, W, Wdata = spectral_analysis_gedmd_rev(X, phi, nev, a=Sigma, tol=tol, eps_ev=eps_ev)
    return d[::-1], W[:, ::-1], Wdata[::-1]


# ============================================================
# Tensor (TT) method
# ============================================================

def tensor_reduced_matrix(Xlist, dmat, r_trunc=R_TRUNC, svd_tol=SVD_TOL):
    p, m = Xlist.shape
    omega = rff_omega(N_BASIS, SIGMA_RFF)
    basis_list = [RandomFourierFeatures(omega=omega) for _ in range(p)]
    psi = [basis_list[i](Xlist[i:i + 1, :]) for i in range(p)]
    dpsi = [basis_list[i].gradient(Xlist[i:i + 1, :])[:, 0, :] for i in range(p)]

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=dmat)
    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k):
        return data_tensor.build_core(k + 1)

    U_tt, Sigma_svd, V_core = global_svd_tt(core_getter, num_cores=p + 1, rmax=r_trunc, tol=svd_tol)
    U_cores = U_tt.cores
    r = U_cores[-1].shape[2]

    A_r = compute_A_r(op=op, U_cores=U_cores, r=r, chunk_size=100, n_workers=None, r_cap=r)
    reduced_matrix = np.linalg.inv(Sigma_svd) @ A_r @ np.linalg.inv(Sigma_svd)

    data_tensor.clear_cache()
    return reduced_matrix, V_core, r


# ============================================================
# Demo: dim=3, both methods, no sweep, no plot
# ============================================================

if __name__ == "__main__":
    from tensor_gedmd.basis_sets.product_basis import ProductBasis

    DEMO_DIM = 3

    trajectories, diff_full = load_raw_data()
    Xlist, dmat = tica_and_diffusion_evenly_spaced(trajectories, diff_full, DEMO_DIM)

    reduced_matrix, V_core, r = tensor_reduced_matrix(Xlist, dmat)
    print(f"\nTensor method -- dim={DEMO_DIM}: TT rank={r}")

    basis_list = [RandomFourierFeatures(omega=rff_omega(N_BASIS, SIGMA_RFF)) for _ in range(DEMO_DIM)]
    phi = ProductBasis(basis_list)
    d_mat, W_mat, Wdata = spectral_analysis_gedmd_dense(
        Xlist, phi, nev=200, Sigma=dmat, tol=1e-5, eps_ev=0.0
    )
    print(f"Matrix method -- dim={DEMO_DIM}: leading eigenvalues {np.real(d_mat[:3])}")
