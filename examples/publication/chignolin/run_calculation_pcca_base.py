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

run_calculation_pcca_multi_dim.py imports load_dim /
spectral_analysis_gedmd_dense / tensor_reduced_matrix from here and loops
over DIMS = [3, 6, 10], using both methods at dim=3 and Tensor-only at
dim=6, 10, to produce chignolin_pcca_multi_dim_results.npz (the actual
paper figure's data).

Unlike chignolin/run_calculation_threshold_and_dim_sweep.py (which
averages over N_RUNS subsamples for error bars), this figure uses one
deterministic, evenly-spaced subsample (np.linspace over the pool) per
dimension -- matching the original notebook, which reports single point
estimates here, not error-barred sweeps.

Data: this loads the precomputed cln_tica.npy / cln_diff.npy (TICA fit
once at dim=10, diffusion tensor projected onto all 10 directions --
validated to give bit-for-bit identical results to fitting/projecting
separately at a smaller dimension and slicing). If you only have the raw
data (cln_tica.pkl, cln_atomic.pkl, chi_vac.pdb -- the last of which is
large, a real jax Jacobian computation), run build_precomputed_data.py
once first to generate these two files; every calculation script here
uses the precomputed files directly and never touches the raw data or
needs jax/mdtraj/deeptime.

Run this file directly for a quick, single-dimension demo -- dim=3, both
methods, no plot. Confirms the pipeline runs correctly on your data before
committing to the full dim=3/6/10 sweep.

Requirements
------------
Just `numpy`/`scipy` plus `tensor_gedmd` itself -- no jax/mdtraj/deeptime
needed, since TICA and the diffusion tensor are already precomputed. See
data/README.md for the precomputed files (cln_tica.npy / cln_diff.npy) or
the raw data if you'd rather regenerate them yourself.

Usage
-----
    python run_calculation_pcca_base.py
"""

from __future__ import annotations

import os
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
CG_TICA_NPY_PATH = Path(os.environ.get("CHIGNOLIN_TICA_NPY", str(DATA_DIR / "cln_tica.npy")))
CG_DIFF_NPY_PATH = Path(os.environ.get("CHIGNOLIN_DIFF_NPY", str(DATA_DIR / "cln_diff.npy")))

# ----------------------------------------------------------------------
# Physical/pipeline constants -- the same regardless of which dimension is
# being computed, so they live here rather than in the multi-dim
# orchestration file. DIMS in particular is the fixed dimension list, not
# something that varies per call. MAX_DIM_AVAILABLE must match whatever
# dimension build_precomputed_data.py fit TICA at (10 by default).
# ----------------------------------------------------------------------
DIMS = [3, 6, 10]
MAX_DIM_AVAILABLE = 10

N_BASIS = 10
SIGMA_RFF = 25.0
TARGET_M = 6000
R_TRUNC = 99999
SVD_TOL = 1e-12
NEV = 40
K_CLUSTERS = 3


def rff_omega(n: int, sigma: float) -> np.ndarray:
    return (np.arange(1, n + 1) / sigma).reshape(-1, 1)


# ============================================================
# Data loading (precomputed -- no jax/mdtraj/deeptime needed)
# ============================================================

def load_dim(dim: int):
    """
    Load the precomputed TICA coordinates and diffusion tensor, slice to
    the requested dimension (dim <= MAX_DIM_AVAILABLE), then take one
    deterministic, evenly-spaced subsample of TARGET_M frames (matching
    the original notebook's np.linspace subsampling -- not a random draw,
    and not repeated/averaged).
    """
    if dim > MAX_DIM_AVAILABLE:
        raise ValueError(
            f"dim={dim} exceeds MAX_DIM_AVAILABLE={MAX_DIM_AVAILABLE} -- "
            f"re-run build_precomputed_data.py with a higher MAX_DIM if you need this."
        )

    tica_data_full = np.load(CG_TICA_NPY_PATH, allow_pickle=True)
    diff_full = np.load(CG_DIFF_NPY_PATH, allow_pickle=True)

    tica_data = tica_data_full[:dim, :]
    dmat_full = diff_full[:dim, :dim, :]

    sample_idx = np.linspace(0, tica_data.shape[1] - 1, TARGET_M, dtype=int)
    Xlist = tica_data[:, sample_idx]
    dmat = dmat_full[:, :, sample_idx]

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

    Xlist, dmat = load_dim(DEMO_DIM)

    reduced_matrix, V_core, r = tensor_reduced_matrix(Xlist, dmat)
    print(f"\nTensor method -- dim={DEMO_DIM}: TT rank={r}")

    basis_list = [RandomFourierFeatures(omega=rff_omega(N_BASIS, SIGMA_RFF)) for _ in range(DEMO_DIM)]
    phi = ProductBasis(basis_list)
    d_mat, W_mat, Wdata = spectral_analysis_gedmd_dense(
        Xlist, phi, nev=200, Sigma=dmat, tol=1e-5, eps_ev=0.0
    )
    print(f"Matrix method -- dim={DEMO_DIM}: leading eigenvalues {np.real(d_mat[:3])}")