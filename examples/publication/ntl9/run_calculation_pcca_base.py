"""
NTL9 -- run_calculation_pcca_base.py

Base file: the actual idea being tested here, with everything needed to
run it once, made explicit and runnable standalone -- separate from the
"do this at every TICA dimension" orchestration in
run_calculation_pcca_multi_dim.py.

The idea: at TICA dim=3, compute PCCA (soft-membership) state assignment
both the Tensor (TT-gEDMD) way and the dense "Matrix" reference way, so
the two can be shown side by side. Dim=3 is small enough for the dense
method to still be affordable -- at higher dimensions (6, 8) the dense
method becomes prohibitively expensive, so those use the Tensor method
only.

run_calculation_pcca_multi_dim.py imports load_dim /
spectral_analysis_gedmd_dense / tensor_reduced_matrix from here and loops
over DIMS = [3, 6, 8], using both methods at dim=3 and Tensor-only at
dim=6, 8, to produce ntl9_pcca_multi_dim_results.npz (the actual paper
figure's data).

Data comes from precomputed .npy files (TICA coordinates + diffusion
tensor), same window/standardization as elsewhere in ntl9/ -- no
jax/mdtraj needed here.

Run this file directly for a quick, single-dimension demo -- dim=3, both
methods, no plot. Confirms the pipeline runs correctly on your data before
committing to the full dim=3/6/8 sweep.

Requirements
------------
Data files at NTL9_DATA_PATH / NTL9_DIFFUSION_PATH below (see data/README.md).

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
# NTL9_DATA_DIR environment variable if your data lives elsewhere -- no
# need to edit this file.
DATA_DIR = Path(os.environ.get(
    "NTL9_DATA_DIR", str(Path(__file__).resolve().parent / "data")
))
NTL9_DATA_PATH = DATA_DIR / "ntl9_tica.npy"
NTL9_DIFFUSION_PATH = DATA_DIR / "ntl9_diff.npy"

# ----------------------------------------------------------------------
# Physical/pipeline constants (verbatim from the source notebook) -- the
# same regardless of which dimension is being computed, so they live here
# rather than in the multi-dim orchestration file. DIMS in particular is
# the fixed dimension list, not something that varies per call.
# ----------------------------------------------------------------------
DIMS = [3, 6, 8]

N_BASIS = 10
SIGMA_RFF = 25.0
STRIDE = 3
R_TRUNC = 99999
SVD_TOL = 1e-12
NEV = 40
K_CLUSTERS = 4

WINDOW_START = 109267
WINDOW_END = 128720


def rff_omega(n: int, sigma: float) -> np.ndarray:
    return (np.arange(1, n + 1) / sigma).reshape(-1, 1)


# ============================================================
# Data loading (same window/standardization as elsewhere in ntl9/)
# ============================================================

def load_dim(dim: int):
    tica_data = np.load(NTL9_DATA_PATH, allow_pickle=True)
    diffusion = np.load(NTL9_DIFFUSION_PATH, allow_pickle=True)

    Xlist = tica_data[:dim, WINDOW_START:WINDOW_END:STRIDE]
    dmat = diffusion[:dim, :dim, WINDOW_START:WINDOW_END:STRIDE]

    s = Xlist.std(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    Xlist = Xlist / s
    sc = s[:, 0]
    dmat = dmat / (sc[:, None, None] * sc[None, :, None])

    print(f"  dim={dim}: Xlist {Xlist.shape}, dmat {dmat.shape}")
    return Xlist, dmat


# ============================================================
# Dense/"Matrix" reference method.
#
# Uses this project's mean-free spectral_analysis_gedmd_rev
# (tensor_gedmd.algorithms.util) -- see that function's docstring.
# ============================================================

def spectral_analysis_gedmd_dense(X, phi, nev, Sigma, tol=1e-4, eps_ev=0.0):
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
        Xlist, phi, nev=200, Sigma=dmat, tol=1e-4, eps_ev=0.0
    )
    print(f"Matrix method -- dim={DEMO_DIM}: leading eigenvalues {np.real(d_mat[:3])}")
