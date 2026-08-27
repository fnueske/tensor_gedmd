"""
NTL9 -- run_calculation_pcca_multi_dim.py

3D soft-PCCA state assignment compared across TICA embedding dimension 3,
6, and 8:
  - TICA dim=3: both the Tensor (TT-gEDMD) method and a dense "Matrix"
    reference method are computed.
  - TICA dim=6, 8: Tensor method only.

This file is the multi-dimension orchestration only -- it imports the
actual "what does one dimension compute" logic from
run_calculation_pcca_base.py (load_dim, spectral_analysis_gedmd_dense,
tensor_reduced_matrix) and loops over DIMS = [3, 6, 8]. See that file to
understand the idea on its own (dim=3, both methods), or run it directly
for a quick single-dimension demo without the full sweep.

Data comes from precomputed .npy files (TICA coordinates + diffusion
tensor), same window/standardization as introduction/plot_introduction.py's
NTL9 loader and ntl9/run_calculation_scratch_vs_incremental.py -- no
jax/mdtraj needed here.

Requirements
------------
Data files at NTL9_DATA_PATH / NTL9_DIFFUSION_PATH (see
run_calculation_pcca_base.py / data/README.md).

Usage
-----
    python run_calculation_pcca_multi_dim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import save_results  # noqa: E402

from run_calculation_pcca_base import (
    DIMS,
    K_CLUSTERS,
    N_BASIS,
    SIGMA_RFF,
    load_dim,
    rff_omega,
    spectral_analysis_gedmd_dense,
    tensor_reduced_matrix,
)
from tensor_gedmd.basis_sets.product_basis import ProductBasis
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "ntl9_pcca_multi_dim_results.npz"


def main() -> None:
    results = {}

    for dim in DIMS:
        print(f"\n{'=' * 64}\n  TICA dim = {dim}\n{'=' * 64}")
        Xlist, dmat = load_dim(dim)

        reduced_matrix, V_core, r = tensor_reduced_matrix(Xlist, dmat)
        results[f"Xlist_{dim}"] = Xlist
        results[f"reduced_matrix_{dim}"] = reduced_matrix
        results[f"V_core_{dim}"] = V_core
        results[f"m_{dim}"] = np.array(Xlist.shape[1])

        if dim == DIMS[0]:
            basis_list = [RandomFourierFeatures(omega=rff_omega(N_BASIS, SIGMA_RFF)) for _ in range(dim)]
            phi = ProductBasis(basis_list)
            d_mat, W_mat, Wdata = spectral_analysis_gedmd_dense(
                Xlist, phi, nev=200, Sigma=dmat, tol=1e-4, eps_ev=0.0
            )
            results["Wdata"] = Wdata
            results["d_matrix"] = d_mat

    save_results(RESULTS_PATH, K=np.array(K_CLUSTERS), dims=np.array(DIMS), **results)


if __name__ == "__main__":
    main()
