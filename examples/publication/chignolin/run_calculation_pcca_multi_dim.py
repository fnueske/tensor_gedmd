"""
Chignolin -- run_calculation_pcca_multi_dim.py

PCCA (soft-membership) state assignment compared across TICA embedding
dimension 3, 6, and 10:
  - TICA dim=3: both the Tensor (TT-gEDMD) method and a dense "Matrix"
    reference method are computed, so the two can be shown side by side.
  - TICA dim=6, 10: Tensor method only (the dense method becomes
    prohibitively expensive at these basis sizes -- n_basis^dim grows fast).

This file is the multi-dimension orchestration only -- it imports the
actual "what does one dimension compute" logic from
run_calculation_pcca_base.py (load_raw_data,
tica_and_diffusion_evenly_spaced, spectral_analysis_gedmd_dense,
tensor_reduced_matrix) and loops over DIMS = [3, 6, 10]. See that file to
understand the idea on its own (dim=3, both methods), or run it directly
for a quick single-dimension demo without the full sweep.

Requirements
------------
Same as run_calculation_pcca_base.py: `deeptime`, `mdtraj`, `jax`
importable, and the same data files (see data/README.md).

Usage
-----
    python run_calculation_pcca_multi_dim.py
"""

from __future__ import annotations

import gc
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
    load_raw_data,
    rff_omega,
    spectral_analysis_gedmd_dense,
    tensor_reduced_matrix,
    tica_and_diffusion_evenly_spaced,
)
from tensor_gedmd.basis_sets.product_basis import ProductBasis
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "chignolin_pcca_multi_dim_results.npz"


def main() -> None:
    trajectories, diff_full = load_raw_data()

    results = {}
    for dim in DIMS:
        print(f"\n{'=' * 64}\n  TICA dim = {dim}\n{'=' * 64}")
        Xlist, dmat = tica_and_diffusion_evenly_spaced(trajectories, diff_full, dim)

        reduced_matrix, V_core, r = tensor_reduced_matrix(Xlist, dmat)
        results[f"Xlist_{dim}"] = Xlist
        results[f"reduced_matrix_{dim}"] = reduced_matrix
        results[f"V_core_{dim}"] = V_core
        results[f"m_{dim}"] = np.array(Xlist.shape[1])

        if dim == DIMS[0]:
            # Dense/"Matrix" reference method -- dim=3 only (cost grows as
            # N_BASIS**dim, prohibitive for dim=6/10). Uses the real
            # samplewise diffusion tensor directly (no averaging).
            basis_list = [RandomFourierFeatures(omega=rff_omega(N_BASIS, SIGMA_RFF)) for _ in range(dim)]
            phi = ProductBasis(basis_list)
            d_mat, W_mat, Wdata = spectral_analysis_gedmd_dense(
                Xlist, phi, nev=200, Sigma=dmat, tol=1e-5, eps_ev=0.0
            )
            results["Wdata"] = Wdata

        del Xlist, dmat
        gc.collect()

    save_results(RESULTS_PATH, K=np.array(K_CLUSTERS), dims=np.array(DIMS), **results)


if __name__ == "__main__":
    main()
