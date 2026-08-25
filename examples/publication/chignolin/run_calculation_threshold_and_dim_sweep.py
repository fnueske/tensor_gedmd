"""
Chignolin -- run_calculation_threshold_and_dim_sweep.py (Experiments 1 & 2: threshold sweep + dimension sweep)

Two combined experiments, both feeding
plots/chignolin_plot_figure_eigenvalues_vs_svd_threshold_and_tica_dimension.py:

  Experiment 2 (threshold sweep): for a few TICA dimensions (CURVE_DIMS),
    sweep the global_svd_tt truncation tolerance and record the leading two
    eigenvalues (kappa_1, kappa_2), with error bars over N_RUNS subsamples.
    -> figure panels (a)/(b)

  Experiment 1 (dimension sweep): at one fixed tolerance, sweep the TICA
    embedding dimension itself and record kappa_1, kappa_2.
    -> figure panels (c)/(d)

This file is the sweep/repeat orchestration only -- it imports the actual
"what does one experiment compute" logic from
run_calculation_threshold_and_dim_sweep_base.py (load_raw_data,
tica_and_diffusion, run_subsample) and wraps run_subsample in loops over
dims x N_RUNS, with per-dimension sigma/rank-cap settings and tolerance
lists. See that file to understand the idea on its own, or run it directly
for a quick single-experiment demo without the full sweep.

Requirements
------------
- `deeptime` (TICA), `mdtraj` (topology/CA selection), `jax` (Jacobian of
  pairwise distances) all importable.
- Data files at CG_PICKLE_PATH / CG_ATOMIC_NUMBERS_PATH / PDB_PATH (see
  run_calculation_threshold_and_dim_sweep_base.py / data/README.md).

Usage
-----
    python run_calculation_threshold_and_dim_sweep.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import save_results  # noqa: E402

from run_calculation_threshold_and_dim_sweep_base import (
    load_raw_data,
    run_subsample,
    tica_and_diffusion,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "chignolin_results.npz"

# ----------------------------------------------------------------------
# Sweep/repeat configuration (verbatim from the original notebook). Physical
# constants (LAGTIME, TARGET_M, N_BASIS, ...) live in
# run_calculation_threshold_and_dim_sweep_base.py, since they're
# the same regardless of which combination is being swept.
# ----------------------------------------------------------------------

# Experiment 2: threshold sweep -> panels (a)/(b)
CURVE_DIMS = [3, 6, 10]
SVD_TOLS = [1e-6, 1e-7, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
SIGMA_E2 = {3: 25.0, 4: 25.0, 6: 25.0, 8: 25.0, 10: 25.0}
RTRUN_E2 = {3: 99999, 4: 99999, 6: 99999, 8: 99999, 10: 99999}

# Experiment 1: dimension sweep -> panels (c)/(d)
TICA_DIMS = [3, 4, 6, 8, 10]
SELECTED_TOL = 1e-10
SIGMA_E1 = {3: 6.0, 4: 6.0, 6: 6.0, 8: 25.0, 10: 25.0}
RTRUN_E1 = {3: 300, 4: 500, 6: 2000, 8: 3500, 10: 4500}

N_RUNS = 10

# Set True for a fast end-to-end smoke test (tiny dims/tols/runs).
QUICK_TEST = False
if QUICK_TEST:
    CURVE_DIMS = [3, 6]
    SVD_TOLS = [1e-6, 1e-10]
    TICA_DIMS = [3, 6]
    N_RUNS = 2

assert set(CURVE_DIMS).issubset(TICA_DIMS), "CURVE_DIMS must be subset of TICA_DIMS"
assert SELECTED_TOL in SVD_TOLS, "SELECTED_TOL must be one of SVD_TOLS"


def main() -> None:
    trajectories, diff_full = load_raw_data()

    all_dims = sorted(set(TICA_DIMS) | set(CURVE_DIMS))

    # Threshold-sweep results: (n_curve_dims, N_RUNS, n_tols)
    thr_k1 = np.full((len(CURVE_DIMS), N_RUNS, len(SVD_TOLS)), np.nan)
    thr_k2 = np.full((len(CURVE_DIMS), N_RUNS, len(SVD_TOLS)), np.nan)
    thr_rank = np.full((len(CURVE_DIMS), N_RUNS, len(SVD_TOLS)), np.nan)

    # Dimension-sweep results: (n_tica_dims, N_RUNS)
    dim_k1 = np.full((len(TICA_DIMS), N_RUNS), np.nan)
    dim_k2 = np.full((len(TICA_DIMS), N_RUNS), np.nan)
    dim_rank = np.full((len(TICA_DIMS), N_RUNS), np.nan)

    t_wall = time.perf_counter()
    for dim in all_dims:
        print(f"\n{'=' * 64}\n  TICA dim = {dim}\n{'=' * 64}")
        tica_data, dmat_all, N_total = tica_and_diffusion(trajectories, diff_full, dim)

        if dim in CURVE_DIMS:
            i_dim = CURVE_DIMS.index(dim)
            print(f"  -- threshold sweep (sigma={SIGMA_E2[dim]}, r_trun={RTRUN_E2[dim]}) --")
            for run_id in range(1, N_RUNS + 1):
                res = run_subsample(dim, run_id, tica_data, dmat_all, N_total,
                                    SIGMA_E2[dim], RTRUN_E2[dim], SVD_TOLS)
                for i_tol, tol in enumerate(SVD_TOLS):
                    k1, k2, rank_got = res[tol]
                    thr_k1[i_dim, run_id - 1, i_tol] = k1
                    thr_k2[i_dim, run_id - 1, i_tol] = k2
                    thr_rank[i_dim, run_id - 1, i_tol] = rank_got

        if dim in TICA_DIMS:
            i_dim = TICA_DIMS.index(dim)
            print(f"  -- dimension-sweep point (sigma={SIGMA_E1[dim]}, "
                  f"r_trun={RTRUN_E1[dim]}, tol={SELECTED_TOL:.0e}) --")
            for run_id in range(1, N_RUNS + 1):
                res = run_subsample(dim, run_id, tica_data, dmat_all, N_total,
                                    SIGMA_E1[dim], RTRUN_E1[dim], [SELECTED_TOL])
                k1, k2, rank_got = res[SELECTED_TOL]
                dim_k1[i_dim, run_id - 1] = k1
                dim_k2[i_dim, run_id - 1] = k2
                dim_rank[i_dim, run_id - 1] = rank_got

        del tica_data, dmat_all
        gc.collect()

    print(f"\nAll dims done. Total: {time.perf_counter() - t_wall:.1f}s")

    save_results(
        RESULTS_PATH,
        curve_dims=np.array(CURVE_DIMS),
        svd_tols=np.array(SVD_TOLS),
        thr_k1=thr_k1, thr_k2=thr_k2, thr_rank=thr_rank,
        tica_dims=np.array(TICA_DIMS),
        dim_k1=dim_k1, dim_k2=dim_k2, dim_rank=dim_rank,
    )


if __name__ == "__main__":
    main()
