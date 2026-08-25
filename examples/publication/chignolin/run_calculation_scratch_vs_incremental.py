"""
Chignolin -- run_calculation_scratch_vs_incremental.py (scratch-vs-incremental comparison)

Compares two ways of computing the leading generator eigenvalues as the
TICA embedding dimension grows from 3 to 10: "scratch" (recompute
everything at every dimension) vs "incremental" (reuse earlier TT-SVD
cores). See run_calculation_scratch_vs_incremental_base.py for
the full explanation of the idea and its scope note on what the
incremental path does and doesn't replicate from the original.

This file is the repeat orchestration only -- it imports the actual
"what does one subsample compute" logic from
run_calculation_scratch_vs_incremental_base.py (load_raw_data,
tica_pool, run_scratch_one_dim, run_incremental_all_dims) and repeats it
for N_RUNS independent seeds, to produce
chignolin_scratch_vs_incremental_results.npz. See the base file to
understand the idea on its own, or run it directly for a quick
single-subsample demo without the full repeats.

Requirements
------------
Same as run_calculation_threshold_and_dim_sweep_base.py:
`deeptime`, `mdtraj`, `jax` importable, and the same data files.

Usage
-----
    python run_calculation_scratch_vs_incremental.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import save_results  # noqa: E402

from run_calculation_scratch_vs_incremental_base import (
    DIMS,
    BASE_SEED,
    load_raw_data,
    run_incremental_all_dims,
    run_scratch_one_dim,
    tica_pool,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "chignolin_scratch_vs_incremental_results.npz"

# ----------------------------------------------------------------------
# Repeat configuration. Physical/pipeline constants (N_BASIS, SIGMA_RFF,
# LAGTIME, TARGET_M, R_TRUNC, TOL, DIMS, ...) live in
# run_calculation_scratch_vs_incremental_base.py, since they're
# the same regardless of which seed is being tested.
# ----------------------------------------------------------------------
N_RUNS = 10
SEEDS = [BASE_SEED + run_id for run_id in range(1, N_RUNS + 1)]

QUICK_TEST = False
if QUICK_TEST:
    SEEDS = SEEDS[:2]


def main() -> None:
    trajectories, diff_full = load_raw_data()

    n_dims, n_seeds = len(DIMS), len(SEEDS)
    dim_index = {d: i for i, d in enumerate(DIMS)}

    full_ev0 = np.full((n_dims, n_seeds), np.nan)
    full_ev1 = np.full((n_dims, n_seeds), np.nan)
    full_time = np.full((n_dims, n_seeds), np.nan)
    upd_ev0 = np.full((n_dims, n_seeds), np.nan)
    upd_ev1 = np.full((n_dims, n_seeds), np.nan)
    upd_time = np.full((n_dims, n_seeds), np.nan)

    print("Pre-fitting TICA pools (once per dim) ...")
    for d in DIMS:
        tica_pool(trajectories, d)

    for j, seed in enumerate(SEEDS):
        print(f"\n==================== SUBSAMPLE {j + 1}/{n_seeds}  (seed={seed}) ====================")

        print(f"########## FULL (scratch) -- seed={seed} ##########")
        for d in DIMS:
            e0, e1, t = run_scratch_one_dim(trajectories, diff_full, d, seed)
            i = dim_index[d]
            full_ev0[i, j] = e0
            full_ev1[i, j] = e1
            full_time[i, j] = t

        iev0, iev1, itime = run_incremental_all_dims(trajectories, diff_full, seed)
        for d in DIMS:
            i = dim_index[d]
            upd_ev0[i, j] = iev0[d]
            upd_ev1[i, j] = iev1[d]
            upd_time[i, j] = itime[d]

        gc.collect()

    print("\nAll subsamples complete.")

    save_results(
        RESULTS_PATH,
        dims=np.array(DIMS),
        full_ev0=full_ev0, full_ev1=full_ev1, full_time=full_time,
        upd_ev0=upd_ev0, upd_ev1=upd_ev1, upd_time=upd_time,
    )


if __name__ == "__main__":
    main()
