"""
NTL9 -- run_calculation_scratch_vs_incremental.py

Same scratch-vs-incremental comparison as
chignolin/run_calculation_scratch_vs_incremental.py, adapted for NTL9. See
run_calculation_scratch_vs_incremental_base.py for the full
explanation of the idea (including NTL9-specific details: precomputed .npy
loading, the non-contiguous DIMS bridging step, and tracking 4 eigenvalues
instead of 2) and its scope note on what the incremental path does and
doesn't replicate from the original.

This file is the repeat orchestration only -- it imports the actual "what
does one subsample compute" logic from
run_calculation_scratch_vs_incremental_base.py (load_pool,
make_get_dim, run_scratch_one_dim, run_incremental_all_dims) and repeats it
for N_RUNS independent seeds, to produce
ntl9_scratch_vs_incremental_results.npz. See the base file to understand
the idea on its own, or run it directly for a quick single-subsample demo
without the full repeats.

Requirements
------------
Data files at NTL9_DATA_PATH / NTL9_DIFFUSION_PATH (see
run_calculation_scratch_vs_incremental_base.py / data/README.md).

Usage
-----
    python run_calculation_scratch_vs_incremental.py
"""

from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import save_results  # noqa: E402

from run_calculation_scratch_vs_incremental_base import (
    DIMS,
    BASE_SEED,
    M_SUB,
    load_pool,
    make_get_dim,
    run_incremental_all_dims,
    run_scratch_one_dim,
)

RESULTS_PATH = (Path(__file__).resolve().parent / "results"
                / "ntl9_scratch_vs_incremental_results.npz")

# ----------------------------------------------------------------------
# Repeat configuration. Physical/pipeline constants (N_BASIS, SIGMA_RFF,
# R_TRUNC, TOL, DIMS, M_SUB, ...) live in
# run_calculation_scratch_vs_incremental_base.py, since they're the
# same regardless of which seed is being tested.
# ----------------------------------------------------------------------
N_RUNS = 10

QUICK_TEST = False
if QUICK_TEST:
    N_RUNS = 2


def main() -> None:
    Xpool, Dpool, N_pool = load_pool()

    scratch_runs = {d: [] for d in DIMS}
    inc_runs = {d: [] for d in DIMS}

    t_wall = time.perf_counter()
    for run_id in range(1, N_RUNS + 1):
        seed = BASE_SEED + run_id
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(N_pool, size=M_SUB, replace=False))
        get_dim = make_get_dim(Xpool, Dpool, idx)
        t0 = time.perf_counter()

        for d in DIMS:
            scratch_runs[d].append(run_scratch_one_dim(get_dim, d, M_SUB))

        inc_d = run_incremental_all_dims(get_dim, DIMS, M_SUB)
        for d in DIMS:
            inc_runs[d].append(inc_d[d])

        dt = time.perf_counter() - t0
        print(f"run {run_id:02d}/{N_RUNS} seed={seed}  (dim {DIMS[-1]})  [{dt:.0f}s]")

    scratch_arr = {d: np.vstack(v) for d, v in scratch_runs.items()}
    inc_arr = {d: np.vstack(v) for d, v in inc_runs.items()}

    print(f"\nTotal {time.perf_counter() - t_wall:.0f}s.")

    scratch_stack = np.stack([scratch_arr[d] for d in DIMS], axis=0)
    inc_stack = np.stack([inc_arr[d] for d in DIMS], axis=0)

    save_results(
        RESULTS_PATH,
        dims=np.array(DIMS),
        scratch_runs=scratch_stack,
        inc_runs=inc_stack,
    )


if __name__ == "__main__":
    main()