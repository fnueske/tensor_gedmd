"""
Chignolin -- chignolin_supplementary.py

Rank & eigenvalues (kappa_1, kappa_2) vs SVD truncation tolerance, swept over
several RFF bandwidths (sigma_omega) and repeated with independent random
subsamples for error bars, for TICA dimensions 3, 4, 6, 8, and 10.

This file is the sweep/repeat orchestration only -- it imports the actual
"what does one experiment compute" logic from chignolin_supplementary_base.py
(load_raw_data, fit_tica_and_diffusion, subsample_one_repeat, solve_one) and
wraps it in loops over TICA_DIMS x N_REPEATS x SIGMA_LIST x SVD_TOLS. See
that file to understand the idea on its own, or run it directly for a quick
single-experiment demo without the full sweep.

The original analysis was done as three separate notebooks (dims 3/4/6/8,
then dim=10 added separately later at the supervisor's request so the first
run didn't need to be repeated, then a third notebook combining both into
one figure). Here all five dimensions are computed in one script -- we
aren't under the "don't repeat the expensive part" constraint that produced
that split, so there's no need to reproduce it.

Requirements
------------
- `deeptime`, `mdtraj`, `jax` importable.
- Data files at CG_PICKLE_PATH / CG_ATOMIC_NUMBERS_PATH / PDB_PATH (see
  chignolin_supplementary_base.py / data/README.md).

Usage
-----
    python chignolin_supplementary.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import save_results  # noqa: E402

from chignolin_supplementary_base import (
    R_TRUN_MAP,
    fit_tica_and_diffusion,
    load_raw_data,
    solve_one,
    subsample_one_repeat,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "chignolin_supplementary_results.npz"

# ----------------------------------------------------------------------
# Sweep/repeat configuration (verbatim from the source notebooks). Physical
# constants (LAGTIME, TARGET_M, N_BASIS, ...) live in
# chignolin_supplementary_base.py, since they're the same regardless of
# which combination is being swept.
# ----------------------------------------------------------------------
TICA_DIMS = [3, 4, 6, 8, 10]
SVD_TOLS = [1e-3, 1e-5, 1e-7, 1e-10, 1e-12, 1e-15]
SIGMA_LIST = [3, 5, 7, 12, 25]  # RFF bandwidths swept for every tica_dim
N_REPEATS = 10                  # independent random subsamples per (dim, sigma)
BASE_SEED = 42                  # repeat r uses seed = BASE_SEED + r

QUICK_TEST = False
if QUICK_TEST:
    TICA_DIMS = [3]
    SIGMA_LIST = [7]
    N_REPEATS = 2
    SVD_TOLS = [1e-3, 1e-10]


def main() -> None:
    trajectories, diff_full = load_raw_data()

    n_dims, n_sigmas, n_tols = len(TICA_DIMS), len(SIGMA_LIST), len(SVD_TOLS)
    rank_mean = np.full((n_dims, n_sigmas, n_tols), np.nan)
    ev0_mean = np.full((n_dims, n_sigmas, n_tols), np.nan)
    ev0_std = np.full((n_dims, n_sigmas, n_tols), np.nan)
    ev1_mean = np.full((n_dims, n_sigmas, n_tols), np.nan)
    ev1_std = np.full((n_dims, n_sigmas, n_tols), np.nan)

    t_wall = time.perf_counter()

    for i_dim, tica_dim in enumerate(TICA_DIMS):
        print(f"\n{'=' * 70}\n  tica_dim = {tica_dim}\n{'=' * 70}")

        # Expensive TICA fit + diffusion projection: once per dim.
        tica_data, dtr, N_total, r_trun = fit_tica_and_diffusion(trajectories, diff_full, tica_dim)

        raw = {sigma: {tol: [] for tol in SVD_TOLS} for sigma in SIGMA_LIST}

        for rep in range(N_REPEATS):
            seed = BASE_SEED + rep
            # Cheap: draw one subsample against the already-fit TICA pool.
            Xlist, dmat = subsample_one_repeat(tica_data, dtr, N_total, seed)

            for sigma_rff in SIGMA_LIST:
                print(f"  repeat {rep + 1}/{N_REPEATS}  sigma={sigma_rff}", flush=True)
                for tol in SVD_TOLS:
                    t0 = time.perf_counter()
                    res = solve_one(Xlist, dmat, sigma_rff, r_trun, tol)
                    raw[sigma_rff][tol].append((res["rank"], res["ev0"], res["ev1"]))
                    print(f"    tol={tol:.0e}  rank={res['rank']:5d}  "
                          f"ev0={res['ev0']:+.6f}  ev1={res['ev1']:+.6f}  "
                          f"({time.perf_counter() - t0:.1f}s)")

            del Xlist, dmat

        del tica_data, dtr

        for i_sig, sigma_rff in enumerate(SIGMA_LIST):
            for i_tol, tol in enumerate(SVD_TOLS):
                trip = np.array(raw[sigma_rff][tol], dtype=float)
                ranks, ev0s, ev1s = trip[:, 0], trip[:, 1], trip[:, 2]
                rank_mean[i_dim, i_sig, i_tol] = np.mean(ranks)
                ev0_mean[i_dim, i_sig, i_tol] = np.nanmean(ev0s)
                ev0_std[i_dim, i_sig, i_tol] = np.nanstd(ev0s)
                ev1_mean[i_dim, i_sig, i_tol] = np.nanmean(ev1s)
                ev1_std[i_dim, i_sig, i_tol] = np.nanstd(ev1s)

        print(f"  dim={tica_dim} complete.")

    print(f"\nTotal time: {time.perf_counter() - t_wall:.1f}s")

    save_results(
        RESULTS_PATH,
        tica_dims=np.array(TICA_DIMS),
        sigma_list=np.array(SIGMA_LIST),
        svd_tols=np.array(SVD_TOLS),
        rank_mean=rank_mean,
        ev0_mean=ev0_mean, ev0_std=ev0_std,
        ev1_mean=ev1_mean, ev1_std=ev1_std,
    )


if __name__ == "__main__":
    main()
