"""
NTL9 -- ntl9_supplementary.py

Rank & eigenvalues vs SVD truncation tolerance, swept over several RFF
bandwidths (sigma_omega) and repeated with independent random subsamples
for error bars, for TICA dimensions 3, 4, 6, and 8.

This file is the sweep/repeat orchestration only -- it imports the actual
"what does one experiment compute" logic from ntl9_supplementary_base.py
(load_ntl9_data, prepare_ntl9_subset, compute_eigs_one_subsample) and wraps
it in loops over SWEEP_TICA_DIMS x N_SUBSAMPLES x SIGMA_RFF_SWEEP. See that
file to understand the idea on its own, or run it directly for a quick
single-experiment demo without the full sweep.

Data comes from precomputed .npy files (TICA coordinates + diffusion
tensor), same window/standardization as elsewhere in ntl9/ -- no
jax/mdtraj needed here (NTL9's diffusion tensor is already precomputed).

Requirements
------------
Data files at NTL9_DATA_PATH / NTL9_DIFFUSION_PATH (see
ntl9_supplementary_base.py / data/README.md).

Usage
-----
    python ntl9_supplementary.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import save_results  # noqa: E402

from ntl9_supplementary_base import (
    N_EIGS_KEEP,
    START_IDX,
    END_IDX,
    SVD_TOLS,
    get_r_trun,
    load_ntl9_data,
    prepare_ntl9_subset,
    compute_eigs_one_subsample,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "ntl9_supplementary_results.npz"

# ----------------------------------------------------------------------
# Sweep/repeat configuration (verbatim from the source notebook). Physical
# constants (SVD_TOLS, N_BASIS, NEV, R_TRUN_MAP, ...) live in
# ntl9_supplementary_base.py, since they're the same regardless of which
# combination is being swept.
# ----------------------------------------------------------------------
SWEEP_TICA_DIMS = [3, 4, 6, 8]
SIGMA_RFF_SWEEP = [3.0, 5.0, 7.0, 12.0, 25.0]

N_SUBSAMPLES = 10
SUBSAMPLE_M = 6485
SUBSAMPLE_BASE_SEED = 42
SUBSAMPLE_POOL_STRIDE = 1

QUICK_TEST = False
if QUICK_TEST:
    SWEEP_TICA_DIMS = [3]
    SIGMA_RFF_SWEEP = [7.0]
    N_SUBSAMPLES = 2


def aggregate_subsamples(per_sub_results, n_eigs: int):
    """Aggregate per-subsample tol_results into mean/std over subsamples."""
    n_sub = len(per_sub_results)
    n_tol = len(per_sub_results[0])
    agg = []
    for ti in range(n_tol):
        tol = per_sub_results[0][ti]["tol"]
        ranks = np.array([per_sub_results[s][ti]["rank"] for s in range(n_sub)], dtype=float)
        out = {"tol": tol, "rank": int(round(np.nanmean(ranks)))}
        for k in range(n_eigs):
            vals = np.array(
                [per_sub_results[s][ti].get(f"ev{k}", np.nan) for s in range(n_sub)], dtype=float
            )
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                out[f"ev{k}"] = float("nan")
                out[f"ev{k}_std"] = float("nan")
            else:
                out[f"ev{k}"] = float(np.mean(finite))
                out[f"ev{k}_std"] = float(np.std(finite)) if finite.size > 1 else 0.0
        agg.append(out)
    return agg


def main() -> None:
    tica_data, diffusion = load_ntl9_data()

    n_dims, n_sigmas, n_tols = len(SWEEP_TICA_DIMS), len(SIGMA_RFF_SWEEP), len(SVD_TOLS)
    rank_mean = np.full((n_dims, n_sigmas, n_tols), np.nan)
    ev_mean = np.full((n_dims, n_sigmas, n_tols, N_EIGS_KEEP), np.nan)
    ev_std = np.full((n_dims, n_sigmas, n_tols, N_EIGS_KEEP), np.nan)

    t_wall = time.perf_counter()

    for i_dim, tica_dim in enumerate(SWEEP_TICA_DIMS):
        print(f"\n{'=' * 70}\n  SIGMA SWEEP -- tica_dim = {tica_dim}  ({N_SUBSAMPLES} subsamples)\n{'=' * 70}")

        X_pool, D_pool = prepare_ntl9_subset(
            tica_data, diffusion, tica_dim=tica_dim,
            start=START_IDX, end=END_IDX, stride=SUBSAMPLE_POOL_STRIDE,
            offset=0, standardize=True,
        )
        N_pool = X_pool.shape[1]
        if SUBSAMPLE_M > N_pool:
            raise ValueError(f"SUBSAMPLE_M={SUBSAMPLE_M} exceeds pool size {N_pool}.")

        subsets = []
        for run_id in range(1, N_SUBSAMPLES + 1):
            seed = SUBSAMPLE_BASE_SEED + run_id
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(N_pool, size=SUBSAMPLE_M, replace=False))
            subsets.append((seed, X_pool[:, idx].copy(), D_pool[:, :, idx].copy()))
            del idx, rng
        del X_pool, D_pool
        gc.collect()

        r_trun = get_r_trun(tica_dim)
        print(f"  N_pool={N_pool}, r_trun={r_trun}")

        for i_sig, sigma_rff in enumerate(SIGMA_RFF_SWEEP):
            print(f"\n  sigma_rff = {sigma_rff}")

            per_sub_results = []
            for si, (seed, Xs, Ds) in enumerate(subsets):
                print(f"    subsample {si + 1}/{len(subsets)} (seed={seed}, m={Xs.shape[1]})")
                per_sub_results.append(compute_eigs_one_subsample(Xs, Ds, sigma_rff, r_trun))
                gc.collect()

            agg = aggregate_subsamples(per_sub_results, N_EIGS_KEEP)
            for i_tol, row in enumerate(agg):
                rank_mean[i_dim, i_sig, i_tol] = row["rank"]
                for k in range(N_EIGS_KEEP):
                    ev_mean[i_dim, i_sig, i_tol, k] = row[f"ev{k}"]
                    ev_std[i_dim, i_sig, i_tol, k] = row[f"ev{k}_std"]

        del subsets
        gc.collect()
        print(f"  dim={tica_dim} sigma-sweep complete.")

    print(f"\nTotal time: {time.perf_counter() - t_wall:.1f}s")

    save_results(
        RESULTS_PATH,
        tica_dims=np.array(SWEEP_TICA_DIMS),
        sigma_list=np.array(SIGMA_RFF_SWEEP),
        svd_tols=np.array(SVD_TOLS),
        rank_mean=rank_mean,
        ev_mean=ev_mean,
        ev_std=ev_std,
    )


if __name__ == "__main__":
    main()
