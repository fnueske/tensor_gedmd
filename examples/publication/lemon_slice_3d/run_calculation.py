"""
Lemon Slice 3D -- run_calculation.py

Does the actual (expensive) computation: simulates the Lemon-Slice SDE,
runs the TT-gEDMD pipeline (basis -> TT data tensor -> global SVD ->
Sigma-aware reduced generator -> whitening -> eigendecomposition -> PCCA),
a dense/"Matrix" reference-method baseline for comparison, an eigenvalue
error-bar sweep over data size m (Matrix vs Tensor), a TT-rank vs
SVD-tolerance sweep, and a mat-vec runtime comparison (block contraction,
i.e. applying the TT-format stiffness operator column-by-column, vs the
direct method that never materializes that operator at all). Saves
everything the figure needs to results/ and does NOT plot anything --
see plots/plot_figure_lemon_slice.py for that.

This file is the sweep/repeat orchestration only for Sections B.1 and B.2
-- it imports the actual "what does one experiment compute" logic (both
the eigenvalue idea and the rank idea) from
lemon_slice_3d_run_calculation_base.py and wraps it in loops over data
sizes x 10 independent experiments. See that file to understand either
idea on its own, or run it directly for a quick single-experiment demo
without the full sweep. Section A (the single, non-repeated main PCCA
pipeline) and Section B.3 (already a single-experiment runtime benchmark)
also live here, since neither is part of the "repeat 10x" pattern.

Refinements applied here vs. the original notebook version:
  - Lemon-Slice potential panel removed (no longer part of the figure)
  - eigenvalue panel moved from bottom-left to upper-left
  - new bottom-left panel: mat-vec runtime, block contraction vs direct

Diffusion here is a scalar constant (isotropic, Sigma = (2/beta) * I_p) for
the main pipeline and eigenvalue/rank sweeps -- the "scalar" fast path in
tensor_gedmd.algorithms.mat_vec_prod_direct.compute_A_r. The runtime
benchmark specifically compares block contraction vs direct for the
no-Sigma-at-all case (Sigma=None), matching the source notebook's
TgStiffnessOperator(psi, dpsi) call with no Sigma argument.

Requirements
------------
- The `tensor_gedmd` package (this repository) importable -- includes the
  Lemon-Slice `Decoupled_3d` SDE simulator (tensor_gedmd.systems), no
  external simulator package needed.

Usage
-----
    python run_calculation.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import save_results  # noqa: E402

from lemon_slice_3d_run_calculation_base import (
    DIFFUSION_CONST,
    DSAVE,
    DT,
    LENGTH_SCALE,
    N_FEATURES,
    NTEST,
    P_DIMS,
    R_TRUNC,
    build_simulator,
    get_max_tt_rank,
    simulate,
    spectral_analysis_gedmd_dense,
    tensor_spectral_analysis,
)
from tensor_gedmd.algorithms.gedmd import build_deterministic_rff_basis, evaluate_basis
from tensor_gedmd.algorithms.global_svd import global_svd_tt
from tensor_gedmd.algorithms.mat_vec_prod import prepare_blocks, tt_matrix_vector_product_csr_prepared
from tensor_gedmd.algorithms.mat_vec_prod_direct import compute_A_r
from tensor_gedmd.basis_sets.product_basis import ProductBasis
from tensor_gedmd.operations import extract_tt_column, tt_inner_product
from tensor_gedmd.reps.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.reps.transformed_data_tensor import Transformed_Data_Tensor_TT

# ----------------------------------------------------------------------
# Section-A-specific and sweep-orchestration config. Physical/pipeline
# constants (KK, LL, Z_PARAM, BETA, N_FEATURES, R_TRUNC, ...) live in
# lemon_slice_3d_run_calculation_base.py, since they're the same regardless
# of which m or repeat is being tested.
# ----------------------------------------------------------------------
M_MAIN = 7000                   # data size for the main PCCA pipeline
SVD_TOL_MAIN = 1e-9
NEV = 40
K_CLUSTERS = 4                   # PCCA cluster count

# Sweep settings (Section B)
M_VALUES_EIG = [2000, 4000, 6000, 7000, 10000]
N_EXPERIMENTS_EIG = 10
NUM_PLOT_EIGS = 4

M_VALUES_RANK = [2000, 4000, 6000, 7000, 10000]
TOL_VALUES_RANK = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
N_EXPERIMENTS_RANK = 10

# Runtime benchmark (block contraction vs direct); assumes P_DIMS == 3, i.e.
# exactly one middle TT core, matching prepare_blocks/
# tt_matrix_vector_product_csr_prepared's fixed rank-2 (0,0)/(1,1)/(0,1)
# structure.
M_VALUES_RUNTIME = [2000, 4000, 6000, 7000, 10000]
N_EXPERIMENTS_RUNTIME = 1

# Set True for a fast end-to-end smoke test (tiny m values, 1-2 experiments,
# 7000 -> a few hundred samples for the main pipeline too). Confirms the
# whole pipeline runs correctly before committing to the full sweep (which
# is the heaviest script in publication/ -- 50 solves for the eigenvalue
# sweep, 350 for the rank sweep, on top of the main pipeline and runtime
# benchmark).
QUICK_TEST = False
if QUICK_TEST:
    M_MAIN = 300
    M_VALUES_EIG = [200, 400]
    N_EXPERIMENTS_EIG = 1
    M_VALUES_RANK = [200, 400]
    TOL_VALUES_RANK = [1e-2, 1e-6]
    N_EXPERIMENTS_RANK = 1
    M_VALUES_RUNTIME = [200, 400]
    N_EXPERIMENTS_RUNTIME = 1

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "lemon_slice_3d_results.npz"

# Toggle individual sections on/off. All default True (full run, unchanged
# behavior). Turn sections off to test one piece in isolation -- e.g. set
# only RUN_SECTION_B3 = True to sanity-check just the (cheap, 1-experiment)
# runtime benchmark without also paying for Section A's main pipeline or
# the 10-experiment sweeps in B.1/B.2. Results from sections you skip are
# preserved from any existing results file rather than being wiped out
# (see main(), which merges into whatever's already saved).
RUN_SECTION_A = True    # main PCCA pipeline
RUN_SECTION_B1 = True   # eigenvalue error-bar sweep (10 experiments/m)
RUN_SECTION_B2 = True   # TT-rank vs SVD-tolerance sweep (10 experiments/m)
RUN_SECTION_B3 = True   # mat-vec runtime benchmark (1 experiment/m)


# ============================================================
# PCCA (Section A only)
# ============================================================

def _pcca_connected_isa(eigenvectors, n_clusters):
    n, m = eigenvectors.shape
    diffs = np.abs(np.max(eigenvectors, axis=0) - np.min(eigenvectors, axis=0))
    assert diffs[0] < 1e-6, "First eigenvector is not constant. Cannot do PCCA."
    assert diffs[1] > 1e-6, "An eigenvector after the first one is constant. Cannot do PCCA."

    c = eigenvectors[:, :n_clusters].copy()
    ind = np.zeros(n_clusters, dtype=np.int32)
    ind[0] = int(np.argmax(np.linalg.norm(c, axis=1)))
    ortho = c - c[ind[0], None, :]

    for k in range(1, n_clusters):
        temp = ortho[ind[k - 1]].copy()
        tnorm = np.linalg.norm(temp)
        if tnorm > 0:
            temp /= tnorm
        dots = ortho @ temp
        proj = ortho - dots[:, None] * temp
        dists = np.linalg.norm(proj, axis=1)
        dists[ind[:k]] = -np.inf
        ind[k] = int(np.argmax(dists))
        ortho = proj

    rot_mat = np.linalg.inv(c[ind, :])
    chi = c @ rot_mat
    return chi, rot_mat


def pcca_from_slow_modes(slow_modes, K):
    m = slow_modes.shape[0]
    evec = np.hstack([np.ones((m, 1)), np.real(slow_modes)])
    evec[:, 0] = 1.0
    for k in range(1, K):
        s = np.std(evec[:, k])
        if s > 0:
            evec[:, k] /= s
    chi, rot = _pcca_connected_isa(evec, n_clusters=K)
    chi = np.clip(chi, 0.0, None)
    rowsum = chi.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    chi /= rowsum
    labels = np.argmax(chi, axis=1)
    return chi, labels, rot


def align_labels(labels_tensor, labels_matrix, K):
    """Permute tensor cluster IDs to best match matrix cluster IDs
    (cosmetic only -- for consistent plot colors across methods)."""
    C = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            C[i, j] = np.sum((labels_tensor == i) & (labels_matrix == j))
    _, col_ind = linear_sum_assignment(-C)
    return col_ind[labels_tensor]


# ============================================================
# Section A: main PCCA pipeline
# ============================================================

def run_main_pipeline(LS, basis_list):
    Xlist = simulate(LS, M_MAIN)

    d_tt, W_tt, Z_white, U_cores, Sigma_svd = tensor_spectral_analysis(
        Xlist, basis_list, R_TRUNC, SVD_TOL_MAIN, NEV
    )

    phi = ProductBasis(basis_list)
    d_mat, W_mat, Wdata = spectral_analysis_gedmd_dense(
        Xlist, phi, nev=200, diffusion_const=DIFFUSION_CONST, tol=1e-4, eps_ev=0.0
    )

    eigfuns, mus = [], []
    for j in range(K_CLUSTERS - 1):
        w = np.ravel(W_tt[:, j])
        eigfuns.append(np.real(w @ Z_white))
        mus.append(d_tt[j])
    slow_tensor = np.column_stack(eigfuns)
    _, labels_tensor, _ = pcca_from_slow_modes(slow_tensor, K_CLUSTERS)

    slow_matrix = np.real(Wdata[:K_CLUSTERS - 1, :].T)
    _, labels_matrix, _ = pcca_from_slow_modes(slow_matrix, K_CLUSTERS)

    labels_tensor_aligned = align_labels(labels_tensor, labels_matrix, K_CLUSTERS)

    return {
        "Xlist": Xlist,
        "labels_tensor_aligned": labels_tensor_aligned,
        "labels_matrix": labels_matrix,
        "K": K_CLUSTERS,
    }


# ============================================================
# Section B.1: eigenvalue error-bar sweep (Matrix vs Tensor)
# ============================================================

def run_eigenvalue_sweep(LS):
    matrix_eigs_exp = np.full((len(M_VALUES_EIG), N_EXPERIMENTS_EIG, NUM_PLOT_EIGS), np.nan)
    tensor_eigs_exp = np.full((len(M_VALUES_EIG), N_EXPERIMENTS_EIG, NUM_PLOT_EIGS), np.nan)

    for i_m, m in enumerate(M_VALUES_EIG):
        print(f"\n[eig sweep] m = {m}")
        Xfull_long = LS.simulate(np.ones((NTEST, 3)), m * DSAVE * N_EXPERIMENTS_EIG, DT)[:, ::DSAVE]
        total_snaps = Xfull_long.shape[1] - 1

        for exp_idx in range(N_EXPERIMENTS_EIG):
            print(f"  experiment {exp_idx + 1}/{N_EXPERIMENTS_EIG} ...", end=" ", flush=True)
            rng = np.random.default_rng(seed=42 + exp_idx * 1000 + i_m)
            idx = rng.choice(total_snaps, size=m, replace=False)
            idx.sort()
            Xlist_s = Xfull_long[:, idx]

            basis_list_s = build_deterministic_rff_basis(P_DIMS, N_FEATURES, LENGTH_SCALE)
            phi_s = ProductBasis(basis_list_s)

            try:
                d_mat, *_ = spectral_analysis_gedmd_dense(
                    Xlist_s, phi_s, nev=200, diffusion_const=DIFFUSION_CONST, tol=1e-4, eps_ev=0.0
                )
                matrix_eigs_exp[i_m, exp_idx, :] = np.real(d_mat[:NUM_PLOT_EIGS])
            except Exception as e:
                print(f"[matrix ERR: {e}]", end=" ")

            try:
                d_tt_s, *_ = tensor_spectral_analysis(
                    Xlist_s, basis_list_s, R_TRUNC, 1e-9, nev=40
                )
                tensor_eigs_exp[i_m, exp_idx, :] = np.real(d_tt_s[:NUM_PLOT_EIGS])
            except Exception as e:
                print(f"[tensor ERR: {e}]", end=" ")

            gc.collect()
            print("done")

        del Xfull_long
        gc.collect()

    return {
        "m_values": np.array(M_VALUES_EIG),
        "mat_mean": np.nanmean(matrix_eigs_exp, axis=1),
        "mat_std": np.nanstd(matrix_eigs_exp, axis=1),
        "ten_mean": np.nanmean(tensor_eigs_exp, axis=1),
        "ten_std": np.nanstd(tensor_eigs_exp, axis=1),
    }


# ============================================================
# Section B.2: TT-rank vs SVD-tolerance sweep
# ============================================================

def run_rank_sweep(LS):
    rank_map_exp = np.zeros((len(M_VALUES_RANK), N_EXPERIMENTS_RANK, len(TOL_VALUES_RANK)), dtype=int)

    for i_m, m in enumerate(M_VALUES_RANK):
        print(f"\n[rank sweep] m = {m}")
        Xfull_long = LS.simulate(np.ones((NTEST, 3)), m * DSAVE * N_EXPERIMENTS_RANK, DT)[:, ::DSAVE]
        total_snaps = Xfull_long.shape[1] - 1

        for exp_idx in range(N_EXPERIMENTS_RANK):
            print(f"  experiment {exp_idx + 1}/{N_EXPERIMENTS_RANK}", end=" ", flush=True)
            rng = np.random.default_rng(seed=42 + exp_idx * 1000 + i_m)
            idx = rng.choice(total_snaps, size=m, replace=False)
            idx.sort()
            Xlist_r = Xfull_long[:, idx]

            basis_list_r = build_deterministic_rff_basis(P_DIMS, N_FEATURES, LENGTH_SCALE)
            psi_r, _ = evaluate_basis(Xlist_r, basis_list_r)
            data_tensor_r = Transformed_Data_Tensor_TT(psi=psi_r)

            for j_t, tol_svd in enumerate(TOL_VALUES_RANK):
                def core_getter(k, dt=data_tensor_r):
                    return dt.build_core(k + 1)

                U_cores_r, _, _ = global_svd_tt(core_getter, num_cores=P_DIMS + 1,
                                                 rmax=R_TRUNC, tol=tol_svd)
                rank_map_exp[i_m, exp_idx, j_t] = get_max_tt_rank(U_cores_r)
                data_tensor_r.clear_cache()

            gc.collect()
            print("done")

        del Xfull_long
        gc.collect()

    rank_mean = rank_map_exp.mean(axis=1)
    rank_std = rank_map_exp.std(axis=1)
    rank_se = rank_std / np.sqrt(N_EXPERIMENTS_RANK)

    return {
        "m_values_rank": np.array(M_VALUES_RANK),
        "tol_values": np.array(TOL_VALUES_RANK),
        "rank_mean": rank_mean,
        "rank_std": rank_std,
        "rank_se": rank_se,
    }


# ============================================================
# Section B.3: mat-vec runtime benchmark, block contraction vs direct
#
#   BLOCK CONTRACTION: builds the dense TT-format stiffness operator
#     (op.build_cores()), then applies it to each column of U_cores one at
#     a time via tensor_gedmd.algorithms.mat_vec_prod's legacy rank-2
#     "prepared blocks" matvec (tensor_gedmd.algorithms.mat_vec_prod.
#     prepare_blocks / tt_matrix_vector_product_csr_prepared), accumulating
#     A_r column by column via TT inner products.
#
#   DIRECT: tensor_gedmd.algorithms.mat_vec_prod_direct.compute_A_r, which
#     never builds the TT operator at all.
#
# Both are timed forming the SAME full reduced operator A_r, for an
# apples-to-apples comparison; both use Sigma=None (no diffusion tensor
# at all), matching the source notebook's TgStiffnessOperator(psi, dpsi)
# call with no Sigma argument.
# ============================================================

def _extract_middle_core_blocks(op: TgStiffnessOperator) -> List[np.ndarray]:
    """
    Per-sample (2, n, n, 2) blocks of the single middle TT core (P_DIMS==3),
    sliced out of the dense block-diagonal middle core built by
    op.build_cores(). This is exactly the information
    tensor_gedmd.algorithms.mat_vec_prod.prepare_blocks needs.
    """
    assert op.p == 3, "The block-contraction runtime benchmark assumes exactly one middle core (p=3)."
    tg_cores = op.build_cores()
    middle_core = tg_cores[1]  # shape (2m, n, n, 2m)
    m = op.m
    return [middle_core[2 * l:2 * l + 2, :, :, 2 * l:2 * l + 2] for l in range(m)]


def block_contraction_A_r(op: TgStiffnessOperator, U_cores: List[np.ndarray], r: int) -> np.ndarray:
    """Forms A_r = U^T A U column-by-column via the TT-format operator."""
    tg_cores = op.build_cores()
    blocks = _extract_middle_core_blocks(op)
    prepared = prepare_blocks(blocks, op.m)

    U_cols = [extract_tt_column(U_cores, i) for i in range(r)]
    A_r = np.zeros((r, r), dtype=float)
    for j, u_j in enumerate(U_cols):
        y_tt = tt_matrix_vector_product_csr_prepared(
            tg_cores, u_j, prepared, max_rank=500, tolerance=1e-10
        )
        for i, u_i in enumerate(U_cols):
            A_r[i, j] = tt_inner_product(u_i, y_tt)
    return A_r


def run_runtime_benchmark(LS):
    runtime_block = np.full((len(M_VALUES_RUNTIME), N_EXPERIMENTS_RUNTIME), np.nan)
    runtime_direct = np.full((len(M_VALUES_RUNTIME), N_EXPERIMENTS_RUNTIME), np.nan)

    for i_m, m in enumerate(M_VALUES_RUNTIME):
        print(f"\n{'=' * 70}\nRuntime benchmark -- m = {m}\n{'=' * 70}")
        Xfull_long = LS.simulate(np.ones((NTEST, 3)), m * DSAVE * N_EXPERIMENTS_RUNTIME, DT)[:, ::DSAVE]
        total_snaps = Xfull_long.shape[1] - 1

        for exp_idx in range(N_EXPERIMENTS_RUNTIME):
            print(f"  exp {exp_idx + 1}/{N_EXPERIMENTS_RUNTIME} ...", end=" ", flush=True)
            rng = np.random.default_rng(seed=1234 + exp_idx * 1000 + i_m)
            idx = rng.choice(total_snaps, size=m, replace=False)
            idx.sort()
            Xlist_bm = Xfull_long[:, idx]
            del idx, rng

            basis_list_bm = build_deterministic_rff_basis(P_DIMS, N_FEATURES, LENGTH_SCALE)
            psi_bm, dpsi_bm = evaluate_basis(Xlist_bm, basis_list_bm)
            del Xlist_bm, basis_list_bm

            data_tensor_bm = Transformed_Data_Tensor_TT(psi=psi_bm)

            def core_getter(k, dt=data_tensor_bm):
                return dt.build_core(k + 1)

            U_tt_bm, Sigma_bm, V_core_bm = global_svd_tt(
                core_getter, num_cores=P_DIMS + 1, rmax=R_TRUNC, tol=1e-9
            )
            U_cores_bm = U_tt_bm.cores
            data_tensor_bm.clear_cache()
            del Sigma_bm, V_core_bm
            r_bm = U_cores_bm[-1].shape[2]

            op_bm = TgStiffnessOperator(psi=psi_bm, dpsi=dpsi_bm, Sigma=None)

            try:
                t0 = time.perf_counter()
                A_r_block = block_contraction_A_r(op_bm, U_cores_bm, r_bm)
                runtime_block[i_m, exp_idx] = time.perf_counter() - t0
                del A_r_block
            except Exception as e:
                print(f"[block ERR: {e}]", end=" ")

            try:
                t0 = time.perf_counter()
                A_r_direct = compute_A_r(op_bm, U_cores_bm, r_bm, chunk_size=200,
                                         n_workers=4, r_cap=r_bm)
                runtime_direct[i_m, exp_idx] = time.perf_counter() - t0
                del A_r_direct
            except Exception as e:
                print(f"[direct ERR: {e}]", end=" ")

            del psi_bm, dpsi_bm, U_cores_bm, op_bm, data_tensor_bm
            gc.collect()

            b = runtime_block[i_m, exp_idx]
            d = runtime_direct[i_m, exp_idx]
            print(f"block={'n/a' if np.isnan(b) else f'{b:.3f}s'}  "
                  f"direct={'n/a' if np.isnan(d) else f'{d:.3f}s'}")

        del Xfull_long
        gc.collect()

    return {
        "m_values_runtime": np.array(M_VALUES_RUNTIME),
        "runtime_block_mean": np.nanmean(runtime_block, axis=1),
        "runtime_block_std": np.nanstd(runtime_block, axis=1),
        "runtime_direct_mean": np.nanmean(runtime_direct, axis=1),
        "runtime_direct_std": np.nanstd(runtime_direct, axis=1),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    LS = build_simulator()
    basis_list = build_deterministic_rff_basis(P_DIMS, N_FEATURES, LENGTH_SCALE)

    # Start from whatever's already saved (if anything), so a section you
    # skip this run keeps its previously computed results instead of being
    # lost.
    all_results = {}
    if RESULTS_PATH.exists():
        existing = np.load(RESULTS_PATH, allow_pickle=False)
        all_results = {key: existing[key] for key in existing.files}
        print(f"Found existing results at {RESULTS_PATH}, will merge into it.")

    if RUN_SECTION_A:
        print("=" * 70)
        print("Section A: main PCCA pipeline")
        print("=" * 70)
        all_results.update(run_main_pipeline(LS, basis_list))
    else:
        print("Section A: skipped (RUN_SECTION_A = False)")

    if RUN_SECTION_B1:
        print("\n" + "=" * 70)
        print("Section B.1: eigenvalue error-bar sweep (Matrix vs Tensor)")
        print("=" * 70)
        all_results.update(run_eigenvalue_sweep(LS))
    else:
        print("Section B.1: skipped (RUN_SECTION_B1 = False)")

    if RUN_SECTION_B2:
        print("\n" + "=" * 70)
        print("Section B.2: TT-rank vs SVD-tolerance sweep")
        print("=" * 70)
        all_results.update(run_rank_sweep(LS))
    else:
        print("Section B.2: skipped (RUN_SECTION_B2 = False)")

    if RUN_SECTION_B3:
        print("\n" + "=" * 70)
        print("Section B.3: mat-vec runtime benchmark (block contraction vs direct)")
        print("=" * 70)
        all_results.update(run_runtime_benchmark(LS))
    else:
        print("Section B.3: skipped (RUN_SECTION_B3 = False)")

    save_results(RESULTS_PATH, **all_results)


if __name__ == "__main__":
    main()
