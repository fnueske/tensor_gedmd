"""
Chignolin -- plot_figure_chignolin_eigenvalues_vs_svd_threshold_and_tica_dimension.py

Loads results/chignolin_results.npz (produced by ../run_calculation.py)
and reproduces the 2x2 figure:
    (a)/(b) kappa_1, kappa_2 vs SVD truncation threshold, for a few TICA
             dimensions (mean +/- std over N_RUNS subsamples)
    (c)/(d) kappa_1, kappa_2 vs TICA dimension itself, at one fixed
             threshold (mean +/- std / std-error over N_RUNS subsamples)

No calculation happens here -- if results/ doesn't exist yet, run
../run_calculation.py first (see the error message this raises).

Usage
-----
    python plot_figure_chignolin_eigenvalues_vs_svd_threshold_and_tica_dimension.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "chignolin_results.npz"
CALC_SCRIPT_HINT = "examples/publication/chignolin/run_calculation_threshold_and_dim_sweep.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "figure_chignolin_eigenvalues_vs_svd_threshold_and_tica_dimension.pdf"


def stats_over_threshold(curve_dims, svd_tols, thr_vals, dim, which):
    """thr_vals: (n_curve_dims, N_RUNS, n_tols) array for kappa index `which`."""
    i_dim = list(curve_dims).index(dim)
    vals = thr_vals[i_dim]  # (N_RUNS, n_tols)
    mean, std, sem = [], [], []
    for i_tol in range(len(svd_tols)):
        v = vals[:, i_tol]
        v = v[np.isfinite(v)]
        mean.append(v.mean())
        std.append(v.std())
        sem.append(v.std() / np.sqrt(len(v)) if len(v) > 0 else np.nan)
    return np.array(svd_tols), np.array(mean), np.array(std), np.array(sem)


def stats_over_dim(tica_dims, dim_vals):
    """dim_vals: (n_tica_dims, N_RUNS) array for one kappa index."""
    mean, std, sem, raw = [], [], [], []
    for i_dim in range(len(tica_dims)):
        v = dim_vals[i_dim]
        v = v[np.isfinite(v)]
        mean.append(v.mean())
        std.append(v.std())
        sem.append(v.std() / np.sqrt(len(v)) if len(v) > 0 else np.nan)
        raw.append(v)
    return np.array(tica_dims), np.array(mean), np.array(std), np.array(sem), raw


def plot_combined_2x2(results, outfile, curve_dims=None, all_dims=None,
                       same_y_cd=False, annotate_thr=True, annot_fmt="{:.3f}",
                       ylim_ab=(-15, 0)):
    curve_dims = curve_dims if curve_dims is not None else results["curve_dims"]
    svd_tols = results["svd_tols"]
    all_dims = all_dims if all_dims is not None else results["tica_dims"]

    thr_k1, thr_k2 = results["thr_k1"], results["thr_k2"]
    dim_k1, dim_k2 = results["dim_k1"], results["dim_k2"]

    palette = ["#440154", "#1f9e89", "#e67e22", "#3b528b", "#5ec962"]
    dim_colors = {d: palette[i % len(palette)] for i, d in enumerate(curve_dims)}
    K1_COL, K2_COL = "#1f77b4", "#d62728"

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    (axA, axB), (axC, axD) = axes
    x = np.arange(len(svd_tols))
    tol_lbl = [f"{t:.0e}" for t in svd_tols]

    # ---- (a)/(b): kappa over threshold ----
    for which, ax, sym, thr_vals in [(0, axA, r"$\kappa_1$", thr_k1), (1, axB, r"$\kappa_2$", thr_k2)]:
        for j, dim in enumerate(curve_dims):
            tg, mu, sd, se = stats_over_threshold(curve_dims, svd_tols, thr_vals, dim, which)
            stat = {t: (m, s) for t, m, s in zip(tg, mu, sd)}
            xs = [xi for xi, t in zip(x, svd_tols) if t in stat]
            mus = [stat[t][0] for t in svd_tols if t in stat]
            sds = [stat[t][1] for t in svd_tols if t in stat]
            ax.errorbar(xs, mus, yerr=sds, fmt="o-", capsize=4, capthick=1.5,
                        elinewidth=1.6, markersize=6, linewidth=1.8,
                        color=dim_colors[dim], label=f"TICA_dim = {dim}")
            if annotate_thr:
                up = (j == len(curve_dims) - 1)
                for xi, m, s in zip(xs, mus, sds):
                    y_anchor, dy, va = (m + s, 5, "bottom") if up else (m - s, -5, "top")
                    ax.annotate(annot_fmt.format(m), xy=(xi, y_anchor),
                                xytext=(0, dy), textcoords="offset points",
                                ha="center", va=va, fontsize=6.8,
                                color=dim_colors[dim], fontweight="bold", clip_on=True)
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(tol_lbl, rotation=35, ha="right", fontsize=9)
        ax.set_xlabel("SVD truncation thresholds", fontsize=11)
        ax.set_ylabel("Eigenvalue (real part)", fontsize=11)
        ax.set_title(f"  {sym} ", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9, framealpha=0.9, title="mean +/- std")
        if ylim_ab is not None:
            ax.set_ylim(*ylim_ab)
        else:
            ax.margins(y=0.12)

    # ---- (c)/(d): kappa over TICA dim ----
    cd = []
    for which, ax, sym, col, dim_vals in [
        (0, axC, r"$\kappa_1$", K1_COL, dim_k1),
        (1, axD, r"$\kappa_2$", K2_COL, dim_k2),
    ]:
        dims, mu, sd, se, raw = stats_over_dim(all_dims, dim_vals)
        for i, dim in enumerate(dims):
            ax.scatter(np.full(len(raw[i]), dim), raw[i], color=col, alpha=0.22, s=18, zorder=2)
        ax.errorbar(dims, mu, yerr=sd, fmt="none", ecolor=col, elinewidth=2.0,
                    capsize=10, capthick=1.5, alpha=0.40, zorder=3, label="mean +/- std")
        ax.errorbar(dims, mu, yerr=se, fmt="o-", color=col, ecolor=col, linewidth=2.0,
                    markersize=9, elinewidth=2.5, capsize=5, capthick=2.0, zorder=4,
                    label="mean +/- std-error")
        for dim, m in zip(dims, mu):
            ax.annotate(annot_fmt.format(m), xy=(dim, m), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.set_xticks(dims)
        ax.set_xlabel("TICA_dimension", fontsize=11)
        ax.set_ylabel("Eigenvalue (real part)", fontsize=11)
        ax.set_title(f"  {sym}", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9, framealpha=0.9)
        cd.append((ax, raw))

    if same_y_cd:
        allv = np.concatenate([v for _, raw in cd for v in raw])
        lo, hi = allv.min(), allv.max()
        pad = 0.05 * (hi - lo or 1)
        for ax, _ in cd:
            ax.set_ylim(lo - pad, hi + pad)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_figure(fig, outfile, also_png=True)
    plt.show()
    plt.close()


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)
    plot_combined_2x2(results, OUTPUT_PATH, same_y_cd=False)


if __name__ == "__main__":
    main()
