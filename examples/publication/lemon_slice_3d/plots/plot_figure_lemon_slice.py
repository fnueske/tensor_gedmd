"""
Lemon Slice 3D -- plot_figure_lemon_slice.py

Loads results/lemon_slice_3d_results.npz (produced by ../run_calculation.py)
and reproduces the 2x2 combined figure:
    eigenvalues (Matrix vs Tensor) | Tensor PCCA
    mat-vec runtime (block contraction vs direct) | TT ranks

Refinements applied here vs. the original layout:
    - Lemon-Slice potential panel removed
    - eigenvalue panel moved from bottom-left to upper-left
    - new bottom-left panel: mat-vec runtime, block contraction vs direct

No calculation happens here -- if results/ doesn't exist yet, run
../run_calculation.py first (see the error message this raises).

Usage
-----
    python plot_figure_lemon_slice.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "lemon_slice_3d_results.npz"
CALC_SCRIPT_HINT = "examples/publication/lemon_slice_3d/run_calculation.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "figure_2_lemon_slice_combined.pdf"

TITLE_FS = 18
LABEL_FS = 16
TICK_FS = 14
LEG_FS = 13
CBAR_FS = 13


def draw_eig_panel(ax, results):
    m_vals = results["m_values"]
    mat_mean, mat_std = results["mat_mean"], results["mat_std"]
    ten_mean, ten_std = results["ten_mean"], results["ten_std"]

    n_eig_plot = 3   # kappa_4 omitted -> y-axis auto-adjusts to kappa_1..kappa_3
    eig_lbls = [r'$\kappa_1$', r'$\kappa_2$', r'$\kappa_3$']
    colors = plt.cm.tab10(range(n_eig_plot))
    ls_mat, ls_ten = '-', '--'
    marker_mat, marker_ten = 'o', 's'

    for j in range(n_eig_plot):
        c = colors[j]
        ax.errorbar(m_vals, mat_mean[:, j], yerr=mat_std[:, j],
                    fmt=f'{marker_mat}{ls_mat}', color=c,
                    capsize=4, linewidth=2, markersize=7)
        ax.errorbar(m_vals, ten_mean[:, j], yerr=ten_std[:, j],
                    fmt=f'{marker_ten}{ls_ten}', color=c,
                    capsize=4, linewidth=2, markersize=7)

    ax.set_xlabel('Number of data points (m)', fontsize=LABEL_FS)
    ax.set_ylabel('Eigenvalue', fontsize=LABEL_FS)
    ax.set_title('First 3 eigenvalues:\nMatrix vs Tensor', fontsize=TITLE_FS, fontweight='bold')
    ax.set_xticks(m_vals)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, linestyle='--', alpha=0.5)

    eig_handles = [Line2D([0], [0], color=colors[j], lw=2, marker=marker_mat,
                          markersize=7, label=eig_lbls[j]) for j in range(n_eig_plot)]
    style_handles = [
        Line2D([0], [0], color='k', lw=2, linestyle=ls_mat, marker=marker_mat,
               markersize=7, label='Matrix'),
        Line2D([0], [0], color='k', lw=2, linestyle=ls_ten, marker=marker_ten,
               markersize=7, label='Tensor'),
    ]
    ax.legend(handles=eig_handles + style_handles, fontsize=LEG_FS, ncol=2, loc='best')


def draw_pcca_tensor_panel(ax, results):
    Xlist = results["Xlist"]
    labels = results["labels_tensor_aligned"]
    K = int(results["K"])

    cmap = ListedColormap(plt.cm.tab10.colors[:K])
    norm_disc = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

    sc = ax.scatter(Xlist[0], Xlist[1], c=labels, s=12, cmap=cmap, norm=norm_disc)
    ax.set_title("Tensor PCCA", fontsize=TITLE_FS)
    ax.set_xlabel(r"$x_1$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$x_2$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    cb = plt.colorbar(sc, ax=ax, ticks=np.arange(K))
    cb.set_ticklabels([str(i) for i in range(K)])
    cb.ax.tick_params(labelsize=CBAR_FS)


def draw_runtime_panel(ax, results):
    m_rt = results["m_values_runtime"]
    runtime_block_mean = results["runtime_block_mean"]
    runtime_block_std = results["runtime_block_std"]
    runtime_direct_mean = results["runtime_direct_mean"]
    runtime_direct_std = results["runtime_direct_std"]

    ax.errorbar(m_rt, runtime_block_mean, yerr=runtime_block_std,
                fmt='o-', color='tab:blue', capsize=4, linewidth=2,
                markersize=7, label='Block contraction (TT format of A)')
    ax.errorbar(m_rt, runtime_direct_mean, yerr=runtime_direct_std,
                fmt='s--', color='tab:red', capsize=4, linewidth=2,
                markersize=7, label='Direct (no TT format of A)')

    ax.set_yscale('log')
    ax.set_xlabel('Number of data points (m)', fontsize=LABEL_FS)
    ax.set_ylabel(r'Runtime to form $\tilde{M}$ (sec, log scale)', fontsize=LABEL_FS)
    ax.set_title('Mat-vec runtime:\nBlock contraction vs Direct',
                 fontsize=TITLE_FS, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(True, linestyle='--', alpha=0.5, which='both')
    ax.legend(fontsize=LEG_FS, loc='best')


def draw_rank_panel(ax, results):
    m_values_rank = results["m_values_rank"]
    tol_values = results["tol_values"]
    rank_mean, rank_std, rank_se = results["rank_mean"], results["rank_std"], results["rank_se"]

    colors_r = plt.cm.tab10(range(len(m_values_rank)))
    for i_m, m in enumerate(m_values_rank):
        c = colors_r[i_m]
        ax.errorbar(tol_values, rank_mean[i_m], yerr=rank_se[i_m],
                    fmt='o-', color=c, capsize=4, linewidth=2,
                    markersize=7, label=f'm={int(m)}')
        ax.fill_between(tol_values,
                        rank_mean[i_m] - rank_std[i_m],
                        rank_mean[i_m] + rank_std[i_m],
                        color=c, alpha=0.10)
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Truncation Thresholds in Global SVD', fontsize=LABEL_FS)
    ax.set_ylabel('Max TT rank', fontsize=LABEL_FS)
    ax.set_title('TT rank vs\nTruncation Thresholds', fontsize=TITLE_FS, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=LEG_FS)


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    draw_eig_panel(axes[0, 0], results)
    draw_pcca_tensor_panel(axes[0, 1], results)
    draw_runtime_panel(axes[1, 0], results)
    draw_rank_panel(axes[1, 1], results)

    plt.tight_layout()
    save_figure(fig, OUTPUT_PATH, also_png=True)
    plt.show()


if __name__ == "__main__":
    main()
