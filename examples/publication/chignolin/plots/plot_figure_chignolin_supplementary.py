"""
Chignolin -- plot_figure_chignolin_supplementary.py

Loads results/chignolin_supplementary_results.npz (produced by
../supplementary.py) and reproduces the summary-grid figure:
    rows = TICA dimensions (3, 4, 6, 8, 10)
    cols = kappa_1 | kappa_2
    one curve per RFF bandwidth (sigma_omega), error bars = std over repeats
    y-limits fixed to (-15, 0) for both columns
    one shared legend at the bottom

No calculation happens here -- if results/ doesn't exist yet, run
../supplementary.py first.

Usage
-----
    python plot_figure_chignolin_supplementary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "chignolin_supplementary_results.npz"
CALC_SCRIPT_HINT = "examples/publication/chignolin/chignolin_supplementary.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "figure_chignolin_supplementary_summary_grid.pdf"

Y_LIMS_OVERRIDE = {"ev0_mean": (-15.0, 0.0), "ev1_mean": (-15.0, 0.0)}


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)

    tica_dims = results["tica_dims"]
    sigma_list = results["sigma_list"]
    svd_tols = results["svd_tols"]
    ev0_mean, ev0_std = results["ev0_mean"], results["ev0_std"]
    ev1_mean, ev1_std = results["ev1_mean"], results["ev1_std"]

    palette = plt.get_cmap("tab10").colors
    sigma_color = {s: palette[i % len(palette)] for i, s in enumerate(sigma_list)}

    ev_meta = [
        (ev0_mean, ev0_std, r'$\kappa_1$', "ev0_mean"),
        (ev1_mean, ev1_std, r'$\kappa_2$', "ev1_mean"),
    ]

    n_rows = len(tica_dims)
    xi = np.arange(len(svd_tols))
    t_lbl = [f"{t:.0e}" for t in svd_tols]

    fig, axes = plt.subplots(
        n_rows, 2, figsize=(14, 3.6 * n_rows), sharex=False, sharey=False,
        gridspec_kw={"hspace": 0.42, "wspace": 0.12},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, dim in enumerate(tica_dims):
        for col_idx, (mean_arr, std_arr, col_title, key) in enumerate(ev_meta):
            ax = axes[row_idx, col_idx]
            y_lo, y_hi = Y_LIMS_OVERRIDE[key]

            for i_sig, sigma in enumerate(sigma_list):
                means = mean_arr[row_idx, i_sig, :]
                stds = std_arr[row_idx, i_sig, :]
                ax.errorbar(
                    xi, means, yerr=stds, marker="o", color=sigma_color[sigma],
                    linewidth=2.0, markersize=6, capsize=3, elinewidth=1.1,
                    label=rf"$\sigma_\omega$={sigma:g}", zorder=4,
                )

            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.55)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(-0.5, len(svd_tols) - 0.5)
            ax.grid(True, alpha=0.28, axis="y")
            ax.set_xticks(xi)
            ax.set_xticklabels(t_lbl, rotation=35, ha="right", fontsize=8.5)
            ax.set_xlabel("SVD truncation thresholds", fontsize=9)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=13, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"Tica_dim = {int(dim)}\nEigenvalue", fontsize=9.5)

    fig.subplots_adjust(bottom=0.16 / (n_rows ** 0.3))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(sigma_list),
        bbox_to_anchor=(0.5, 0.005 / (n_rows ** 0.3)), fontsize=10, frameon=True,
    )

    save_figure(fig, OUTPUT_PATH, also_png=True)
    plt.show()


if __name__ == "__main__":
    main()
