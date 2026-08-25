"""
NTL9 -- plot_figure_ntl9_supplementary.py

Loads results/ntl9_supplementary_results.npz (produced by
../supplementary.py) and reproduces the one-page summary figure:
    rows = TICA dimensions (3, 4, 6, 8)
    cols = kappa_1 | kappa_2 | kappa_3
    one curve per RFF bandwidth (sigma_omega), error bars = std over
    subsamples
    y-limits fixed per column: kappa_1 (-1, 0), kappa_2 (-4, 0), kappa_3 (-4, 0)
    one shared legend at the bottom

No calculation happens here -- if results/ doesn't exist yet, run
../supplementary.py first.

Usage
-----
    python plot_figure_ntl9_supplementary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "ntl9_supplementary_results.npz"
CALC_SCRIPT_HINT = "examples/publication/ntl9/ntl9_supplementary.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "figure_ntl9_supplementary_onepage.pdf"

N_EIGS_PLOT = 3
FIXED_YLIMS = {
    0: (-1.0, 0.0),  # kappa_1
    1: (-4.0, 0.0),  # kappa_2
    2: (-4.0, 0.0),  # kappa_3
}
PANEL_W, PANEL_H = 4.0, 3.4


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)

    tica_dims = results["tica_dims"]
    sigma_list = results["sigma_list"]
    svd_tols = results["svd_tols"]
    ev_mean = results["ev_mean"]  # (n_dims, n_sigmas, n_tols, n_eigs_saved)
    ev_std = results["ev_std"]

    n_dims = len(tica_dims)
    n_sigma = len(sigma_list)

    x = np.arange(len(svd_tols))
    t_lbl = [f"{t:.0e}" for t in svd_tols]

    if n_sigma <= 10:
        colors = [plt.cm.tab10.colors[i % 10] for i in range(n_sigma)]
    else:
        colors = [plt.cm.viridis(v) for v in np.linspace(0, 0.92, n_sigma)]

    fig, axes = plt.subplots(
        n_dims, N_EIGS_PLOT,
        figsize=(PANEL_W * N_EIGS_PLOT, PANEL_H * n_dims),
        squeeze=False,
    )

    handles, labels = [], []
    for ri, dim in enumerate(tica_dims):
        for k in range(N_EIGS_PLOT):
            ax = axes[ri][k]
            for si, sigma in enumerate(sigma_list):
                ev_vals = ev_mean[ri, si, :, k]
                ev_stds = ev_std[ri, si, :, k]
                ev_stds = np.where(np.isfinite(ev_stds), ev_stds, 0.0)

                line = ax.errorbar(
                    x, ev_vals, yerr=ev_stds, marker="o", color=colors[si],
                    linewidth=1.8, markersize=5, capsize=2.5, elinewidth=1.0,
                    capthick=1.0, zorder=4, label=rf"$\sigma_\omega = {sigma:g}$",
                )[0]

                if ri == 0 and k == 0:
                    handles.append(line)
                    labels.append(rf"$\sigma_\omega = {sigma:g}$")

            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.grid(True, alpha=0.28)
            ax.set_xticks(x)
            ax.tick_params(bottom=False)

            if ri == n_dims - 1:
                ax.set_xticklabels(t_lbl, fontsize=8, rotation=25)
                ax.set_xlabel("SVD truncation thresholds", fontsize=9)
            else:
                ax.set_xticklabels([])

            if ri == 0:
                ax.set_title(rf"$\kappa_{{{k + 1}}}$", fontsize=11)
            if k == 0:
                ax.set_ylabel(f"Tica_dim = {int(dim)}\nEigenvalue", fontsize=10, fontweight="bold")

            lim = FIXED_YLIMS.get(k)
            if lim is not None:
                ax.set_ylim(*lim)

    fig.legend(handles, labels, loc="lower center", ncol=min(n_sigma, 8),
              bbox_to_anchor=(0.5, -0.012), fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    save_figure(fig, OUTPUT_PATH, also_png=True)
    plt.show()


if __name__ == "__main__":
    main()
