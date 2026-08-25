"""
NTL9 -- plot_figure_ntl9_eigenvalues_scratch_vs_incremental.py

Loads results/ntl9_scratch_vs_incremental_results.npz (produced by
../run_calculation_scratch_vs_incremental.py) and reproduces the single-panel
figure: -Re(kappa) (log scale) vs TICA dimension, scratch (solid, full error
bars) vs incremental (dashed, overlaid), for kappa_1..kappa_4.

No calculation happens here -- if results/ doesn't exist yet, run
../run_calculation_scratch_vs_incremental.py first.

Usage
-----
    python plot_figure_ntl9_eigenvalues_scratch_vs_incremental.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = (Path(__file__).resolve().parents[1] / "results"
                / "ntl9_scratch_vs_incremental_results.npz")
CALC_SCRIPT_HINT = "examples/publication/ntl9/run_calculation_scratch_vs_incremental.py"
OUTPUT_PATH = (Path(__file__).resolve().parent / "output"
               / "figure_ntl9_eigenvalues_scratch_vs_incremental.pdf")

N_SHOW = 4
KCOLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _ms(arr):
    """mean, std, std-error across the run axis (axis 0)."""
    mu = arr.mean(0)
    sd = arr.std(0)
    se = sd / np.sqrt(arr.shape[0])
    return mu, sd, se


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)

    dims = results["dims"]
    # scratch_runs / inc_runs: (n_dims, N_RUNS, N_SHOW)
    scratch_runs = {int(d): results["scratch_runs"][i] for i, d in enumerate(dims)}
    inc_runs = {int(d): results["inc_runs"][i] for i, d in enumerate(dims)}
    dims_list = sorted(scratch_runs.keys())

    DY = {0: +10, 1: -14, 2: +10, 3: +10}
    LBL = dict(fontsize=8, ha="center", zorder=7,
               bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    allmag = []
    for j in range(N_SHOW):
        c = KCOLORS[j]
        smu = np.array([_ms(scratch_runs[d])[0][j] for d in dims_list])
        ssd = np.array([_ms(scratch_runs[d])[1][j] for d in dims_list])
        sse = np.array([_ms(scratch_runs[d])[2][j] for d in dims_list])
        imu = np.array([_ms(inc_runs[d])[0][j] for d in dims_list])

        mags, magi = -smu, -imu
        allmag += list(mags)

        ax.errorbar(dims_list, mags, yerr=ssd, fmt="none", ecolor=c, elinewidth=1.4,
                    capsize=7, capthick=1.2, alpha=0.30, zorder=2)
        ax.errorbar(dims_list, mags, yerr=sse, fmt="o-", color=c, linewidth=2.0, markersize=8,
                    elinewidth=2.2, capsize=4, capthick=1.8, markeredgecolor="k",
                    markeredgewidth=0.5, zorder=4, label=rf"$\kappa_{j + 1}$")
        ax.plot(dims_list, magi, "--s", color=c, linewidth=1.8, markersize=7, markerfacecolor="none",
                markeredgewidth=1.4, alpha=0.95, zorder=5)

        for x, m, M in zip(dims_list, smu, mags):
            ax.annotate(f"{m:.3f}", (x, M), textcoords="offset points",
                        xytext=(0, DY[j]), color=c, **LBL)

    ax.set_yscale("log")
    ax.set_ylim(min(allmag) * 0.65, max(allmag) * 1.8)
    ax.set_xticks(dims_list)
    ax.set_xticklabels([str(d) for d in dims_list])
    ax.set_xlabel("TICA_dimensions", fontsize=12)
    ax.set_ylabel(r"$-\mathrm{Re}(\kappa)$  (log scale)", fontsize=12)
    ax.grid(True, which="both", alpha=0.25)

    kappa_handles = [Line2D([0], [0], color=KCOLORS[j], lw=2.4, marker="o", markeredgecolor="k",
                            markeredgewidth=0.5, label=rf"$\kappa_{j + 1}$") for j in range(N_SHOW)]
    method_handles = [Line2D([0], [0], color="k", lw=2.2, ls="-", marker="o", label="scratch"),
                       Line2D([0], [0], color="k", lw=1.8, ls="--", marker="s",
                              markerfacecolor="none", label="incremental")]
    leg1 = ax.legend(handles=kappa_handles, fontsize=10, loc="upper right", ncol=2, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, fontsize=10, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, OUTPUT_PATH, also_png=True)
    plt.show()


if __name__ == "__main__":
    main()
