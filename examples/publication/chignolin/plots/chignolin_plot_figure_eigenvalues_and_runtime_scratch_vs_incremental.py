"""
Chignolin -- plot_figure_chignolin_eigenvalues_and_runtime_scratch_vs_incremental.py

Loads results/chignolin_scratch_vs_incremental_results.npz (produced by
../run_calculation_scratch_vs_incremental.py) and reproduces the 1x2 figure:
    (a) kappa_1, kappa_2 vs TICA dimension, scratch vs incremental
    (b) wall-clock runtime vs TICA dimension, scratch vs incremental (linear scale)

No calculation happens here -- if results/ doesn't exist yet, run
../run_calculation_scratch_vs_incremental.py first (see the error message
this raises).

Usage
-----
    python plot_figure_chignolin_eigenvalues_and_runtime_scratch_vs_incremental.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = (Path(__file__).resolve().parents[1] / "results"
                / "chignolin_scratch_vs_incremental_results.npz")
CALC_SCRIPT_HINT = "examples/publication/chignolin/run_calculation_scratch_vs_incremental.py"
OUTPUT_PATH = (Path(__file__).resolve().parent / "output"
               / "figure_chignolin_eigenvalues_and_runtime_scratch_vs_incremental.pdf")

SCRATCH_COLOR = "#1f77b4"    # blue
INCREMENTAL_COLOR = "#d62728"  # red


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)

    dims = results["dims"]
    full_ev0_m, full_ev0_s = np.nanmean(results["full_ev0"], axis=1), np.nanstd(results["full_ev0"], axis=1)
    full_ev1_m, full_ev1_s = np.nanmean(results["full_ev1"], axis=1), np.nanstd(results["full_ev1"], axis=1)
    upd_ev0_m, upd_ev0_s = np.nanmean(results["upd_ev0"], axis=1), np.nanstd(results["upd_ev0"], axis=1)
    upd_ev1_m, upd_ev1_s = np.nanmean(results["upd_ev1"], axis=1), np.nanstd(results["upd_ev1"], axis=1)
    full_time_m, full_time_s = np.nanmean(results["full_time"], axis=1), np.nanstd(results["full_time"], axis=1)
    upd_time_m, upd_time_s = np.nanmean(results["upd_time"], axis=1), np.nanstd(results["upd_time"], axis=1)

    fig, (ax_ev, ax_t) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Panel (a): eigenvalues kappa_1, kappa_2 ----
    ax_ev.errorbar(dims, full_ev0_m, yerr=full_ev0_s, color=SCRATCH_COLOR,
                   linestyle="-", marker="o", linewidth=2, markersize=8,
                   markeredgecolor="k", markeredgewidth=0.5, capsize=3,
                   label=r"$\kappa_1$ scratch")
    ax_ev.errorbar(dims, upd_ev0_m, yerr=upd_ev0_s, color=INCREMENTAL_COLOR,
                   linestyle="--", marker="s", linewidth=2, markersize=8,
                   markeredgecolor="k", markeredgewidth=0.5, capsize=3,
                   label=r"$\kappa_1$ incremental")
    ax_ev.errorbar(dims, full_ev1_m, yerr=full_ev1_s, color=SCRATCH_COLOR,
                   linestyle="-", marker="^", linewidth=2, markersize=8,
                   markeredgecolor="k", markeredgewidth=0.5, capsize=3,
                   alpha=0.65, label=r"$\kappa_2$ scratch")
    ax_ev.errorbar(dims, upd_ev1_m, yerr=upd_ev1_s, color=INCREMENTAL_COLOR,
                   linestyle="--", marker="D", linewidth=2, markersize=7,
                   markeredgecolor="k", markeredgewidth=0.5, capsize=3,
                   alpha=0.65, label=r"$\kappa_2$ incremental")
    ax_ev.set_title(r"Eigenvalues $\kappa_1,\ \kappa_2$ vs TICA dimensions", fontsize=13)
    ax_ev.set_xlabel("TICA_dimension", fontsize=12)
    ax_ev.set_ylabel("Eigenvalue", fontsize=12)
    ax_ev.set_xticks(dims)
    ax_ev.set_xticklabels([str(d) for d in dims], fontsize=11)
    ax_ev.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax_ev.grid(True, alpha=0.3)
    ax_ev.legend(fontsize=10, ncol=2, loc="best")

    # ---- Panel (b): wall-clock time (linear) ----
    ax_t.errorbar(dims, full_time_m, yerr=full_time_s, color=SCRATCH_COLOR,
                  linestyle="-", marker="o", linewidth=2, markersize=8,
                  markeredgecolor="k", markeredgewidth=0.5, capsize=3, label="scratch")
    ax_t.errorbar(dims, upd_time_m, yerr=upd_time_s, color=INCREMENTAL_COLOR,
                  linestyle="--", marker="s", linewidth=2, markersize=8,
                  markeredgecolor="k", markeredgewidth=0.5, capsize=3, label="incremental")
    ax_t.set_title("Wall-clock time: scratch vs incremental", fontsize=13)
    ax_t.set_xlabel("TICA_dimension", fontsize=12)
    ax_t.set_ylabel("Runtime (seconds)", fontsize=12)
    ax_t.set_xticks(dims)
    ax_t.set_xticklabels([str(d) for d in dims], fontsize=11)
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=11, loc="best")
    ax_t.set_yscale("linear")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, OUTPUT_PATH, also_png=True)
    plt.show()


if __name__ == "__main__":
    main()
