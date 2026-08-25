"""
Introduction -- plot_introduction.py

Loads results/three_panel_result_data.npz (produced by
../introduction_run_calculation.py) and reproduces the 3-panel figure:
Lemon-Slice potential | Chignolin free-energy landscape | NTL9 free-energy
landscape.

No calculation happens here -- if results/ doesn't exist yet, run
../introduction_run_calculation.py first (see the error message this
raises). This script itself needs nothing beyond numpy/scipy/matplotlib --
no dmp_methods, no deeptime, no raw data.

Usage
-----
    python plot_introduction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "three_panel_result_data.npz"
CALC_SCRIPT_HINT = "examples/publication/introduction/introduction_run_calculation.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "three_panel_potential_FEL.pdf"

# Font sizes (kept large for a paper figure).
TITLE_FS = 18
LABEL_FS = 16
TICK_FS = 13
CBAR_FS = 13


# ============================================================
# Shared helper
# ============================================================

def free_energy_surface(x, y, bins=60, kT=1.0, smooth_sigma=1.0, pad=0.08):
    """
    Free-energy landscape F = -kT ln p(x, y) from a (lightly smoothed) 2D
    histogram. Returns F (rows=y, cols=x, masked where empty), bin centers.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    xr = float(x.max() - x.min()) or 1.0
    yr = float(y.max() - y.min()) or 1.0
    hist_range = [
        [x.min() - pad * xr, x.max() + pad * xr],
        [y.min() - pad * yr, y.max() + pad * yr],
    ]
    counts, xe, ye = np.histogram2d(x, y, bins=bins, range=hist_range)

    dens = counts.astype(float)
    if smooth_sigma and smooth_sigma > 0:
        dens = gaussian_filter(dens, sigma=smooth_sigma)

    area = (xe[1] - xe[0]) * (ye[1] - ye[0])
    total = dens.sum() * area
    p = dens / total if total > 0 else dens

    with np.errstate(divide="ignore"):
        F = -kT * np.log(p)
    F = np.ma.masked_invalid(F)
    if F.count() > 0:
        F = F - F.min()

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return F.T, xc, yc


def draw_fel_panel(fig, ax, X, kT, title, fel_cmap, fe_levels, fe_ticks,
                    fe_max, bins, smooth_sigma, pad, ix=0, iy=1):
    F, xc, yc = free_energy_surface(X[ix], X[iy], bins=bins, kT=kT,
                                     smooth_sigma=smooth_sigma, pad=pad)
    F = np.ma.masked_greater(F, fe_max)
    XC, YC = np.meshgrid(xc, yc)

    ax.set_facecolor("white")
    pc = ax.contourf(XC, YC, F, levels=fe_levels, cmap=fel_cmap, extend="neither")
    ax.contour(XC, YC, F, levels=fe_levels[::8], colors="k", linewidths=0.3, alpha=0.35)
    ax.set_xlim(xc.min(), xc.max())
    ax.set_ylim(yc.min(), yc.max())

    cb = fig.colorbar(pc, ax=ax, ticks=fe_ticks, fraction=0.046, pad=0.02)
    cb.set_label("Free energy (kJ/mol)", fontsize=CBAR_FS)
    cb.ax.tick_params(labelsize=CBAR_FS)

    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(rf"TICA $x_{{{ix + 1}}}$", fontsize=LABEL_FS)
    ax.set_ylabel(rf"TICA $x_{{{iy + 1}}}$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)

    X_lemon = results["X_lemon"]
    beta_lemon = float(results["beta_lemon"])
    GX, GY, V = results["GX_lemon"], results["GY_lemon"], results["V_lemon"]
    X_chig_full = results["X_chig_plot"]
    kT_chig = float(results["kT_chig"])
    X_ntl9 = results["X_ntl9"]
    kT_ntl9 = float(results["kT_ntl9"])

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.0))

    fel_cmap = plt.cm.jet.copy()
    fel_cmap.set_bad("white")

    SMOOTH_SIGMA = 1.0
    N_BINS = 70
    N_LEVELS = 40
    FE_MAX = 25.0
    PAD = 0.08

    fe_levels = np.linspace(0.0, FE_MAX, N_LEVELS + 1)
    fe_ticks = np.linspace(0.0, FE_MAX, 6)

    # ---- Panel 1 (left): Lemon-slice potential ----
    ax = axes[0]
    if not np.isnan(V).all():
        V = V - np.nanmin(V)
        pc = ax.contourf(GX, GY, V, levels=N_LEVELS, cmap="jet")
        ax.contour(GX, GY, V, levels=12, colors="k", linewidths=0.3, alpha=0.30)
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Potential", fontsize=CBAR_FS)
    else:
        F, xc, yc = free_energy_surface(X_lemon[0], X_lemon[1], bins=120,
                                         kT=1.0 / beta_lemon, smooth_sigma=0.0, pad=0.0)
        pc = ax.imshow(F, origin="lower", aspect="auto",
                        extent=[xc[0], xc[-1], yc[0], yc[-1]], cmap=fel_cmap)
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"Potential ($-\beta^{-1}\ln p$)", fontsize=CBAR_FS)
    cb.ax.tick_params(labelsize=CBAR_FS)
    ax.set_title("Lemon-slice potential", fontsize=TITLE_FS)
    ax.set_xlabel(r"$x_1$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$x_2$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)

    # ---- Panels 2 & 3: protein FELs ----
    draw_fel_panel(fig, axes[1], X_chig_full, kT_chig, "Chignolin free energy landscape",
                   fel_cmap, fe_levels, fe_ticks, FE_MAX, N_BINS, SMOOTH_SIGMA, PAD)
    draw_fel_panel(fig, axes[2], X_ntl9, kT_ntl9, "NTL9 free energy landscape (2D)",
                   fel_cmap, fe_levels, fe_ticks, FE_MAX, N_BINS, SMOOTH_SIGMA, PAD)

    plt.tight_layout()
    save_figure(fig, OUTPUT_PATH, also_png=True)
    plt.show()


if __name__ == "__main__":
    main()
