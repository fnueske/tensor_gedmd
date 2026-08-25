"""
Shared plotting style for all publication figures.

Every plot_figure_*.py script should call set_publication_style() once at
the top, before creating any figure, so figures from different case
studies look consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# TODO: tune these to match the paper's actual figure style once the
# notebooks are in hand (font, sizes, colors may differ from the drafts).
# ----------------------------------------------------------------------

FIGURE_WIDTH_SINGLE_COLUMN = 3.4   # inches
FIGURE_WIDTH_DOUBLE_COLUMN = 7.0   # inches

# A small, consistent, colorblind-friendly palette for eigenvalue /
# eigenfunction curves shared across all three case studies.
PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
]


def set_publication_style() -> None:
    """Apply consistent rcParams for all publication figures."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed fonts as TrueType, not bitmap, in PDFs
        "ps.fonttype": 42,
    })


def save_figure(fig, output_path, *, also_png: bool = False) -> None:
    """
    Save a figure to output_path (creating parent directories as needed),
    printing where it went. If also_png, additionally saves a .png next to
    the primary file (handy for quickly previewing without a PDF viewer).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"[plotting] saved figure -> {output_path}")

    if also_png and output_path.suffix.lower() != ".png":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path)
        print(f"[plotting] saved figure -> {png_path}")
