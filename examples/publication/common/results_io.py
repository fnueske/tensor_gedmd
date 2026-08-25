"""
Shared save/load helpers for the publication figures.

The pattern used throughout examples/publication/:
    run_calculation.py   -> computes everything, calls save_results(...)
    plots/plot_figure_*.py -> calls load_results(...), never recomputes

This keeps "give me the plots" fast (pure deserialization) and fully
decoupled from "run the actual calculation" (slow, needs the real data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_results(path: str | Path, **arrays: Any) -> None:
    """
    Save named arrays/scalars to a .npz file, creating parent directories
    as needed.

    Example
    -------
    save_results(
        "results/lemon_slice_3d_results.npz",
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        lag_times=lag_times,
    )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[io] saved results -> {path}")


def load_results(path: str | Path, *, calc_script_hint: str | None = None) -> Dict[str, np.ndarray]:
    """
    Load a .npz results file into a plain dict.

    Parameters
    ----------
    path : str or Path
        Path to the .npz file written by save_results().
    calc_script_hint : str, optional
        Path to the run_calculation.py script that produces this file, used
        only to build a clearer error message if the file is missing.

    Raises
    ------
    FileNotFoundError
        With a message telling the user which script to run first, rather
        than a bare "file not found".
    """
    path = Path(path)
    if not path.exists():
        hint = (
            f" Run `{calc_script_hint}` first to generate it."
            if calc_script_hint
            else " Run the corresponding run_calculation.py first to generate it."
        )
        raise FileNotFoundError(f"Results file not found: {path}.{hint}")

    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def require_data_file(path: str | Path, *, zenodo_readme_hint: str | None = None) -> Path:
    """
    Check that an expected raw-data file exists, raising a clear error
    pointing at the case study's data/README.md (Zenodo instructions) if not.

    Use this at the top of run_calculation.py before touching the data.
    """
    path = Path(path)
    if not path.exists():
        hint = (
            f" See {zenodo_readme_hint} for the Zenodo download link and "
            "expected file layout."
            if zenodo_readme_hint
            else " See this case study's data/README.md for the Zenodo download link."
        )
        raise FileNotFoundError(f"Expected data file not found: {path}.{hint}")
    return path
