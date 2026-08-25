"""
Introduction -- run_calculation.py

Loads the raw Lemon-Slice/Chignolin/NTL9 data, computes the (cheap)
derived quantities the introduction figure needs -- the Lemon-Slice
potential grid, and the TICA coordinates + kT for each protein's free-energy
surface -- and saves them to results/. Does NOT plot anything; see
plot_introduction.py for that.

Added so introduction/ follows the same run -> save results -> plot split
as the other three case studies: someone who only wants to re-draw the
figure (e.g. tweak a font size or color) doesn't need deeptime or any raw
data at all -- just the small results file this produces.

Requirements
------------
- The `tensor_gedmd` package (this repository) importable -- includes the
  Lemon-Slice `Decoupled_3d` simulator (tensor_gedmd.systems), no external
  simulator package needed.
- `deeptime` importable (for the Chignolin TICA).
- Data files: same raw data as chignolin/ and ntl9/ (see those folders'
  data/README.md) -- not duplicated here.

Usage
-----
    python introduction_run_calculation.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from results_io import save_results  # noqa: E402
from tensor_gedmd.systems.decoupled_potential_3d import Decoupled_3d

# ----------------------------------------------------------------------
# Paths. Defaults reuse the same data files as the case-study folders
# (../chignolin/data/, ../ntl9/data/) rather than duplicating them here --
# see each folder's data/README.md for the Zenodo download link. Override
# via environment variables if your data lives elsewhere; no need to edit
# this file.
# ----------------------------------------------------------------------
_PUBLICATION_ROOT = Path(__file__).resolve().parents[1]

CHIGNOLIN_TICA_PICKLE = os.environ.get(
    "CHIGNOLIN_TICA_PICKLE", str(_PUBLICATION_ROOT / "chignolin" / "data" / "cln_tica.pkl")
)
NTL9_TICA_NPY = os.environ.get(
    "NTL9_TICA_NPY", str(_PUBLICATION_ROOT / "ntl9" / "data" / "ntl9_tica.npy")
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "three_panel_result_data.npz"


# ============================================================
# Section 1 -- Lemon-Slice (toy) data + potential grid
# ============================================================

def load_lemon_slice():
    kk, ll, z, beta = 1.0, 1.0, 5.0, 1.0
    LS = Decoupled_3d(beta, kk, ll, z)

    ntest = 1
    x0 = np.ones((ntest, 3))
    dt = 1e-3
    dsave = 20
    m = 7000

    Xfull = LS.simulate(x0, m * dsave, dt)[:, ::dsave]
    Xlist = Xfull[:, :-1]

    print("Lemon-slice trajectory:", Xlist.shape)
    return Xlist, LS, beta


def lemon_slice_potential_grid(LS, X, grid_n=300):
    """
    Evaluate the Lemon-Slice potential on an (x1, x2) grid (x3 fixed at 0).
    Returns V filled with NaN (not None -- needs to survive an .npz
    round-trip) if the simulator doesn't expose a recognized potential
    method; the plot script falls back to a density estimate in that case.
    """
    xmin, xmax = X[0].min(), X[0].max()
    ymin, ymax = X[1].min(), X[1].max()
    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    GX, GY = np.meshgrid(gx, gy)

    V = None
    for meth in ("potential", "V", "energy", "compute_potential", "pot", "U"):
        fn = getattr(LS, meth, None)
        if callable(fn):
            try:
                pts = np.vstack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
                V = np.asarray(fn(pts)).reshape(GX.shape)
                break
            except Exception:
                V = None
    if V is None:
        V = np.full(GX.shape, np.nan)
    return GX, GY, V


# ============================================================
# Section 2 -- Chignolin data
# ============================================================

def load_chignolin():
    from deeptime import decomposition

    tica_dim = 4
    T = 300.0
    K_B = 8.314462 * 1e-3
    kT = K_B * T

    with open(CHIGNOLIN_TICA_PICKLE, "rb") as f:
        cg_data = pickle.load(f)
    trajectories = list(cg_data["x"])
    print("Number of trajectories:", len(trajectories))

    lagtime = 10
    tica = decomposition.TICA(lagtime=lagtime, dim=tica_dim)
    tica_model = tica.fit(trajectories).fetch_model()
    tica_data = tica_model.transform(trajectories)
    tica_data = tica_data.reshape(-1, tica_dim).T

    X_chig_full = tica_data  # full, unstrided -> well-sampled FEL; the only
    # one actually plotted (the strided version the gEDMD method uses isn't
    # needed for this figure).

    print("Chignolin TICA shape (full):", X_chig_full.shape)
    return X_chig_full, kT


# ============================================================
# Section 3 -- NTL9 data
# ============================================================

def load_ntl9():
    tica_dim = 4
    tica_data = np.load(NTL9_TICA_NPY, allow_pickle=True)

    start, end, stride = 109267, 128720, 3
    Xlist = tica_data[:tica_dim, start:end:stride]

    s = Xlist.std(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    Xlist = Xlist / s

    T = 300.0
    K_B = 8.314462 * 1e-3
    kT = K_B * T

    print("NTL9 Xlist shape:", Xlist.shape)
    return Xlist, kT


# ============================================================
# Main
# ============================================================

def main() -> None:
    X_lemon, LS_lemon, beta_lemon = load_lemon_slice()
    GX, GY, V = lemon_slice_potential_grid(LS_lemon, X_lemon)

    X_chig_full, kT_chig = load_chignolin()
    X_ntl9, kT_ntl9 = load_ntl9()

    save_results(
        RESULTS_PATH,
        X_lemon=X_lemon,
        beta_lemon=np.array(beta_lemon),
        GX_lemon=GX,
        GY_lemon=GY,
        V_lemon=V,
        X_chig_plot=X_chig_full,
        kT_chig=np.array(kT_chig),
        X_ntl9=X_ntl9,
        kT_ntl9=np.array(kT_ntl9),
    )


if __name__ == "__main__":
    main()
