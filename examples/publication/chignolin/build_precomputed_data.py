"""
Chignolin -- build_precomputed_data.py

Run this ONCE, locally, where you have the full raw data (including the
large cln_atomic.pkl). Produces two much smaller files:

    cln_tica.npy  -- TICA coordinates, shape (10, N_total)
    cln_diff.npy  -- diffusion tensor projected onto all 10 TICA
                     directions, shape (10, 10, N_total)

Fits TICA once at the maximum useful dimension (10) instead of separately
per dimension -- validated (on this project's actual data) to give
bit-for-bit identical results to fitting directly at a smaller dimension
and slicing: e.g. dim=10's leading 3 components exactly match a direct
dim=3 fit (max abs difference: 0.0). So any downstream script can just
slice cln_tica.npy[:dim, :] / cln_diff.npy[:dim, :dim, :] for whichever
TICA dimension it needs, instead of re-fitting TICA or re-running the
(expensive) Jacobian from scratch.

This mirrors ntl9_tica.npy / ntl9_diff.npy exactly -- once these two files
exist, chignolin's calculation scripts can load them the same lightweight
way NTL9's already do, with no jax/mdtraj/deeptime needed downstream.

Requirements
------------
`deeptime`, `mdtraj`, `jax` importable, and the full raw data: cln_tica.pkl
(trajectories), cln_atomic.pkl (raw atomic coordinates -- the large file),
chi_vac.pdb (topology).

Usage
-----
    CHIGNOLIN_DATA_DIR=/path/to/your/raw/data python build_precomputed_data.py
"""

from __future__ import annotations

import gc
import itertools
import os
import pickle
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get(
    "CHIGNOLIN_DATA_DIR", str(Path(__file__).resolve().parent / "data")
))
CG_PICKLE_PATH = Path(os.environ.get("CHIGNOLIN_TICA_PICKLE", str(DATA_DIR / "cln_tica.pkl")))
CG_ATOMIC_NUMBERS_PATH = Path(os.environ.get("CHIGNOLIN_ATOMIC_PATH", str(DATA_DIR / "cln_atomic.pkl")))
PDB_PATH = Path(os.environ.get("CHIGNOLIN_PDB_PATH", str(DATA_DIR / "chi_vac.pdb")))

OUT_DIR = Path(os.environ.get("CHIGNOLIN_OUTPUT_DIR", str(DATA_DIR)))
TICA_OUT_PATH = OUT_DIR / "cln_tica.npy"
DIFF_OUT_PATH = OUT_DIR / "cln_diff.npy"

LAGTIME = 10
MAX_DIM = 10  # fit TICA at this dimension; every downstream script slices
              # to whatever smaller dim it actually needs.

# Physical constants (matches every chignolin calc script) -- applied here
# so the saved file is the final, ready-to-use diffusion tensor, exactly
# matching ntl9_diff.npy's convention (downstream code there only slices
# and standardizes, never re-applies a physics prefactor).
M_MASS = 12.0
T_K = 300.0
GAMMA = 0.1
K_B = 8.314462618e-3
BETA = 1.0 / (K_B * T_K)


def main() -> None:
    import mdtraj as md
    import jax.numpy as jnp
    from jax import jacrev, vmap
    from deeptime import decomposition

    print("Loading raw data ...")
    with open(CG_PICKLE_PATH, "rb") as f:
        trajectories = list(pickle.load(f)["x"])
    print(f"  {len(trajectories)} trajectories (first: {trajectories[0].shape})")

    top = md.load_topology(PDB_PATH)
    CA_atoms = top.select("name CA")
    print(f"  {len(CA_atoms)} CA atoms")

    with open(CG_ATOMIC_NUMBERS_PATH, "rb") as f:
        x_atomic = np.concatenate(
            [d.xyz[:, CA_atoms] for d in pickle.load(f)["x"]], axis=0
        )
    print(f"  x_atomic: {x_atomic.shape}")

    CA_comb = np.array(list(itertools.combinations(np.arange(len(CA_atoms)), 2)))

    def distances(t):
        return jnp.linalg.norm(t[CA_comb[:, 1]] - t[CA_comb[:, 0]], axis=1)

    def jac(x):
        return jacrev(distances, argnums=0)(x)

    print("Jacobian (vmap) ...", end="", flush=True)
    t0 = time.perf_counter()
    diff_full = np.asarray(vmap(jac, in_axes=(0,), out_axes=0)(x_atomic).transpose(1, 2, 3, 0))
    print(f" done ({time.perf_counter() - t0:.1f}s)  shape: {diff_full.shape}")
    del x_atomic
    gc.collect()

    print(f"\nFitting TICA at dim={MAX_DIM} ...")
    tica = decomposition.TICA(lagtime=LAGTIME, dim=MAX_DIM)
    tica_model = tica.fit(trajectories).fetch_model()
    tica_data_full = np.concatenate(tica_model.transform(trajectories), axis=0).T  # (MAX_DIM, N_total)
    eigs = tica_model.singular_values
    ind_sort = np.argsort(eigs)[::-1]
    eigvec_full = tica_model.singular_vectors_left[:, ind_sort][:, :MAX_DIM]
    print(f"  tica_data_full: {tica_data_full.shape}")

    print("Projecting diffusion tensor onto all TICA directions ...")
    dtr_full = np.einsum("ijlr,ik->kjlr", diff_full, eigvec_full)
    diff_projected = (2.0 / BETA / M_MASS / GAMMA) * np.einsum(
        "gikl,hikl->ghl", dtr_full, dtr_full
    )  # (MAX_DIM, MAX_DIM, N_total) -- final, ready-to-use diffusion tensor,
       # matching ntl9_diff.npy's convention (prefactor already applied).
    print(f"  diff_projected: {diff_projected.shape}")
    del diff_full, dtr_full
    gc.collect()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TICA_OUT_PATH, tica_data_full)
    np.save(DIFF_OUT_PATH, diff_projected)
    print(f"\nSaved:\n  {TICA_OUT_PATH}  ({tica_data_full.nbytes / 1e6:.1f} MB)")
    print(f"  {DIFF_OUT_PATH}  ({diff_projected.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()