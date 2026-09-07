# Reproducing the paper's figures

This folder contains everything needed to reproduce every figure in the
paper, one subfolder per figure/case study:

```
publication/
├── introduction/      Figure 1: three-panel overview
├── lemon_slice_3d/     Figure 2: Lemon-Slice toy system
├── chignolin/           Figures 3-5 + appendix: chignolin
├── ntl9/                 Figures 6-7 + appendix: NTL9
└── common/                shared plotting/results-loading helpers (not run directly)
```

## The pattern every case study follows

Each case study is split into a **calculation** step (slow, does the real
work, saves a small results file) and a **plot** step (fast, just draws
the figure from that results file):

```bash
cd <case_study>
python <run_calculation_script>.py     # slow -- produces results/*.npz
python plots/<plot_script>.py          # fast -- reads that file, draws the figure
```

You only need to run the calculation step once per figure. If you just
want to re-draw a figure (e.g. tweak a color), the plot step alone is
enough, using the results file already in `results/`.

## Base / full script pairs

Several calculation scripts are further split into two files:

- **`..._base.py`** -- the actual idea being tested, with everything
  needed to run it *once*, on a single fixed configuration. Runnable
  standalone as a quick demo/sanity check, with no sweeping or repeating.
- **the plain script** (no `_base` suffix) -- imports from the base file
  and wraps it in the full sweep/repeat that the actual paper figure
  needs (e.g. 10 independent subsamples for error bars, or every
  dimension in a multi-dimension comparison).

If you only want to confirm the pipeline runs correctly on your data
before committing to a long full run, run the `_base.py` file directly
first.

## Data

Every raw data file and every precomputed results file below is on this
project's Zenodo record.

**Raw data -- needed only if you want to run the calculation scripts
yourself:**

| Case study | Files needed | Where to put them |
|---|---|---|
| `introduction/` | None -- loads a precomputed results file only, see below | -- |
| `lemon_slice_3d/` | None -- simulates its own data from a physics model built into `tensor_gedmd` itself | -- |
| `chignolin/` | `cln_tica.npy`, `cln_diff.npy` | `chignolin/data/` (or set `CHIGNOLIN_DATA_DIR` to point elsewhere) |
| `ntl9/` | `ntl9_tica.npy`, `ntl9_diff.npy` | `ntl9/data/` (or set `NTL9_DATA_DIR` to point elsewhere) |

Each script falls back to that case study's own `data/` folder by
default -- you only need the environment variable if your data lives
somewhere else. `introduction/`'s own figure reuses the same
Chignolin/NTL9 raw data above rather than duplicating it, so no separate
download is needed for that folder.

**Just want a figure, not the raw data?** Every case study also has a
small precomputed results file on Zenodo that lets you skip the
calculation step entirely and go straight to plotting -- e.g. download
`lemon_slice_3d_results.npz` and drop it in
`lemon_slice_3d/results/lemon_slice_3d_results.npz`, then just run that
folder's `plots/` script directly. **See each case study's own README for
the exact filename and path** -- they differ slightly between folders.

## Requirements

Beyond `tensor_gedmd` itself (`pip install -e .` from the repository
root), the case studies that touch real molecular dynamics data need:

```bash
pip install deeptime mdtraj jax jaxlib
```

`lemon_slice_3d/` and `introduction/`'s Lemon-Slice portion need nothing
beyond `tensor_gedmd` itself.