# Introduction Figure

Three-panel overview figure: Lemon-Slice (toy) potential | Chignolin
free-energy landscape | NTL9 free-energy landscape.

## Files

```
introduction/
├── README.md                          (this file)
├── introduction_run_calculation.py    loads raw data, computes the figure's inputs, saves results/
└── plot_introduction.py               loads results/, draws the figure -- no raw data needed
```

Running `introduction_run_calculation.py` creates `results/three_panel_result_data.npz`
locally (this folder doesn't ship with it -- `results/` isn't committed to
the repo, since it's regenerable from code + data).

## Reproducing the figure

**Step 1 -- generate the results file** (needs raw data + dependencies,
see below):

```bash
python introduction_run_calculation.py
```

**Step 2 -- draw the figure** (fast, needs only `numpy`/`scipy`/`matplotlib`
once `results/three_panel_result_data.npz` exists):

```bash
python plot_introduction.py
```

If you already have `three_panel_result_data.npz` from elsewhere (e.g.
downloaded from the project's Zenodo record instead of regenerating it
yourself), drop it into `results/` and skip straight to Step 2.

## Requirements

Requires `deeptime` (Chignolin TICA), plus the raw data below. The
Lemon-Slice `Decoupled_3d` simulator ships with `tensor_gedmd` itself --
no external simulator package needed.

## Raw data

This figure reuses the same raw trajectory data as `chignolin/` and
`ntl9/` -- it is **not duplicated here**. See those folders' own
`README.md` / `data/README.md` for where to get it and expected filenames:

- Chignolin: `cln_tica.pkl`
- NTL9: `ntl9_tica.npy`
- Lemon-Slice: no data file needed -- simulated fresh from the
  `Decoupled_3d` physics model, which is part of `tensor_gedmd`
  (`tensor_gedmd.systems`)

Point `introduction_run_calculation.py` at wherever you've placed these
files via environment variables (no need to edit the script):

```bash
CHIGNOLIN_TICA_PICKLE=/path/to/cln_tica.pkl \
NTL9_TICA_NPY=/path/to/ntl9_tica.npy \
python introduction_run_calculation.py
```
