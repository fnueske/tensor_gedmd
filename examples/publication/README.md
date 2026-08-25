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

Raw trajectory/simulation data and the small cached results files are
available on Zenodo.

- `lemon_slice_3d/` needs no external data -- it simulates its own data
  from a physics model built into `tensor_gedmd` itself.
- `chignolin/` and `ntl9/` need raw data -- see each folder's
  `data/README.md`.
- `introduction/` reuses the same raw data as `chignolin/`/`ntl9/`
  rather than duplicating it.

Point each script at your data via the environment variables described in
that case study's own README (or accept the defaults, which expect the
data inside that case's own `data/` folder).

## Requirements

Beyond `tensor_gedmd` itself (`pip install -e .` from the repository
root), the case studies that touch real molecular dynamics data need:

```bash
pip install deeptime mdtraj jax jaxlib
```

`lemon_slice_3d/` and `introduction/`'s Lemon-Slice portion need nothing
beyond `tensor_gedmd` itself.
