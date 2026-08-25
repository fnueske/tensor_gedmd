# Lemon Slice 3D

Toy 3D "decoupled" potential (an isotropic double-well in the first two
coordinates, a harmonic well in the third). Produces a 2x2 figure:
eigenvalues (Matrix vs Tensor method, with error bars over 5 data sizes),
PCCA cluster scatter, mat-vec runtime benchmark (block contraction vs
direct), and TT-rank vs SVD-truncation-tolerance.

## Files

```
lemon_slice_3d/
├── README.md                    (this file)
├── run_calculation_base.py       the idea: one Matrix-vs-Tensor eigenvalue
│                                  comparison + one TT-rank measurement, at
│                                  fixed parameters -- run standalone for a
│                                  quick demo, no plot
├── run_calculation.py            imports from the base file; runs the main
│                                  PCCA pipeline, the full 10-experiment
│                                  eigenvalue sweep, the 10-experiment rank
│                                  sweep, and the runtime benchmark
├── results/
│   └── lemon_slice_3d_results.npz
└── plots/
    └── plot_figure_lemon_slice.py
```

## No external data needed

Unlike the other case studies, this one simulates its own data on the fly
-- the Lemon-Slice SDE simulator (`Decoupled_3d`) ships with `tensor_gedmd`
itself (`tensor_gedmd.systems`). No raw data files, no Zenodo download
needed for this figure.

## Running it

```bash
cd lemon_slice_3d
python run_calculation.py       # slow -- see the section toggles below
python plots/plot_figure_lemon_slice.py
```

**This is the heaviest script in `publication/`** -- the eigenvalue sweep
alone is 50 solves (5 data sizes x 10 experiments), the rank sweep is 350
(5 data sizes x 10 experiments x 7 tolerances). Two things make this
easier to work with:

**Quick smoke test first.** Set `QUICK_TEST = True` near the top of
`run_calculation.py` to run everything with tiny data sizes and 1-2
experiments, before committing to the full run.

**Section toggles.** `RUN_SECTION_A` / `RUN_SECTION_B1` / `RUN_SECTION_B2`
/ `RUN_SECTION_B3` let you run just one piece at a time -- results from
sections you skip are preserved (merged, not overwritten) from any
existing results file, so you can build up the full results file
incrementally across several runs.

## Quick demo of the underlying idea

```bash
python run_calculation_base.py
```

Runs one Matrix-vs-Tensor eigenvalue comparison and one TT-rank
measurement at fixed parameters, prints the result, no sweep, no plot.
Useful to confirm the pipeline works before committing to the full run
above.
