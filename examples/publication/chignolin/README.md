# Chignolin

Three main figures + one appendix figure, all from chignolin folding
trajectory data.

## Files

```
chignolin/
├── README.md                                          (this file)
├── data/
│   └── README.md                                       raw data / Zenodo link
├── run_calculation_threshold_and_dim_sweep_base.py      idea: one subsample,
│                                                          one dim, swept over
│                                                          tolerances
├── run_calculation_threshold_and_dim_sweep.py            full: 3 dims x 10 runs
│                                                          -> eigenvalues vs SVD
│                                                          threshold and TICA dim
├── run_calculation_scratch_vs_incremental_base.py        idea: one subsample,
│                                                          scratch vs incremental
│                                                          across all dims
├── run_calculation_scratch_vs_incremental.py              full: 10 independent
│                                                          subsamples
├── run_calculation_pcca_base.py                           idea: dim=3, both
│                                                          Matrix and Tensor
├── run_calculation_pcca_multi_dim.py                      full: dims 3, 6, 10
├── chignolin_supplementary_base.py                        idea: one (dim, sigma,
│                                                          tol) combination
├── chignolin_supplementary.py                             full: 5 dims x 5
│                                                          sigmas x 10 repeats
│                                                          (appendix figure)
├── results/
└── plots/
    ├── chignolin_plot_figure_eigenvalues_vs_svd_threshold_and_tica_dimension.py
    ├── chignolin_plot_figure_eigenvalues_and_runtime_scratch_vs_incremental.py
    ├── chignolin_plot_figure_pcca_matrix_vs_tensor_multi_dim.py
    └── plot_figure_chignolin_supplementary.py
```

Every `..._base.py` file is runnable standalone -- a quick, single-case
demo of the underlying idea, with no sweeping/repeating and no plot. See
`publication/README.md` for the general base/full pattern this follows.

## Data

See `data/README.md` for the Zenodo download link. By default, every
script here looks for `cln_tica.pkl`, `cln_atomic.pkl`, and `chi_vac.pdb`
in `chignolin/data/`. Override with the `CHIGNOLIN_DATA_DIR` environment
variable if your data lives elsewhere -- no need to edit any script:

```bash
CHIGNOLIN_DATA_DIR=/path/to/your/data python run_calculation_pcca_multi_dim.py
```

## Requirements

```bash
pip install deeptime mdtraj jax jaxlib
```

## Running it

Each pair follows the same pattern:

```bash
cd chignolin
python run_calculation_threshold_and_dim_sweep.py
python plots/chignolin_plot_figure_eigenvalues_vs_svd_threshold_and_tica_dimension.py

python run_calculation_scratch_vs_incremental.py
python plots/chignolin_plot_figure_eigenvalues_and_runtime_scratch_vs_incremental.py

python run_calculation_pcca_multi_dim.py
python plots/chignolin_plot_figure_pcca_matrix_vs_tensor_multi_dim.py

python chignolin_supplementary.py
python plots/plot_figure_chignolin_supplementary.py
```

All four calculation scripts do a real `jax` Jacobian computation over the
trajectory data -- expect each to take a while, not something to run
casually multiple times. `run_calculation_threshold_and_dim_sweep.py`,
`run_calculation_scratch_vs_incremental.py`, and
`chignolin_supplementary.py` each have a `QUICK_TEST` toggle near the top
for a fast smoke test with tiny parameters first.
`run_calculation_pcca_multi_dim.py` doesn't have one -- use
`run_calculation_pcca_base.py` (dim=3 only, no sweep) as its quick check
instead.
