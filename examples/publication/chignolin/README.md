# Chignolin

Three main figures + one appendix figure, all from chignolin folding
trajectory data.

## Files

```
chignolin/
├── README.md                                          (this file)
├── data/
│   └── README.md                                       precomputed data / Zenodo link
├── build_precomputed_data.py                            optional: regenerate the
│                                                          precomputed files from raw data
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

**Default: precomputed data (recommended).** Every calculation script here
loads two small files -- `cln_tica.npy` (TICA coordinates, fit once at
dimension 10) and `cln_diff.npy` (the diffusion tensor, projected onto all
10 TICA directions). Any script needing a smaller dimension just slices
these -- e.g. dim=3 uses `cln_tica.npy[:3, :]` -- which has been validated
to give bit-for-bit identical results to fitting/projecting separately at
that smaller dimension. With these two files, **no `jax`/`mdtraj`/`deeptime`
is needed at all** -- just `numpy`/`scipy`.

See `data/README.md` for the Zenodo download link. By default, every
script looks for `cln_tica.npy` and `cln_diff.npy` in `chignolin/data/`.
Override with the `CHIGNOLIN_DATA_DIR` environment variable if your data
lives elsewhere -- no need to edit any script:

```bash
CHIGNOLIN_DATA_DIR=/path/to/your/data python run_calculation_pcca_multi_dim.py
```

**If you have the raw data instead** (`cln_tica.pkl`, `cln_atomic.pkl` --
the large one -- and `chi_vac.pdb`) and want to regenerate the precomputed
files yourself (or verify them), run `build_precomputed_data.py` once:

```bash
CHIGNOLIN_DATA_DIR=/path/to/your/raw/data python build_precomputed_data.py
```

This does the real `jax` Jacobian computation and TICA fit (the expensive,
one-time step) and writes `cln_tica.npy`/`cln_diff.npy` -- after that,
every other script in this folder never touches the raw data again.

## Requirements

Just `tensor_gedmd` itself (`pip install -e .` from the repository root) --
no `deeptime`/`mdtraj`/`jax` needed for the calculation scripts, since TICA
and the diffusion tensor are already precomputed. Those three packages are
only needed if you're running `build_precomputed_data.py` to regenerate the
precomputed files from raw data:

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

`run_calculation_threshold_and_dim_sweep.py`,
`run_calculation_scratch_vs_incremental.py`, and
`chignolin_supplementary.py` each have a `QUICK_TEST` toggle near the top
for a fast smoke test with tiny parameters first.
`run_calculation_pcca_multi_dim.py` doesn't have one -- use
`run_calculation_pcca_base.py` (dim=3 only, no sweep) as its quick check
instead.