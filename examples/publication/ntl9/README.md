# NTL9

Two main figures + one appendix figure, from NTL9 folding trajectory data.
Data comes from precomputed `.npy` files (TICA coordinates + diffusion
tensor already computed) -- no `jax`/`mdtraj` needed here, unlike
chignolin.

## Files

```
ntl9/
├── README.md                                        (this file)
├── data/
│   └── README.md                                     raw data / Zenodo link
├── run_calculation_scratch_vs_incremental_base.py     idea: one subsample,
│                                                       scratch vs incremental
│                                                       across all dims (with
│                                                       bridging through the
│                                                       skipped dim 7)
├── run_calculation_scratch_vs_incremental.py           full: 10 independent
│                                                       subsamples
├── run_calculation_pcca_base.py                        idea: dim=3, both
│                                                       Matrix and Tensor
├── run_calculation_pcca_multi_dim.py                   full: dims 3, 6, 8
├── ntl9_supplementary_base.py                          idea: one (dim, sigma,
│                                                       tol) combination
├── ntl9_supplementary.py                               full: 4 dims x 5
│                                                       sigmas x 10 subsamples
│                                                       (appendix figure)
├── results/
└── plots/
    ├── ntl9_plot_figure_eigenvalues_scratch_vs_incremental.py
    ├── ntl9_plot_figure_pcca_matrix_vs_tensor_multi_dim.py
    └── plot_figure_ntl9_supplementary.py
```

Every `..._base.py` file is runnable standalone -- a quick, single-case
demo of the underlying idea, with no sweeping/repeating and no plot. See
`publication/README.md` for the general base/full pattern this follows.

## Data

See `data/README.md` for the Zenodo download link. By default, every
script here looks for `ntl9_tica.npy` and `ntl9_diff.npy` in
`ntl9/data/`. Override with the `NTL9_DATA_DIR` environment variable if
your data lives elsewhere -- no need to edit any script:

```bash
NTL9_DATA_DIR=/path/to/your/data python run_calculation_pcca_multi_dim.py
```

## Requirements

Just `tensor_gedmd` itself -- no `deeptime`/`mdtraj`/`jax` needed, since
NTL9's TICA and diffusion tensor are already precomputed in the data
files.

## Running it

Each pair follows the same pattern:

```bash
cd ntl9
python run_calculation_scratch_vs_incremental.py
python plots/ntl9_plot_figure_eigenvalues_scratch_vs_incremental.py

python run_calculation_pcca_multi_dim.py
python plots/ntl9_plot_figure_pcca_matrix_vs_tensor_multi_dim.py

python ntl9_supplementary.py
python plots/plot_figure_ntl9_supplementary.py
```

`run_calculation_scratch_vs_incremental.py` and `ntl9_supplementary.py`
each have a `QUICK_TEST` toggle near the top for a fast smoke test with
tiny parameters first. `run_calculation_pcca_multi_dim.py` doesn't have
one -- use `run_calculation_pcca_base.py` (dim=3 only, no sweep) as its
quick check instead.
