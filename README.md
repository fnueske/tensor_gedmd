# tensor_gedmd

Tensor-train (TT) methods for generator learning (gEDMD): estimating the
infinitesimal generator of a stochastic dynamical system, and its leading
eigenvalues/eigenfunctions, from simulation or trajectory data -- using a
tensor-train representation of the basis/data tensor to scale to high
dimensions without ever forming the full dense generator matrix.

## Installation

Requires Python >= 3.13.

```bash
git clone <REPO_URL>
cd tensor_gedmd
pip install -e .
```

This installs `tensor_gedmd` in editable mode, so changes to the source
take effect immediately without reinstalling.

## Package structure

```
src/tensor_gedmd/
├── algorithms/       core gEDMD pipeline
│   ├── gedmd.py             end-to-end pipeline: data -> eigenvalues
│   ├── global_svd.py        global SVD of the TT data tensor
│   ├── mat_vec_prod.py      TT matvec via block contraction (applies the
│   │                        TT-format stiffness operator explicitly)
│   ├── mat_vec_prod_direct.py   direct reduced-generator assembly (never
│   │                        materializes the TT operator)
│   └── util.py               whitening, eigenvalue filtering, rank
│                              truncation, dense reference method
├── basis_sets/        dictionary/basis function classes
│   ├── random_fourier_features.py
│   └── product_basis.py
├── operations/         generic TT array operations
├── reps/               core data structures
│   ├── tensor_train.py            TT class
│   ├── stiffness_tt.py            TT-format stiffness/generator operator
│   └── transformed_data_tensor.py  lazy TT data tensor from basis evaluations
└── systems/             SDE simulators for generating example data
    ├── ol_generic.py             generic overdamped Langevin base class
    └── decoupled_potential_3d.py  3D decoupled (Lemon-Slice) toy potential
```

## Quick start

```python
import numpy as np
from tensor_gedmd.algorithms.gedmd import run_gedmd_pipeline

# X: (p, m) array of m samples in p dimensions
X = np.random.randn(3, 500)

result = run_gedmd_pipeline(X, n_features=5, length_scale=1.0, nev=5)
print(result.eigenvalues)
```

(Purely random `X` here has no real dynamical structure, so this is just
confirming the pipeline runs -- for anything with genuine slow/metastable
structure, `n_features`/`m` can go much higher. If you do increase them,
also set `rmax` to cap the retained TT rank -- unstructured data has no
natural low-rank cutoff, so leaving `rmax=None` can get expensive fast.)

See `examples/` for runnable demos of individual pieces (`demo_global_svd.py`,
`demo_mat_vec_prod.py`, `demo_reps_tt.py`, `demo_transformed_data_tensor.py`).

## Reproducing the paper's figures

`examples/publication/` contains one folder per figure/case study
(`introduction/`, `lemon_slice_3d/`, `chignolin/`, `ntl9/`). Each
calculation script is slow (does the real computation, saves a small
results file); each plot script is fast (loads that file, draws the
figure). Several calculation scripts are further split into a `..._base.py`
file (the underlying idea, runnable standalone as a quick demo) and the
full script (imports from base, runs the complete sweep/repeat the actual
figure needs).

See `examples/publication/README.md` for the full pattern, and each case
study's own `README.md` for exact script names, requirements, and data
paths. Raw data and cached results are available on Zenodo.

## Testing

```bash
pip install pytest
pytest tests/
```
