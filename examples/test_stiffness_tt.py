from __future__ import annotations

import numpy as np
import pytest

from tensor_gedmd.algorithms.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures
from tensor_gedmd.reps.tensor_train import TT


def make_local_rff_data():
    """
    Build small deterministic 1D local basis evaluations for a 4D product space.

    Returns
    -------
    psi : list[np.ndarray]
        Each entry has shape (n, m).
    dpsi : list[np.ndarray]
        Each entry has shape (n, m).
    x_by_dim : list[np.ndarray]
        Each entry has shape (m, 1), the input samples used for each local basis.
    bases : list[RandomFourierFeatures]
        The local basis objects.
    """
    m = 4   # number of samples
    p = 4   # number of dimensions / TT cores
    n = 2   # number of local features

    x_by_dim = [
        np.array([[0.0], [0.2], [0.4], [0.6]], dtype=float),
        np.array([[0.1], [0.3], [0.5], [0.7]], dtype=float),
        np.array([[0.2], [0.4], [0.6], [0.8]], dtype=float),
        np.array([[0.15], [0.35], [0.55], [0.75]], dtype=float),
    ]

    omegas = [
        np.array([[1.0], [2.0]], dtype=float),
        np.array([[0.5], [1.5]], dtype=float),
        np.array([[1.2], [2.2]], dtype=float),
        np.array([[0.8], [1.8]], dtype=float),
    ]

    offsets = [
        np.array([0.0, 0.3], dtype=float),
        np.array([0.1, 0.4], dtype=float),
        np.array([0.2, 0.5], dtype=float),
        np.array([0.15, 0.45], dtype=float),
    ]

    bases = [
        RandomFourierFeatures(omega=omega, b=b)
        for omega, b in zip(omegas, offsets)
    ]

    psi = []
    dpsi = []

    for basis, xk in zip(bases, x_by_dim):
        vals = basis(xk)               # (m, n)
        grads = basis._gradient(xk)    # (m, n, 1)

        psi.append(vals.T)             # -> (n, m)
        dpsi.append(grads[:, :, 0].T)  # -> (n, m)

    assert len(psi) == p
    assert all(a.shape == (n, m) for a in psi)
    assert all(a.shape == (n, m) for a in dpsi)

    return psi, dpsi, x_by_dim, bases


def kron_all(mats):
    out = mats[0]
    for a in mats[1:]:
        out = np.kron(out, a)
    return out


def build_dense_reference(psi, dpsi):
    """
    Dense reference for the TT stiffness operator represented by the cores.

    For each sample s:
      G_k^(s) = psi_k(:, s) psi_k(:, s)^T
      S_k^(s) = dpsi_k(:, s) dpsi_k(:, s)^T

    The operator is
      A = -(1/m) * sum_s sum_ell kron(F_0, ..., F_{p-1})
    where
      F_k = G_k^(s) for k != ell
      F_ell = S_ell^(s)
    """
    p = len(psi)
    m = psi[0].shape[1]
    n = psi[0].shape[0]

    A = np.zeros((n**p, n**p), dtype=float)

    for s in range(m):
        gram_blocks = [np.outer(psi[k][:, s], psi[k][:, s]) for k in range(p)]
        stiff_blocks = [np.outer(dpsi[k][:, s], dpsi[k][:, s]) for k in range(p)]

        for ell in range(p):
            factors = list(gram_blocks)
            factors[ell] = stiff_blocks[ell]
            A += kron_all(factors)

    A *= -1.0 / m
    return A


def tt_to_dense(tt):
    """
    Convert a TT operator with cores of shape
        (r_{k-1}, n_k, n_k, r_k)
    into a dense matrix of shape
        (prod_k n_k, prod_k n_k).

    Assumes operator-form TT cores, consistent with your printed shapes.
    """
    cores = tt.cores
    if len(cores) == 0:
        raise ValueError("TT has no cores.")

    # Start from first core: (1, n, n, r1) -> (n, n, r1)
    first = cores[0]
    if first.ndim != 4:
        raise ValueError("Expected TT operator cores with 4 dimensions.")

    if first.shape[0] != 1:
        raise ValueError("First TT rank must be 1.")

    tensor = first[0]  # shape: (n1, n1, r1)

    # Contract successive cores along TT-rank indices
    for k in range(1, len(cores)):
        core = cores[k]  # shape: (r_{k-1}, nk, nk, r_k)

        if core.ndim != 4:
            raise ValueError(f"Core {k} is not a 4D TT operator core.")

        # tensor: (...row_dims..., ...col_dims..., r_prev)
        # core  : (r_prev, nk, nk, r_next)
        #
        # Contract over r_prev
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
        # Result shape before reorder:
        # (...rows..., ...cols..., nk, nk, r_next)

    # Last rank must be 1
    if tensor.shape[-1] != 1:
        raise ValueError("Last TT rank must be 1.")

    tensor = tensor[..., 0]

    # At this point tensor shape is:
    # (n1, n1, n2, n2, ..., np, np)
    # We need to permute it to:
    # (n1, n2, ..., np, n1, n2, ..., np)
    p = len(cores)
    perm = list(range(0, 2 * p, 2)) + list(range(1, 2 * p, 2))
    tensor = np.transpose(tensor, axes=perm)

    row_dims = [core.shape[1] for core in cores]
    col_dims = [core.shape[2] for core in cores]

    return tensor.reshape(int(np.prod(row_dims)), int(np.prod(col_dims)))


def test_build_returns_tt_operator():
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    tt = op.build()

    assert isinstance(tt, TT)
    assert tt.is_operator is True
    assert len(tt.cores) == 4


def test_core_shapes_from_rff():
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    cores = op.build_cores()

    m = psi[0].shape[1]
    n = psi[0].shape[0]

    assert len(cores) == 4
    assert cores[0].shape == (1, n, n, 2 * m)
    assert cores[1].shape == (2 * m, n, n, 2 * m)
    assert cores[2].shape == (2 * m, n, n, 2 * m)
    assert cores[3].shape == (2 * m, n, n, 1)


def test_first_core_expected_blocks():
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    cores = op.build_cores()

    core0 = cores[0]
    m = psi[0].shape[1]

    for s in [0, 1]:
        v = psi[0][:, s]
        dv = dpsi[0][:, s]

        expected_gram = -np.outer(v, v) / m
        expected_stiff = -np.outer(dv, dv) / m

        np.testing.assert_allclose(core0[0, :, :, 2 * s], expected_gram)
        np.testing.assert_allclose(core0[0, :, :, 2 * s + 1], expected_stiff)


@pytest.mark.parametrize("core_idx", [1, 2])
def test_middle_core_expected_local_2x2_pattern(core_idx):
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    cores = op.build_cores()

    core = cores[core_idx]
    s = 0

    v = psi[core_idx][:, s]
    dv = dpsi[core_idx][:, s]

    ek = np.outer(v, v)
    fk = np.outer(dv, dv)

    np.testing.assert_allclose(core[0, :, :, 0], ek)
    np.testing.assert_allclose(core[1, :, :, 1], ek)
    np.testing.assert_allclose(core[0, :, :, 1], fk)
    np.testing.assert_allclose(core[1, :, :, 0], 0.0)


def test_last_core_expected_blocks():
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    cores = op.build_cores()

    core_last = cores[3]
    s = 0

    v = psi[3][:, s]
    dv = dpsi[3][:, s]

    expected_stiff = np.outer(dv, dv)
    expected_gram = np.outer(v, v)

    np.testing.assert_allclose(core_last[2 * s, :, :, 0], expected_stiff)
    np.testing.assert_allclose(core_last[2 * s + 1, :, :, 0], expected_gram)


def test_tt_ranks_are_consistent():
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    tt = op.build()

    m = psi[0].shape[1]
    assert tt.tt_ranks() == [1, 2 * m, 2 * m, 2 * m, 1]


def test_tt_matches_dense_reference():
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    tt = op.build()

    A_ref = build_dense_reference(psi, dpsi)
    A_tt = tt_to_dense(tt)

    assert A_tt.shape == A_ref.shape

    abs_err = np.linalg.norm(A_tt - A_ref, ord="fro")
    rel_err = abs_err / np.linalg.norm(A_ref, ord="fro")

    print("\nDense reference shape:", A_ref.shape)
    print("TT dense shape       :", A_tt.shape)
    print("Absolute Frobenius error:", abs_err)
    print("Relative Frobenius error:", rel_err)

    np.testing.assert_allclose(rel_err, 0.0, atol=1e-12)


def test_print_demo(capsys):
    """
    Optional test that prints a readable summary.
    Run with: pytest -s tests/test_stiffness_tt.py -q
    """
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    tt = op.build()

    print("\nTT object:")
    print(tt)
    print("\nCore shapes:")
    for i, core in enumerate(tt.cores):
        print(f"core[{i}].shape = {core.shape}")

    print("\nFirst core, sample-0 gram block:")
    print(tt.cores[0][0, :, :, 0])

    print("\nFirst core, sample-0 stiffness block:")
    print(tt.cores[0][0, :, :, 1])

    captured = capsys.readouterr()
    assert "core[0].shape" in captured.out
    assert "core[3].shape" in captured.out


if __name__ == "__main__":
    psi, dpsi, _, _ = make_local_rff_data()

    op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
    tt = op.build()

    A_ref = build_dense_reference(psi, dpsi)
    A_tt = tt_to_dense(tt)

    abs_err = np.linalg.norm(A_tt - A_ref, ord="fro")
    rel_err = abs_err / np.linalg.norm(A_ref, ord="fro")

    print("=" * 60)
    print("TT STIFFNESS OPERATOR DEMO")
    print("=" * 60)
    print("\nTT object:")
    print(tt)

    print("\nTT ranks:")
    print(tt.tt_ranks())

    print("\nCore shapes:")
    for i, core in enumerate(tt.cores):
        print(f"core[{i}].shape = {core.shape}")

    print("\nDense reference shape:", A_ref.shape)
    print("TT dense shape       :", A_tt.shape)
    print("Absolute Frobenius error:", abs_err)
    print("Relative Frobenius error:", rel_err)








