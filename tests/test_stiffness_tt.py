from __future__ import annotations

import numpy as np
import pytest

from tensor_gedmd.reps.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures
from tensor_gedmd.reps.tensor_train import TT
from tensor_gedmd.operations import tt_operator_to_dense as tt_to_dense


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
        vals = basis(xk.T)               # xk.T: (1, m) -> vals: (n, m)
        grads = basis._gradient(xk.T)    # (n, 1, m)

        psi.append(vals)                # (n, m)
        dpsi.append(grads[:, 0, :])     # (n, m)

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










class TestConstantSigmaNowGenuinelyUsed:
    """
    Regression coverage for a behavior change: TgStiffnessOperator used to
    silently ignore a real (non-identity) constant Sigma when building the
    TT operator. It now genuinely incorporates it via the same band-based
    construction used for samplewise Sigma.
    """

    def test_constant_nonidentity_sigma_changes_the_operator(self) -> None:
        rng = np.random.default_rng(11)
        p, m, dims = 2, 6, [3, 4]
        psi = [rng.normal(size=(n, m)) for n in dims]
        dpsi = [rng.normal(size=(n, m)) for n in dims]

        op_none = TgStiffnessOperator(psi=psi, dpsi=dpsi)
        Sigma_const = np.array([[2.0, 0.7], [0.7, 1.3]])
        op_const = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma_const)

        A_none = op_none.to_dense()
        A_const = op_const.to_dense()

        # A real, non-identity Sigma must actually change the operator.
        assert not np.allclose(A_none, A_const)

    def test_constant_sigma_matches_dense_direct_ground_truth(self) -> None:
        rng = np.random.default_rng(12)
        p, m, dims = 3, 5, [2, 3, 2]
        psi = [rng.normal(size=(n, m)) for n in dims]
        dpsi = [rng.normal(size=(n, m)) for n in dims]

        A = rng.normal(size=(p, p))
        Sigma = A @ A.T + np.eye(p)  # SPD, non-identity

        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
        A_tt = op.to_dense()
        A_direct = op.build_dense_direct()

        assert np.allclose(A_tt, A_direct, atol=1e-10)
        assert np.allclose(A_tt, A_tt.T)

    def test_no_warning_for_constant_sigma(self, recwarn) -> None:
        rng = np.random.default_rng(13)
        psi = [rng.normal(size=(3, 5)), rng.normal(size=(4, 5))]
        dpsi = [rng.normal(size=(3, 5)), rng.normal(size=(4, 5))]
        Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])

        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
        op.to_dense()

        assert len(recwarn) == 0


class TestBandBasedGroundTruth:
    """build_dense_direct() as an independent ground truth for the TT construction."""

    @pytest.mark.parametrize("p,dims", [(2, [3, 4]), (3, [2, 3, 2]), (4, [2, 3, 4, 2])])
    @pytest.mark.parametrize("sigma_case", ["none", "constant", "variable"])
    def test_to_dense_matches_build_dense_direct(self, p, dims, sigma_case) -> None:
        rng = np.random.default_rng(hash((p, sigma_case)) % (2**31))
        m = 6
        psi = [rng.normal(size=(n, m)) for n in dims]
        dpsi = [rng.normal(size=(n, m)) for n in dims]

        if sigma_case == "none":
            Sigma = None
        elif sigma_case == "constant":
            A = rng.normal(size=(p, p))
            Sigma = A @ A.T + np.eye(p)
        else:
            Sigma = np.zeros((p, p, m))
            for l in range(m):
                A = rng.normal(size=(p, p))
                Sigma[:, :, l] = A @ A.T + np.eye(p)

        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
        A_tt = op.to_dense()
        A_direct = op.build_dense_direct()

        assert np.allclose(A_tt, A_direct, atol=1e-9)
        assert np.allclose(A_tt, A_tt.T, atol=1e-12)
