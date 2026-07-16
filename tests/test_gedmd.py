from __future__ import annotations

import numpy as np
import pytest

from tensor_gedmd.algorithms.gedmd import (
    GedmdResult,
    build_deterministic_rff_basis,
    deterministic_rff_frequencies,
    evaluate_basis,
    mean_free_whitening,
    run_gedmd_pipeline,
)
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures


def _make_X(p=2, m=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(p, m))


class TestBasisHelpers:
    def test_deterministic_rff_frequencies_shape_and_values(self) -> None:
        omega = deterministic_rff_frequencies(5, length_scale=2.0)
        assert omega.shape == (5, 1)
        assert np.allclose(omega[:, 0], np.arange(1, 6) / 2.0)

    def test_deterministic_rff_frequencies_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError):
            deterministic_rff_frequencies(0, 1.0)
        with pytest.raises(ValueError):
            deterministic_rff_frequencies(5, 0.0)

    def test_build_deterministic_rff_basis_scalar_length_scale(self) -> None:
        basis_list = build_deterministic_rff_basis(3, n_features=4, length_scale=1.5)
        assert len(basis_list) == 3
        assert all(isinstance(b, RandomFourierFeatures) for b in basis_list)
        assert all(b.n == 4 for b in basis_list)

    def test_build_deterministic_rff_basis_per_dim_length_scale(self) -> None:
        basis_list = build_deterministic_rff_basis(2, n_features=3, length_scale=[1.0, 2.0])
        assert len(basis_list) == 2

    def test_build_deterministic_rff_basis_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError):
            build_deterministic_rff_basis(3, n_features=4, length_scale=[1.0, 2.0])

    def test_evaluate_basis_shapes(self) -> None:
        X = _make_X(p=2, m=10)
        basis_list = build_deterministic_rff_basis(2, n_features=4, length_scale=1.0)
        psi, dpsi = evaluate_basis(X, basis_list)
        assert len(psi) == len(dpsi) == 2
        assert all(a.shape == (4, 10) for a in psi)
        assert all(a.shape == (4, 10) for a in dpsi)

    def test_evaluate_basis_rejects_mismatched_basis_count(self) -> None:
        X = _make_X(p=2, m=10)
        basis_list = build_deterministic_rff_basis(3, n_features=4, length_scale=1.0)
        with pytest.raises(ValueError, match="expected p=2"):
            evaluate_basis(X, basis_list)


class TestMeanFreeWhitening:
    def test_shapes_and_drops_one_eigenvalue(self) -> None:
        rng = np.random.default_rng(2)
        r, m = 6, 50
        Z = rng.normal(size=(r, m))
        d_G, W_G = mean_free_whitening(Z)
        assert d_G.shape == (r - 1,)
        assert W_G.shape == (r, r - 1)

    def test_eigenvalues_are_positive_after_dropping_zero(self) -> None:
        # I - outer(mean, mean) is positive semi-definite with exactly one
        # near-zero eigenvalue when ||mean|| ~ 1 (the direction along mean);
        # for generic Z the rest should be strictly positive.
        rng = np.random.default_rng(3)
        Z = rng.normal(size=(5, 40)) * 0.1  # small mean -> well-conditioned
        d_G, _ = mean_free_whitening(Z)
        assert np.all(d_G > 0)


class TestRunGedmdPipeline:
    @pytest.mark.parametrize("Sigma_case", ["none", "constant_identity", "constant_general"])
    def test_constant_sigma_modes_run_and_give_real_eigs(self, Sigma_case) -> None:
        X = _make_X(p=2, m=150, seed=1)

        if Sigma_case == "none":
            Sigma = None
        elif Sigma_case == "constant_identity":
            Sigma = np.eye(2)
        else:
            Sigma = np.array([[1.5, 0.2], [0.2, 0.8]])

        result = run_gedmd_pipeline(
            X, n_features=5, length_scale=1.0, Sigma=Sigma, nev=4, tol=1e-6
        )
        assert isinstance(result, GedmdResult)
        assert result.eigenvalues.shape == (4,)
        assert np.all(np.isreal(result.eigenvalues))
        # eigenvalues should be returned in descending order
        assert np.all(np.diff(result.eigenvalues) <= 1e-10)

    def test_samplewise_sigma_runs_and_gives_real_eigs(self) -> None:
        rng = np.random.default_rng(4)
        p, m = 3, 120
        X = _make_X(p=p, m=m, seed=4)

        Sigma = np.zeros((p, p, m))
        for l in range(m):
            A = rng.normal(size=(p, p)) * 0.3
            Sigma[:, :, l] = np.eye(p) + A @ A.T

        result = run_gedmd_pipeline(
            X, n_features=4, length_scale=1.5, Sigma=Sigma, nev=3,
            tol=1e-6, chunk_size=40,
        )
        assert result.eigenvalues.shape == (3,)
        assert np.all(np.isreal(result.eigenvalues))

    def test_eigenfunctions_shape(self) -> None:
        X = _make_X(p=2, m=80, seed=5)
        result = run_gedmd_pipeline(X, n_features=4, nev=3, tol=1e-6)
        ef = result.eigenfunctions()
        assert ef.shape == (3, 80)

    def test_custom_basis_list_is_used(self) -> None:
        rng = np.random.default_rng(6)
        X = _make_X(p=2, m=60, seed=6)
        basis_list = [
            RandomFourierFeatures(omega=rng.normal(size=(4, 1))) for _ in range(2)
        ]
        result = run_gedmd_pipeline(X, basis_list=basis_list, nev=2, tol=1e-6)
        assert result.eigenvalues.shape == (2,)

    def test_nev_is_capped_without_error(self) -> None:
        X = _make_X(p=2, m=60, seed=7)
        result = run_gedmd_pipeline(X, n_features=3, nev=10_000, tol=1e-6)
        assert result.eigenvalues.shape[0] < 10_000
        assert result.eigenvalues.shape[0] > 0

    def test_rejects_wrong_x_ndim(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            run_gedmd_pipeline(np.zeros(10))

    def test_rejects_mismatched_basis_list_length(self) -> None:
        X = _make_X(p=2, m=20)
        basis_list = build_deterministic_rff_basis(3, n_features=4, length_scale=1.0)
        with pytest.raises(ValueError, match="expected p=2"):
            run_gedmd_pipeline(X, basis_list=basis_list)

    def test_result_pieces_have_consistent_shapes(self) -> None:
        X = _make_X(p=2, m=100, seed=8)
        result = run_gedmd_pipeline(X, n_features=5, nev=3, tol=1e-6)

        r = result.U_cores[-1].shape[2]
        assert result.Sigma_svd.shape == (r, r)
        assert result.Z.shape[0] == r
        assert result.Z.shape[1] == 100
        assert result.Z_white.shape == (r - 1, 100)
        assert result.d_G.shape == (r - 1,)
        assert result.W_G.shape == (r, r - 1)
        assert result.A_r.shape == (r, r)
        assert result.reduced_matrix.shape == (r, r)
