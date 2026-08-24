from __future__ import annotations

import numpy as np
import pytest

from tensor_gedmd.algorithms.util import _truncate_rank, filter_ev


class TestTruncateRank:
    def test_no_truncation_by_default(self) -> None:
        s = np.array([5.0, 3.0, 1.0, 0.1])
        assert _truncate_rank(s) == 4

    def test_rmax_caps_rank(self) -> None:
        s = np.array([5.0, 3.0, 1.0, 0.1])
        assert _truncate_rank(s, rmax=2) == 2

    def test_tol_truncates_small_tail(self) -> None:
        s = np.array([1.0, 1e-10])
        # tail energy of just the last singular value is negligible
        assert _truncate_rank(s, tol=1e-6) == 1

    def test_all_zero_singular_values_returns_one_not_nan(self) -> None:
        # Regression test: previously divided by tail_energy[0] == 0 -> NaN.
        s = np.zeros(4)
        assert _truncate_rank(s, tol=0.5) == 1

    def test_empty_singular_values_returns_one(self) -> None:
        assert _truncate_rank(np.array([])) == 1

    def test_rank_never_below_one(self) -> None:
        s = np.array([1.0, 1e-15, 1e-16])
        assert _truncate_rank(s, rmax=1, tol=0.999999) == 1


class TestFilterEv:
    def test_sorts_ascending_by_real_part(self) -> None:
        d = np.array([3.0, -1.0, 2.0])
        W = np.eye(3)
        d_out, W_out = filter_ev(d, W)
        assert np.allclose(d_out, [-1.0, 2.0, 3.0])
        # columns of W follow the same permutation as d
        assert np.allclose(W_out[:, 0], W[:, 1])
        assert np.allclose(W_out[:, 1], W[:, 2])
        assert np.allclose(W_out[:, 2], W[:, 0])

    def test_filters_by_real_part_bounds(self) -> None:
        d = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        W = np.eye(5)
        d_out, W_out = filter_ev(d, W, eps1=-2.0, eps2=2.0)
        assert np.allclose(d_out, [-1.0, 0.0, 1.0])
        assert W_out.shape == (5, 3)

    def test_bounds_are_exclusive(self) -> None:
        d = np.array([-1.0, 0.0, 1.0])
        W = np.eye(3)
        d_out, _ = filter_ev(d, W, eps1=-1.0, eps2=1.0)
        assert np.allclose(d_out, [0.0])

    def test_no_bounds_keeps_everything_sorted(self) -> None:
        d = np.array([2.0, -3.0, 0.5])
        W = np.eye(3)
        d_out, W_out = filter_ev(d, W)
        assert np.allclose(d_out, [-3.0, 0.5, 2.0])
        assert W_out.shape == (3, 3)

    def test_complex_eigenvalues_sorted_by_real_part(self) -> None:
        d = np.array([1.0 + 2.0j, -1.0 + 5.0j, 0.0 - 1.0j])
        W = np.eye(3, dtype=complex)
        d_out, _ = filter_ev(d, W)
        assert np.allclose(np.real(d_out), [-1.0, 0.0, 1.0])

    def test_shape_mismatch_raises(self) -> None:
        d = np.array([1.0, 2.0])
        W = np.eye(3)
        with pytest.raises(ValueError, match="shape"):
            filter_ev(d, W)


class TestWhiteningTransformFaithfulPort:
    """
    whitening_transform, evaluate_generator_rev, and spectral_analysis_gedmd_rev
    are faithful ports of the actual dmp_methods.Util.util / dmp_methods.gEDMD.gEDMD
    reference implementation (source provided directly). These check the
    core correctness properties that implementation must satisfy, plus
    self-consistency between its scalar-diffusion and samplewise-diffusion
    code paths.
    """

    def test_whitening_transform_orthonormalizes_gram(self) -> None:
        from tensor_gedmd.algorithms.util import whitening_transform

        rng = np.random.default_rng(0)
        n, m = 15, 200
        PhiX = rng.normal(size=(n, m))

        L = whitening_transform(PhiX, tol=1e-10)
        Gram = PhiX @ PhiX.T
        check = L.T @ Gram @ L
        assert np.allclose(check, np.eye(check.shape[0]), atol=1e-8)

    def test_whitening_transform_respects_rmin(self) -> None:
        from tensor_gedmd.algorithms.util import whitening_transform

        rng = np.random.default_rng(1)
        # Nearly rank-1 PhiX: only one singular value should survive a
        # strict tol, but rmin should force at least 3 columns.
        u = rng.normal(size=(10, 1))
        v = rng.normal(size=(1, 50))
        PhiX = u @ v + 1e-12 * rng.normal(size=(10, 50))

        L = whitening_transform(PhiX, tol=0.5, rmin=3)
        assert L.shape[1] >= 3

    def test_evaluate_generator_rev_symmetric(self) -> None:
        from tensor_gedmd.algorithms.util import evaluate_generator_rev
        from tensor_gedmd.basis_sets.product_basis import ProductBasis
        from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures

        rng = np.random.default_rng(2)
        p, m, dims = 3, 100, [4, 4, 4]
        X = rng.normal(size=(p, m))
        basis_list = [RandomFourierFeatures(omega=rng.normal(size=(n, 1))) for n in dims]
        phi = ProductBasis(basis_list)

        A_L = evaluate_generator_rev(X, phi, a=1.5)
        assert np.allclose(A_L, A_L.T)

    def test_evaluate_generator_rev_scalar_matches_equivalent_tensor(self) -> None:
        from tensor_gedmd.algorithms.util import evaluate_generator_rev
        from tensor_gedmd.basis_sets.product_basis import ProductBasis
        from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures

        rng = np.random.default_rng(3)
        p, m, dims = 3, 80, [3, 3, 3]
        X = rng.normal(size=(p, m))
        basis_list = [RandomFourierFeatures(omega=rng.normal(size=(n, 1))) for n in dims]
        phi = ProductBasis(basis_list)

        a_scalar = 2.0
        A_L_scalar = evaluate_generator_rev(X, phi, a=a_scalar)

        a_tensor = np.stack([a_scalar * np.eye(p) for _ in range(m)], axis=-1)
        A_L_tensor = evaluate_generator_rev(X, phi, a=a_tensor)

        assert np.allclose(A_L_scalar, A_L_tensor, atol=1e-10)

    def test_spectral_analysis_gedmd_rev_runs_and_gives_real_sorted_eigs(self) -> None:
        from tensor_gedmd.algorithms.util import spectral_analysis_gedmd_rev
        from tensor_gedmd.basis_sets.product_basis import ProductBasis
        from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures

        rng = np.random.default_rng(4)
        p, m, dims = 3, 200, [4, 4, 4]
        X = rng.normal(size=(p, m))
        basis_list = [RandomFourierFeatures(omega=rng.normal(size=(n, 1))) for n in dims]
        phi = ProductBasis(basis_list)

        d, W, Wdata = spectral_analysis_gedmd_rev(X, phi, nev=10, a=2.0, tol=1e-6, eps_ev=0.0)

        assert d.shape == (10,)
        assert np.all(np.isreal(d))
        assert np.all(np.diff(d) >= -1e-10)  # ascending
        assert Wdata.shape == (10, m)

    def test_spectral_analysis_gedmd_rev_is_mean_free(self) -> None:
        """
        spectral_analysis_gedmd_rev is this project's deliberate mean-free
        variant (see its docstring): PhiX is mean-centered before
        whitening, unlike the plain dmp_methods.gEDMD.gEDMD library. This
        confirms that behavior directly: shifting the basis by a constant
        offset should be absorbed by the mean-centering and leave the
        result unchanged.
        """
        from tensor_gedmd.algorithms.util import spectral_analysis_gedmd_rev
        from tensor_gedmd.basis_sets.product_basis import ProductBasis
        from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures

        rng = np.random.default_rng(5)
        p, m, dims = 2, 150, [4, 4]
        X = rng.normal(size=(p, m))
        basis_list = [RandomFourierFeatures(omega=rng.normal(size=(n, 1))) for n in dims]
        phi = ProductBasis(basis_list)

        d1, _, _ = spectral_analysis_gedmd_rev(X, phi, nev=5, a=1.0, tol=1e-6)

        # A basis shift large enough to move PhiX's row-means substantially.
        class ShiftedBasis:
            def __init__(self, inner, shift):
                self.inner = inner
                self.shift = shift

            def __call__(self, Xin):
                return self.inner(Xin) + self.shift

            def gradient(self, Xin):
                return self.inner.gradient(Xin)

        shift = rng.normal(size=(phi.n, 1)) * 5.0
        phi_shifted = ShiftedBasis(phi, shift)
        d2, _, _ = spectral_analysis_gedmd_rev(X, phi_shifted, nev=5, a=1.0, tol=1e-6)

        # Mean-centering should absorb the constant shift entirely.
        assert np.allclose(d1, d2, atol=1e-6)
