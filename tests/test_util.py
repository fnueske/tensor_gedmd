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
