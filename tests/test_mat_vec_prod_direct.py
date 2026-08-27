from __future__ import annotations

import numpy as np
import pytest

from tensor_gedmd.algorithms.mat_vec_prod_direct import classify_sigma
from tensor_gedmd.algorithms.mat_vec_prod_direct import compute_A_r as compute_A_r_fast
from tensor_gedmd.algorithms.global_svd import global_svd_tt
from tensor_gedmd.algorithms.mat_vec_prod import compute_A_r as compute_A_r_orig
from tensor_gedmd.reps.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.reps.transformed_data_tensor import Transformed_Data_Tensor_TT


def _make_reduced_basis(psi, tol=1e-8):
    """Build U_cores the same way the real pipeline does (gedmd.py)."""
    builder = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)
    U_tt, Sigma_svd, V_core = global_svd_tt(builder.to_tt(), rmax=None, tol=tol)
    return U_tt.cores


def _make_problem(p=3, m=30, dims=(3, 4, 3), seed=0):
    rng = np.random.default_rng(seed)
    dims = list(dims)
    psi = [rng.normal(size=(n, m)) for n in dims]
    dpsi = [rng.normal(size=(n, m)) for n in dims]
    U_cores = _make_reduced_basis(psi)
    return rng, psi, dpsi, U_cores


class TestClassifySigma:
    def test_none_is_identity(self) -> None:
        _, psi, dpsi, _ = _make_problem()
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
        kind, payload = classify_sigma(op)
        assert kind == "identity"
        assert payload is None

    def test_scalar_times_identity(self) -> None:
        rng, psi, dpsi, _ = _make_problem()
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=2.5 * np.eye(3))
        kind, payload = classify_sigma(op)
        assert kind == "scalar"
        assert payload == pytest.approx(2.5)

    def test_diagonal_constant(self) -> None:
        rng, psi, dpsi, _ = _make_problem()
        diag_vals = np.array([1.0, 2.0, 3.0])
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=np.diag(diag_vals))
        kind, payload = classify_sigma(op)
        assert kind == "diagonal"
        assert np.allclose(payload, diag_vals)

    def test_dense_constant(self) -> None:
        rng, psi, dpsi, _ = _make_problem()
        A = rng.normal(size=(3, 3))
        Sigma = A @ A.T + np.eye(3)
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
        kind, payload = classify_sigma(op)
        assert kind == "dense_constant"
        assert np.allclose(payload, Sigma)

    def test_diagonal_samplewise(self) -> None:
        rng, psi, dpsi, _ = _make_problem()
        m = psi[0].shape[1]
        Sigma = np.zeros((3, 3, m))
        for l in range(m):
            Sigma[:, :, l] = np.diag(rng.uniform(0.5, 3.0, size=3))
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
        kind, payload = classify_sigma(op)
        assert kind == "diag_samplewise"
        assert payload.shape == (3, m)

    def test_dense_samplewise(self) -> None:
        rng, psi, dpsi, _ = _make_problem()
        m = psi[0].shape[1]
        Sigma = np.zeros((3, 3, m))
        for l in range(m):
            A = rng.normal(size=(3, 3))
            Sigma[:, :, l] = A @ A.T + np.eye(3)
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
        kind, payload = classify_sigma(op)
        assert kind == "dense_samplewise"
        assert payload.shape == (3, 3, m)


class TestComputeARFastMatchesOriginal:
    """
    compute_A_r_fast must be numerically identical to mat_vec_prod.compute_A_r
    for every Sigma structure -- it's a performance optimization on the same
    math, not a different algorithm.
    """

    @pytest.mark.parametrize(
        "sigma_case",
        ["none", "scalar", "diagonal_constant", "dense_constant",
         "diag_samplewise", "dense_samplewise"],
    )
    def test_matches_original_exactly(self, sigma_case) -> None:
        rng, psi, dpsi, U_cores = _make_problem(p=3, m=25, dims=(3, 4, 3), seed=1)
        p, m = 3, 25

        if sigma_case == "none":
            Sigma = None
        elif sigma_case == "scalar":
            Sigma = 1.7 * np.eye(p)
        elif sigma_case == "diagonal_constant":
            Sigma = np.diag(np.array([0.5, 2.0, 1.2]))
        elif sigma_case == "dense_constant":
            A = rng.normal(size=(p, p))
            Sigma = A @ A.T + np.eye(p)
        elif sigma_case == "diag_samplewise":
            Sigma = np.zeros((p, p, m))
            for l in range(m):
                Sigma[:, :, l] = np.diag(rng.uniform(0.5, 3.0, size=p))
        else:  # dense_samplewise
            Sigma = np.zeros((p, p, m))
            for l in range(m):
                A = rng.normal(size=(p, p))
                Sigma[:, :, l] = A @ A.T + np.eye(p)

        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
        r = U_cores[-1].shape[2]

        A_r_orig = compute_A_r_orig(op, U_cores, r, chunk_size=10, n_workers=2)
        A_r_fast = compute_A_r_fast(op, U_cores, r, chunk_size=10, n_workers=2)

        assert np.allclose(A_r_orig, A_r_fast, atol=1e-10)

    def test_exact_call_signature_from_user(self) -> None:
        # Mirrors the exact calling convention requested:
        # A_r = compute_A_r(op=op, U_cores=U_cores, r=U_cores[-1].shape[2],
        #                    chunk_size=100, n_workers=None, r_cap=U_cores[-1].shape[2])
        rng, psi, dpsi, U_cores = _make_problem(p=3, m=20, dims=(3, 3, 3), seed=2)
        p, m = 3, 20
        Sigma = np.zeros((p, p, m))
        for l in range(m):
            A = rng.normal(size=(p, p))
            Sigma[:, :, l] = A @ A.T + np.eye(p)
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)

        A_r = compute_A_r_fast(
            op=op, U_cores=U_cores,
            r=U_cores[-1].shape[2],
            chunk_size=100, n_workers=None,
            r_cap=U_cores[-1].shape[2],
        )

        r = U_cores[-1].shape[2]
        assert A_r.shape == (r, r)
        assert np.allclose(A_r, A_r.T)

    def test_r_cap_truncates_consistently_with_original(self) -> None:
        rng, psi, dpsi, U_cores = _make_problem(p=3, m=20, dims=(3, 3, 3), seed=3)
        p, m = 3, 20
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=1.5 * np.eye(p))
        r_full = U_cores[-1].shape[2]
        r_cap = max(1, r_full - 2)

        A_r_orig = compute_A_r_orig(op, U_cores, r_full, chunk_size=10, n_workers=1, r_cap=r_cap)
        A_r_fast = compute_A_r_fast(op, U_cores, r_full, chunk_size=10, n_workers=1, r_cap=r_cap)

        assert A_r_orig.shape == A_r_fast.shape
        assert np.allclose(A_r_orig, A_r_fast, atol=1e-10)

    def test_rejects_fewer_than_two_dimensions(self) -> None:
        rng = np.random.default_rng(4)
        psi = [rng.normal(size=(3, 10))]
        dpsi = [rng.normal(size=(3, 10))]
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi)
        with pytest.raises(ValueError, match="at least 2"):
            compute_A_r_fast(op, [rng.normal(size=(1, 3, 2))], r=2)


class TestComputeARSigmaNoneMatchesGroundTruth:
    """
    Regression test for a real scale bug: compute_A_r's Sigma=None ("identity")
    fast path used to apply the same -1/(2m) prefactor as a real Sigma matrix,
    but TgStiffnessOperator's own convention uses -1/m when no Sigma is given
    at all (has_sigma=False) -- these are NOT the same thing. This checks
    compute_A_r against an independent ground truth: U^T @ op.to_dense() @ U,
    computed without going through compute_A_r's sample-sweep machinery at all.
    """

    def test_sigma_none_matches_udense_ground_truth(self) -> None:
        from tensor_gedmd.operations import tt_vector_to_dense

        rng = np.random.default_rng(99)
        p, m, dims = 3, 20, [3, 3, 3]
        psi = [rng.normal(size=(n, m)) for n in dims]
        dpsi = [rng.normal(size=(n, m)) for n in dims]

        builder = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)
        U_tt, Sigma_svd, V_core = global_svd_tt(builder.to_tt(), rmax=None, tol=1e-8)
        U_cores = U_tt.cores
        r = U_cores[-1].shape[2]

        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=None)

        A_dense = op.to_dense()
        U_dense = np.zeros((int(np.prod(dims)), r))
        for j in range(r):
            x_cores = [c.copy() for c in U_cores]
            x_cores[-1] = x_cores[-1][:, :, j:j + 1]
            U_dense[:, j] = tt_vector_to_dense(x_cores).reshape(-1)
        A_r_ground_truth = U_dense.T @ A_dense @ U_dense

        A_r = compute_A_r_fast(op, U_cores, r, chunk_size=100, n_workers=1)

        assert np.allclose(A_r, A_r_ground_truth, atol=1e-8)

    def test_real_sigma_still_uses_the_other_prefactor(self) -> None:
        # Sanity check that fixing the Sigma=None case didn't disturb the
        # real-Sigma prefactor (-1/(2m)), which was already correct.
        rng = np.random.default_rng(100)
        p, m, dims = 3, 20, [3, 3, 3]
        psi = [rng.normal(size=(n, m)) for n in dims]
        dpsi = [rng.normal(size=(n, m)) for n in dims]

        builder = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)
        U_tt, Sigma_svd, V_core = global_svd_tt(builder.to_tt(), rmax=None, tol=1e-8)
        U_cores = U_tt.cores
        r = U_cores[-1].shape[2]

        A = rng.normal(size=(p, p))
        Sigma = A @ A.T + np.eye(p)
        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)

        A_r_fast = compute_A_r_fast(op, U_cores, r, chunk_size=100, n_workers=1)
        A_r_orig = compute_A_r_orig(op, U_cores, r, chunk_size=100, n_workers=1)

        assert np.allclose(A_r_fast, A_r_orig, atol=1e-10)
