from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tensor_gedmd.algorithms.mat_vec_prod import (
    TTInnerProductMixin,
    extract_tt_column,
    make_A_mv,
    prepare_blocks,
    tt_inner_product,
    tt_matrix_to_dense,
    tt_matrix_vector_product_csr_prepared,
    tt_matrix_vector_product_general,
    tt_norm,
    tt_vector_to_dense,
)


# =========================================================
# Helpers
# =========================================================
@dataclass
class DummyGeneratorOp:
    tg_cores: list[np.ndarray]


class DummyTTVector(TTInnerProductMixin):
    def __init__(self, tt_cores: list[np.ndarray]) -> None:
        self.tt_cores = tt_cores


def random_tt_vector(
    dims: list[int],
    ranks: list[int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Build a random TT vector with cores of shape (r_{k-1}, n_k, r_k).
    """
    if len(ranks) != len(dims) + 1:
        raise ValueError("ranks must have length len(dims) + 1.")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("TT boundary ranks must be 1.")

    return [
        rng.normal(size=(ranks[k], dims[k], ranks[k + 1]))
        for k in range(len(dims))
    ]


def random_tt_matrix(
    out_dims: list[int],
    in_dims: list[int],
    ranks: list[int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Build a random TT matrix with cores of shape (r_{k-1}, n_k, m_k, r_k).
    """
    if len(out_dims) != len(in_dims):
        raise ValueError("out_dims and in_dims must have the same length.")
    if len(ranks) != len(out_dims) + 1:
        raise ValueError("ranks must have length len(out_dims) + 1.")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("TT boundary ranks must be 1.")

    return [
        rng.normal(size=(ranks[k], out_dims[k], in_dims[k], ranks[k + 1]))
        for k in range(len(out_dims))
    ]


# =========================================================
# Pytest fixture
# =========================================================
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


# =========================================================
# Tests: 4-core general mat-vec and error norms
# =========================================================
class TestGeneralMatVec4Cores:
    def test_general_matvec_matches_dense_random_4cores(self, rng: np.random.Generator) -> None:
        out_dims = [2, 3, 2, 2]
        in_dims = [2, 2, 3, 2]

        M = random_tt_matrix(out_dims, in_dims, [1, 2, 3, 2, 1], rng)
        x = random_tt_vector(in_dims, [1, 2, 2, 2, 1], rng)

        y_tt = tt_matrix_vector_product_general(M, x)

        y_dense_tt = tt_vector_to_dense(y_tt).reshape(-1)
        M_dense = tt_matrix_to_dense(M).reshape(np.prod(out_dims), np.prod(in_dims))
        x_dense = tt_vector_to_dense(x).reshape(-1)
        y_dense_ref = M_dense @ x_dense

        abs_err = np.linalg.norm(y_dense_tt - y_dense_ref)
        rel_err = abs_err / max(np.linalg.norm(y_dense_ref), 1e-15)

        print("\n[4-core general] absolute error norm =", abs_err)
        print("[4-core general] relative error norm =", rel_err)

        assert np.allclose(y_dense_tt, y_dense_ref, atol=1e-10, rtol=1e-10)
        assert rel_err < 1e-10

    def test_general_matvec_norm_matches_dense_4cores(self, rng: np.random.Generator) -> None:
        dims = [2, 3, 2, 2]
        x = random_tt_vector(dims, [1, 2, 2, 2, 1], rng)

        tt_val = tt_norm(x)
        dense_val = float(np.linalg.norm(tt_vector_to_dense(x).reshape(-1)))
        err = abs(tt_val - dense_val)

        print("\n[4-core norm] |tt_norm - dense_norm| =", err)

        assert np.isclose(tt_val, dense_val, atol=1e-10, rtol=1e-10)


# =========================================================
# Tests: prepared blocks
# =========================================================
class TestPreparedBlocks:
    def test_prepare_blocks_basic(self) -> None:
        blocks = []
        for i in range(3):
            blk = np.zeros((2, 4, 4, 2), dtype=float)
            if i == 0:
                blk[0, :, :, 0] = np.eye(4)
            if i == 1:
                blk[1, :, :, 1] = 2.0 * np.eye(4)
            if i == 2:
                blk[0, :, :, 1] = np.ones((4, 4))
            blocks.append(blk)

        prepared = prepare_blocks(blocks, m=3)

        assert prepared.BT00.shape == (3, 4, 4)
        assert prepared.BT11.shape == (3, 4, 4)
        assert prepared.BT01.shape == (3, 4, 4)

        assert np.array_equal(prepared.idx00, np.array([0]))
        assert np.array_equal(prepared.idx11, np.array([1]))
        assert np.array_equal(prepared.idx01, np.array([2]))

    def test_prepare_blocks_rejects_invalid_m(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            prepare_blocks([], m=0)

    def test_prepare_blocks_rejects_short_blocks(self) -> None:
        blocks = [np.zeros((2, 3, 3, 2))]
        with pytest.raises(ValueError, match="must contain at least"):
            prepare_blocks(blocks, m=2)


# =========================================================
# Tests: prepared mat-vec
# =========================================================
class TestPreparedMatVec:
    def test_prepared_matvec_matches_general_for_d2(self, rng: np.random.Generator) -> None:
        out_dims = [2, 3]
        in_dims = [2, 2]

        M = random_tt_matrix(out_dims, in_dims, [1, 2, 1], rng)
        x = random_tt_vector(in_dims, [1, 2, 1], rng)

        y_general = tt_matrix_vector_product_general(M, x)
        y_prepared = tt_matrix_vector_product_csr_prepared(M, x)

        y_general_dense = tt_vector_to_dense(y_general).reshape(-1)
        y_prepared_dense = tt_vector_to_dense(y_prepared).reshape(-1)

        abs_err = np.linalg.norm(y_general_dense - y_prepared_dense)
        rel_err = abs_err / max(np.linalg.norm(y_general_dense), 1e-15)

        print("\n[d=2 specialized] absolute error norm =", abs_err)
        print("[d=2 specialized] relative error norm =", rel_err)

        assert np.allclose(y_general_dense, y_prepared_dense, atol=1e-10, rtol=1e-10)

    def test_prepared_matvec_timing_output_for_d2(self, rng: np.random.Generator) -> None:
        M = random_tt_matrix([2, 2], [2, 2], [1, 2, 1], rng)
        x = random_tt_vector([2, 2], [1, 2, 1], rng)

        y_tt, timing = tt_matrix_vector_product_csr_prepared(M, x, timing=True)

        assert isinstance(y_tt, list)
        assert isinstance(timing, dict)
        assert "total" in timing
        assert timing["total"] >= 0.0

    def test_prepared_matvec_requires_prepared_when_d_gt_2(self, rng: np.random.Generator) -> None:
        M = random_tt_matrix([2, 2, 2, 2], [2, 2, 2, 2], [1, 2, 2, 2, 1], rng)
        x = random_tt_vector([2, 2, 2, 2], [1, 2, 2, 2, 1], rng)

        with pytest.raises(ValueError, match="prepared"):
            tt_matrix_vector_product_csr_prepared(M, x, prepared=None)

    def test_prepared_matvec_raises_on_core_count_mismatch(self) -> None:
        M = [np.zeros((1, 2, 2, 1)), np.zeros((1, 2, 2, 1))]
        x = [np.zeros((1, 2, 1))]

        with pytest.raises(ValueError, match="same number of cores"):
            tt_matrix_vector_product_csr_prepared(M, x)

    def test_prepared_matvec_raises_on_single_core(self) -> None:
        M = [np.zeros((1, 2, 2, 1))]
        x = [np.zeros((1, 2, 1))]

        with pytest.raises(ValueError, match="At least 2 TT cores are required|at least two TT cores"):
            tt_matrix_vector_product_csr_prepared(M, x)


# =========================================================
# Tests: make_A_mv
# =========================================================
class TestMakeAMV:
    def test_make_A_mv_general_4cores(self, rng: np.random.Generator) -> None:
        out_dims = [2, 3, 2, 2]
        in_dims = [2, 2, 3, 2]

        M = random_tt_matrix(out_dims, in_dims, [1, 2, 3, 2, 1], rng)
        x = random_tt_vector(in_dims, [1, 2, 2, 2, 1], rng)

        op = DummyGeneratorOp(tg_cores=M)
        A_mv = make_A_mv(op, use_general=True)

        y_tt = A_mv(x)
        y_dense = tt_vector_to_dense(y_tt).reshape(-1)
        M_dense = tt_matrix_to_dense(M).reshape(np.prod(out_dims), np.prod(in_dims))
        x_dense = tt_vector_to_dense(x).reshape(-1)

        abs_err = np.linalg.norm(y_dense - (M_dense @ x_dense))
        rel_err = abs_err / max(np.linalg.norm(M_dense @ x_dense), 1e-15)

        print("\n[make_A_mv general 4-core] absolute error norm =", abs_err)
        print("[make_A_mv general 4-core] relative error norm =", rel_err)

        assert np.allclose(y_dense, M_dense @ x_dense, atol=1e-10, rtol=1e-10)

    def test_make_A_mv_specialized_d2(self, rng: np.random.Generator) -> None:
        M = random_tt_matrix([2, 2], [2, 2], [1, 2, 1], rng)
        x = random_tt_vector([2, 2], [1, 2, 1], rng)

        op = DummyGeneratorOp(tg_cores=M)
        A_mv = make_A_mv(op, use_general=False)

        y_tt = A_mv(x)
        y_dense = tt_vector_to_dense(y_tt).reshape(-1)
        M_dense = tt_matrix_to_dense(M).reshape(4, 4)
        x_dense = tt_vector_to_dense(x).reshape(-1)

        assert np.allclose(y_dense, M_dense @ x_dense, atol=1e-10, rtol=1e-10)

    def test_make_A_mv_requires_blocks_for_specialized_middle_cores_4cores(
        self,
        rng: np.random.Generator,
    ) -> None:
        M = random_tt_matrix([2, 2, 2, 2], [2, 2, 2, 2], [1, 2, 2, 2, 1], rng)
        op = DummyGeneratorOp(tg_cores=M)

        with pytest.raises(ValueError, match="blocks"):
            make_A_mv(op, use_general=False)

    def test_make_A_mv_rejects_single_core_specialized(self, rng: np.random.Generator) -> None:
        M = random_tt_matrix([2], [2], [1, 1], rng)
        op = DummyGeneratorOp(tg_cores=M)

        with pytest.raises(ValueError, match="At least 2 TT cores are required|at least 2 TT cores"):
            make_A_mv(op, use_general=False)

    def test_make_A_mv_rejects_missing_tg_cores(self) -> None:
        class BadOp:
            pass

        with pytest.raises(ValueError, match="tg_cores"):
            make_A_mv(BadOp(), use_general=True)


# =========================================================
# Tests: inner product and norm
# =========================================================
class TestInnerProductAndNorm:
    def test_tt_inner_product_matches_dense_4cores(self, rng: np.random.Generator) -> None:
        dims = [2, 3, 2, 2]

        A = random_tt_vector(dims, [1, 2, 2, 2, 1], rng)
        B = random_tt_vector(dims, [1, 3, 2, 2, 1], rng)

        got = tt_inner_product(A, B)
        ref = float(
            np.dot(
                tt_vector_to_dense(A).reshape(-1),
                tt_vector_to_dense(B).reshape(-1),
            )
        )

        err = abs(got - ref)
        print("\n[4-core inner product] |tt - dense| =", err)

        assert np.isclose(got, ref, atol=1e-10, rtol=1e-10)

    def test_tt_norm_matches_dense_4cores(self, rng: np.random.Generator) -> None:
        x = random_tt_vector([2, 3, 2, 2], [1, 2, 2, 2, 1], rng)

        got = tt_norm(x)
        ref = float(np.linalg.norm(tt_vector_to_dense(x).reshape(-1)))

        err = abs(got - ref)
        print("\n[4-core tt_norm] |tt_norm - dense_norm| =", err)

        assert np.isclose(got, ref, atol=1e-10, rtol=1e-10)

    def test_tt_inner_product_self_equals_norm_squared_4cores(self, rng: np.random.Generator) -> None:
        x = random_tt_vector([2, 2, 3, 2], [1, 2, 2, 2, 1], rng)

        ip = tt_inner_product(x, x)
        norm_sq = tt_norm(x) ** 2

        err = abs(ip - norm_sq)
        print("\n[4-core self ip vs norm^2] error =", err)

        assert np.isclose(ip, norm_sq, atol=1e-10, rtol=1e-10)

    def test_tt_inner_product_raises_on_different_lengths(self, rng: np.random.Generator) -> None:
        A = random_tt_vector([2, 2], [1, 2, 1], rng)
        B = random_tt_vector([2, 2, 2], [1, 2, 2, 1], rng)

        with pytest.raises(ValueError, match="same number of cores"):
            tt_inner_product(A, B)

    def test_tt_inner_product_raises_on_mode_mismatch(self, rng: np.random.Generator) -> None:
        A = random_tt_vector([2, 3], [1, 2, 1], rng)
        B = random_tt_vector([2, 4], [1, 2, 1], rng)

        with pytest.raises(ValueError, match="Mode mismatch"):
            tt_inner_product(A, B)

    def test_matmul_operator_via_mixin(self, rng: np.random.Generator) -> None:
        A = DummyTTVector(random_tt_vector([2, 2, 2, 2], [1, 2, 2, 2, 1], rng))
        B = random_tt_vector([2, 2, 2, 2], [1, 2, 2, 2, 1], rng)

        got = A @ B
        ref = tt_inner_product(A.tt_cores, B)

        assert np.isclose(got, ref, atol=1e-10, rtol=1e-10)


# =========================================================
# Tests: extract_tt_column
# =========================================================
class TestExtractTTColumn:
    def test_extract_tt_column_basic(self) -> None:
        U = [
            np.ones((1, 2, 1), dtype=float),
            np.arange(12.0).reshape(2, 3, 2),
        ]

        col = extract_tt_column(U, 1)

        assert len(col) == 2
        assert col[-1].shape == (2, 3, 1)
        assert np.allclose(col[-1][:, :, 0], U[-1][:, :, 1])

    def test_extract_tt_column_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="must contain at least one core"):
            extract_tt_column([], 0)

    def test_extract_tt_column_raises_on_bad_last_core(self) -> None:
        with pytest.raises(ValueError, match="last core must have shape"):
            extract_tt_column([np.zeros((2, 2))], 0)

    def test_extract_tt_column_raises_on_bad_index(self) -> None:
        with pytest.raises(IndexError, match="out of bounds"):
            extract_tt_column([np.zeros((1, 2, 3))], 7)


# =========================================================
# New: generalized banded matvec (constant + variable Sigma)
# via TgStiffnessOperator + prepare_operator + tt_matrix_vector_product
# =========================================================
from tensor_gedmd.reps.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.algorithms.mat_vec_prod import prepare_operator, tt_matrix_vector_product


class TestStiffnessOperatorBandedMatVec:
    @pytest.mark.parametrize("sigma_case", ["none", "constant", "variable"])
    @pytest.mark.parametrize("p,dims", [(2, [3, 4]), (3, [2, 3, 2])])
    def test_matches_dense_reference(self, sigma_case, p, dims, rng: np.random.Generator) -> None:
        m = 5
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
        A_dense = op.to_dense()

        x_cores = random_tt_vector(dims, [1] + [3] * (p - 1) + [1], rng)
        x_dense = tt_vector_to_dense(x_cores).reshape(-1)

        prepared = prepare_operator(op)
        y_cores = tt_matrix_vector_product(prepared, x_cores, max_rank=10_000, tolerance=1e-14)
        y_dense = tt_vector_to_dense(y_cores).reshape(-1)

        y_ref = A_dense @ x_dense
        rel_err = np.linalg.norm(y_dense - y_ref) / max(np.linalg.norm(y_ref), 1e-15)
        assert rel_err < 1e-8

    def test_make_A_mv_dispatches_to_banded_path_for_stiffness_operator(
        self, rng: np.random.Generator
    ) -> None:
        p, m, dims = 3, 5, [2, 3, 2]
        psi = [rng.normal(size=(n, m)) for n in dims]
        dpsi = [rng.normal(size=(n, m)) for n in dims]
        Sigma = np.zeros((p, p, m))
        for l in range(m):
            A = rng.normal(size=(p, p))
            Sigma[:, :, l] = A @ A.T + np.eye(p)

        op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
        A_mv = make_A_mv(op, max_rank=10_000, tolerance=1e-14)

        x_cores = random_tt_vector(dims, [1, 3, 3, 1], rng)
        y_cores = A_mv(x_cores)
        y_dense = tt_vector_to_dense(y_cores).reshape(-1)

        y_ref = op.to_dense() @ tt_vector_to_dense(x_cores).reshape(-1)
        rel_err = np.linalg.norm(y_dense - y_ref) / max(np.linalg.norm(y_ref), 1e-15)
        assert rel_err < 1e-8

    def test_make_A_mv_still_dispatches_to_legacy_path_for_plain_tg_cores(
        self, rng: np.random.Generator
    ) -> None:
        # Existing (pre-TgStiffnessOperator) usage must still work unchanged:
        # an object with a plain .tg_cores list, not a TgStiffnessOperator.
        M = random_tt_matrix([2, 2], [2, 2], [1, 2, 1], rng)
        op = DummyGeneratorOp(tg_cores=M)
        A_mv = make_A_mv(op, use_general=True)

        x = random_tt_vector([2, 2], [1, 2, 1], rng)
        y_dense = tt_vector_to_dense(A_mv(x)).reshape(-1)
        M_dense = tt_matrix_to_dense(M).reshape(4, 4)
        x_dense = tt_vector_to_dense(x).reshape(-1)

        assert np.allclose(y_dense, M_dense @ x_dense, atol=1e-10, rtol=1e-10)
