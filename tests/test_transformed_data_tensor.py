from __future__ import annotations

import numpy as np
import pytest

from tensor_gedmd.reps.transformed_data_tensor import Transformed_Data_Tensor_TT


def _make_psi(dims=(3, 2, 4), m=5, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n, m)) for n in dims]


class TestBuildCoreMatchesBuildTTCores:
    def test_build_core_matches_batch_build(self) -> None:
        psi = _make_psi()
        builder = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)

        cores_batch = builder.build_tt_cores()

        builder2 = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)
        cores_individual = [builder2.build_core(j) for j in range(1, builder2.p + 2)]

        assert len(cores_batch) == len(cores_individual) == 4  # p=3 -> p+1=4 cores
        for a, b in zip(cores_batch, cores_individual):
            assert np.array_equal(a, b)

    def test_iter_tt_cores_matches_build_tt_cores(self) -> None:
        psi = _make_psi()
        builder = Transformed_Data_Tensor_TT(psi=psi)
        cores_batch = builder.build_tt_cores()

        builder2 = Transformed_Data_Tensor_TT(psi=psi)
        cores_iter = list(builder2.iter_tt_cores())

        assert len(cores_iter) == len(cores_batch)
        for a, b in zip(cores_batch, cores_iter):
            assert np.array_equal(a, b)

    def test_final_core_is_identity(self) -> None:
        psi = _make_psi(m=4)
        builder = Transformed_Data_Tensor_TT(psi=psi)
        last = builder.build_core(builder.p + 1)
        assert last.shape == (4, 4, 1)
        assert np.allclose(last[:, :, 0], np.eye(4))

    def test_first_core_shape_and_normalization_toggle(self) -> None:
        psi = _make_psi(dims=(3, 2), m=6)

        b_norm = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=True)
        b_unnorm = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)

        g1_norm = b_norm.build_core(1)
        g1_unnorm = b_unnorm.build_core(1)

        assert g1_norm.shape == g1_unnorm.shape == (1, 3, 6)
        assert np.allclose(g1_norm, g1_unnorm / np.sqrt(6))

    def test_invalid_core_index_raises(self) -> None:
        psi = _make_psi()
        builder = Transformed_Data_Tensor_TT(psi=psi)
        with pytest.raises(ValueError, match="between 1 and"):
            builder.build_core(0)
        with pytest.raises(ValueError, match="between 1 and"):
            builder.build_core(builder.p + 2)


class TestCaching:
    def test_cache_returns_same_array_object(self) -> None:
        psi = _make_psi()
        builder = Transformed_Data_Tensor_TT(psi=psi)

        core_a = builder.build_core(2, use_cache=True)
        core_b = builder.build_core(2, use_cache=True)
        assert core_a is core_b  # served from cache, not rebuilt

    def test_use_cache_false_rebuilds_and_does_not_populate_cache(self) -> None:
        psi = _make_psi()
        builder = Transformed_Data_Tensor_TT(psi=psi)

        core_a = builder.build_core(2, use_cache=False)
        core_b = builder.build_core(2, use_cache=False)
        assert core_a is not core_b  # freshly rebuilt each time
        assert np.array_equal(core_a, core_b)  # but numerically identical
        assert 2 not in builder._core_cache

    def test_clear_cache_forces_rebuild(self) -> None:
        psi = _make_psi()
        builder = Transformed_Data_Tensor_TT(psi=psi)

        core_a = builder.build_core(1, use_cache=True)
        builder.build_tt_cores()
        assert builder.tt_cores

        builder.clear_cache()
        assert builder._core_cache == {}
        assert builder.tt_cores == []

        core_b = builder.build_core(1, use_cache=True)
        assert core_a is not core_b
        assert np.array_equal(core_a, core_b)


class TestReconstructionAgainstDense:
    def test_to_tt_reconstructs_reference_tensor(self) -> None:
        psi = _make_psi(dims=(3, 4), m=5)
        builder = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)
        tt = builder.to_tt()

        core0, core1, core2 = tt.cores
        dense = np.tensordot(core0[0], core1, axes=([-1], [0]))
        dense = np.tensordot(dense, core2, axes=([-1], [0]))
        dense = dense[..., 0]  # squeeze trailing rank-1

        ref = np.einsum("il,jl->ijl", psi[0], psi[1])
        assert dense.shape == ref.shape == (3, 4, 5)
        assert np.allclose(dense, ref)

    def test_sample_tensor_matches_slice_of_reconstruction(self) -> None:
        psi = _make_psi(dims=(3, 2, 4), m=5)
        builder = Transformed_Data_Tensor_TT(psi=psi, normalize_first_core=False)
        builder.build_tt_cores()
        dense = builder.reconstruct_dense()

        for k in range(5):
            assert np.allclose(builder.sample_tensor(k), dense[..., k])
