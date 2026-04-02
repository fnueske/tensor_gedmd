from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from tensor_gedmd.reps.Tensor_Train import TT


ArrayLikeSequence = Sequence[np.ndarray]


@dataclass
class TgStiffnessOperator:
    psi: ArrayLikeSequence
    dpsi: ArrayLikeSequence
    tg_cores: List[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.psi = [np.asarray(a, dtype=float) for a in self.psi]
        self.dpsi = [
            self._normalize_derivative_block(np.asarray(a, dtype=float), k)
            for k, a in enumerate(self.dpsi)
        ]

        self._validate_inputs()

        self.p: int = len(self.psi)
        self.m: int = self.psi[0].shape[1]
        self.local_dims: List[int] = [arr.shape[0] for arr in self.psi]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def build(self) -> TT:
        self.tg_cores = self.build_cores()
        return TT(self.tg_cores, require_right_rank_one=True)

    def build_cores(self) -> List[np.ndarray]:
        """
        Build one TT core per physical dimension.

        Rules:
        - p < 2  -> error
        - p == 2 -> first, last
        - p > 2  -> first, all middle cores, last
        """
        if self.p < 2:
            raise ValueError("We need at least 2 cores.")

        if self.p == 2:
            return [
                self._build_first_core(),
                self._build_last_core(),
            ]

        cores: List[np.ndarray] = [self._build_first_core()]
        for k in range(1, self.p - 1):
            cores.append(self._build_middle_core(k))
        cores.append(self._build_last_core())
        return cores

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        if not isinstance(self.psi, (list, tuple)) or len(self.psi) == 0:
            raise TypeError("psi must be a non-empty list or tuple of numpy arrays.")

        if not isinstance(self.dpsi, (list, tuple)) or len(self.dpsi) == 0:
            raise TypeError("dpsi must be a non-empty list or tuple of numpy arrays.")

        if len(self.psi) != len(self.dpsi):
            raise ValueError("psi and dpsi must have the same length.")

        sample_count = None
        for k, arr in enumerate(self.psi):
            if arr.ndim != 2:
                raise ValueError(f"psi[{k}] must have shape (n_k, m).")

            if sample_count is None:
                sample_count = arr.shape[1]
            elif arr.shape[1] != sample_count:
                raise ValueError("All psi arrays must share the same sample count m.")

        for k, arr in enumerate(self.dpsi):
            if arr.ndim != 2:
                raise ValueError(f"dpsi[{k}] must have shape (n_k, m) after normalization.")

            if arr.shape[1] != sample_count:
                raise ValueError("dpsi has incompatible sample count.")

            if arr.shape[0] != self.psi[k].shape[0]:
                raise ValueError("dpsi and psi must have matching local basis size.")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _normalize_derivative_block(self, arr: np.ndarray, k: int) -> np.ndarray:
        if arr.ndim == 2:
            return arr

        if arr.ndim == 3:
            if arr.shape[1] == 1:
                return arr[:, 0, :]
            if arr.shape[2] == 1:
                return arr[:, :, 0]

        raise ValueError(
            f"Invalid dpsi[{k}] shape {arr.shape}. Expected (n_k, m), (n_k, 1, m), or (n_k, m, 1)."
        )

    def _phi(self, k: int, s: int) -> np.ndarray:
        return self.psi[k][:, s]

    def _dphi(self, k: int, s: int) -> np.ndarray:
        return self.dpsi[k][:, s]

    def _gram_block(self, k: int, s: int) -> np.ndarray:
        v = self._phi(k, s)
        return np.outer(v, v)

    def _stiff_block(self, k: int, s: int) -> np.ndarray:
        dv = self._dphi(k, s)
        return np.outer(dv, dv)

    # -------------------------------------------------------------------------
    # TT core builders
    # -------------------------------------------------------------------------
    def _build_first_core(self) -> np.ndarray:
        n0 = self.local_dims[0]
        core = np.zeros((1, n0, n0, 2 * self.m), dtype=float)

        for s in range(self.m):
            idx = 2 * s
            core[0, :, :, idx] = self._gram_block(0, s)
            core[0, :, :, idx + 1] = self._stiff_block(0, s)

        core /= -float(self.m)
        return core

    def _build_middle_core(self, k: int) -> np.ndarray:
        nk = self.local_dims[k]
        core = np.zeros((2 * self.m, nk, nk, 2 * self.m), dtype=float)

        for s in range(self.m):
            idx = 2 * s
            ek = self._gram_block(k, s)
            fk = self._stiff_block(k, s)

            local = np.zeros((2, nk, nk, 2), dtype=float)
            local[0, :, :, 0] = ek
            local[1, :, :, 1] = ek
            local[0, :, :, 1] = fk

            core[idx:idx + 2, :, :, idx:idx + 2] = local

        return core

    def _build_last_core(self) -> np.ndarray:
        n_last = self.local_dims[-1]
        core = np.zeros((2 * self.m, n_last, n_last, 1), dtype=float)

        for s in range(self.m):
            idx = 2 * s
            core[idx, :, :, 0] = self._stiff_block(self.p - 1, s)
            core[idx + 1, :, :, 0] = self._gram_block(self.p - 1, s)

        return core















