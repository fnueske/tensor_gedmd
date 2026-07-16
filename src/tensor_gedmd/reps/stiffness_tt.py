from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from tensor_gedmd.reps.tensor_train import TT


ArrayLikeSequence = Sequence[np.ndarray]
SigmaLike = Optional[np.ndarray]


@dataclass
class TgStiffnessOperator:
    """
    Build the TT cores of the generator's "stiffness" operator, i.e. the
    term involving second derivatives / the diffusion tensor Sigma.

    Sigma handling
    --------------
    - ``Sigma=None`` (default): diffusion is *not* modeled explicitly and the
      operator reduces to the original, cheap construction (rank 2 per
      sample: a Gram block and a stiffness block, no cross-derivative
      terms). This is unchanged from the previous implementation.
    - ``Sigma`` given as a constant ``(p, p)`` matrix: currently handled the
      *same way* as ``Sigma=None`` -- i.e. the constant case still uses the
      cheap construction above and does not fold Sigma's entries into the
      cores. See the note in ``_prepare_diffusion`` if you actually need
      constant off-diagonal/non-identity diffusion baked into the operator.
    - ``Sigma`` given as a samplewise ``(p, p, m)`` array (i.e. it varies
      per sample): uses the more general construction, which factorizes
      Sigma at each sample and includes the cross-derivative terms.
    """

    psi: ArrayLikeSequence
    dpsi: ArrayLikeSequence
    Sigma: SigmaLike = None
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

        self.sigma_mode, self._Sigma = self._prepare_diffusion(self.Sigma)

        # Caches used only by the variable-Sigma (samplewise) construction.
        self._sample_cache: Dict[int, dict] = {}
        self._first_core_cache: Optional[np.ndarray] = None
        self._last_core_cache: Optional[np.ndarray] = None
        self._middle_blocks_cache: Dict[int, List[np.ndarray]] = {}

    @property
    def Sigma_prepared(self) -> SigmaLike:
        """
        The validated/symmetrized diffusion tensor actually used internally:
        ``None`` if no Sigma was given (implicit identity), a constant
        ``(p, p)`` array, or a samplewise ``(p, p, m)`` array -- consistent
        with ``self.sigma_mode``.
        """
        return self._Sigma

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

        Dispatches on ``self.sigma_mode``:
        - "constant" (Sigma is None, or a constant (p, p) matrix): the
          original cheap construction.
        - "variable" (Sigma is a samplewise (p, p, m) array): the general
          construction with cross-derivative terms.
        """
        if self.p < 2:
            raise ValueError("We need at least 2 cores.")

        if self.sigma_mode == "variable":
            return self._build_cores_variable_sigma()

        return self._build_cores_constant()

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

    def _prepare_diffusion(self, Sigma: SigmaLike):
        """
        Classify Sigma as "constant" or "variable" (samplewise).

        Returns
        -------
        mode : {"constant", "variable"}
        Sigma_prepared : np.ndarray or None
        """
        if Sigma is None:
            return "constant", None

        Sigma = np.asarray(Sigma, dtype=float)

        if Sigma.ndim == 2:
            if Sigma.shape != (self.p, self.p):
                raise ValueError(f"Sigma must have shape ({self.p}, {self.p}), got {Sigma.shape}.")

            Sigma = 0.5 * (Sigma + Sigma.T)

            # NOTE: the constant case is currently dispatched to the same
            # cheap construction used when Sigma is None, i.e. these entries
            # are not folded into the TT cores. If Sigma is not (close to)
            # the identity, this silently ignores real diffusion structure,
            # so we warn rather than fail quietly.
            if not np.allclose(Sigma, np.eye(self.p), atol=1e-12):
                warnings.warn(
                    "A constant, non-identity Sigma was provided, but the constant-Sigma "
                    "path currently reuses the original construction and does not use "
                    "Sigma's entries. Pass a samplewise (p, p, m) Sigma if you need the "
                    "diffusion tensor to actually affect the operator.",
                    stacklevel=2,
                )

            return "constant", Sigma

        if Sigma.ndim == 3:
            if Sigma.shape != (self.p, self.p, self.m):
                raise ValueError(
                    f"Samplewise Sigma must have shape ({self.p}, {self.p}, {self.m}), "
                    f"got {Sigma.shape}."
                )

            Sigma = 0.5 * (Sigma + Sigma.transpose(1, 0, 2))
            return "variable", Sigma

        raise ValueError(f"Sigma must have ndim 2 or 3, got ndim={Sigma.ndim} with shape {Sigma.shape}.")

    # -------------------------------------------------------------------------
    # Shared helpers
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

    # =========================================================================
    # Constant-Sigma / no-Sigma construction (original implementation)
    # =========================================================================
    def _build_cores_constant(self) -> List[np.ndarray]:
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

    # =========================================================================
    # Variable (samplewise) Sigma construction
    # =========================================================================
    def _build_cores_variable_sigma(self) -> List[np.ndarray]:
        cores: List[np.ndarray] = [self.build_first_core()]
        for k in range(1, self.p - 1):
            cores.append(self.build_middle_core_dense(k))
        cores.append(self.build_last_core())
        return cores

    def _sigma_at_sample(self, l: int) -> np.ndarray:
        return self._Sigma[:, :, l]

    def _factor_sigma(self, Sigma_l: np.ndarray, tol: float = 1e-14) -> np.ndarray:
        Sigma_l = 0.5 * (Sigma_l + Sigma_l.T)
        evals, evecs = np.linalg.eigh(Sigma_l)
        keep = evals > tol
        if not np.any(keep):
            return np.zeros((self.p, 0), dtype=float)
        return evecs[:, keep] * np.sqrt(evals[keep])[None, :]

    def _build_sample_data(self, l: int) -> dict:
        if l in self._sample_cache:
            return self._sample_cache[l]

        Sigma_l = self._sigma_at_sample(l)
        L = self._factor_sigma(Sigma_l)
        R = L.shape[1]
        q = 2 + 2 * R

        e, f, g1, h1, g2, h2 = {}, {}, {}, {}, {}, {}
        for k in range(self.p):
            phi_k = self._phi(k, l)
            dphi_k = self._dphi(k, l)

            e[k] = np.outer(phi_k, phi_k)
            f[k] = Sigma_l[k, k] * np.outer(dphi_k, dphi_k)

            g1[k], h1[k], g2[k], h2[k] = [], [], [], []
            for r in range(R):
                coeff = L[k, r]
                g1[k].append(coeff * np.outer(dphi_k, phi_k))
                h1[k].append(coeff * np.outer(phi_k, dphi_k))
                g2[k].append(coeff * np.outer(phi_k, dphi_k))
                h2[k].append(coeff * np.outer(dphi_k, phi_k))

        data = {
            "Sigma_l": Sigma_l, "L": L, "R": R, "q": q,
            "e": e, "f": f, "g1": g1, "h1": h1, "g2": g2, "h2": h2,
        }
        self._sample_cache[l] = data
        return data

    def _sample_rank(self, l: int) -> int:
        return self._build_sample_data(l)["q"]

    def get_sample_ranks(self) -> List[int]:
        return [self._sample_rank(l) for l in range(self.m)]

    def _build_sample_first_core(self, l: int) -> np.ndarray:
        data = self._build_sample_data(l)
        q, R, e, f, g1, g2 = data["q"], data["R"], data["e"], data["f"], data["g1"], data["g2"]

        n0 = self.local_dims[0]
        G0 = np.zeros((1, n0, n0, q), dtype=float)
        G0[0, :, :, 0] = e[0]
        G0[0, :, :, 1] = f[0]
        for r in range(R):
            G0[0, :, :, 2 + r] = g1[0][r]
            G0[0, :, :, 2 + R + r] = g2[0][r]
        return G0

    def _build_sample_middle_block(self, l: int, k: int) -> np.ndarray:
        if not (1 <= k <= self.p - 2):
            raise ValueError(f"k={k} is not a middle site index")

        data = self._build_sample_data(l)
        q, R = data["q"], data["R"]
        e, f, g1, h1, g2, h2 = data["e"], data["f"], data["g1"], data["h1"], data["g2"], data["h2"]

        nk = self.local_dims[k]
        G = np.zeros((q, nk, nk, q), dtype=float)
        G[0, :, :, 0] = e[k]
        G[0, :, :, 1] = f[k]
        G[1, :, :, 1] = e[k]
        for r in range(R):
            i1, i2 = 2 + r, 2 + R + r
            G[0, :, :, i1] = g1[k][r]
            G[i1, :, :, 1] = h1[k][r]
            G[i1, :, :, i1] = e[k]
            G[0, :, :, i2] = g2[k][r]
            G[i2, :, :, 1] = h2[k][r]
            G[i2, :, :, i2] = e[k]
        return G

    def _build_sample_last_core(self, l: int) -> np.ndarray:
        data = self._build_sample_data(l)
        q, R, e, f, h1, h2 = data["q"], data["R"], data["e"], data["f"], data["h1"], data["h2"]

        n_last = self.local_dims[-1]
        k_last = self.p - 1
        Gp = np.zeros((q, n_last, n_last, 1), dtype=float)
        Gp[0, :, :, 0] = f[k_last]
        Gp[1, :, :, 0] = e[k_last]
        for r in range(R):
            Gp[2 + r, :, :, 0] = h1[k_last][r]
            Gp[2 + R + r, :, :, 0] = h2[k_last][r]
        return Gp

    def build_first_core(self) -> np.ndarray:
        if self._first_core_cache is not None:
            return self._first_core_cache

        pref = -(1.0 / (2.0 * self.m))
        G0 = np.concatenate([pref * self._build_sample_first_core(l) for l in range(self.m)], axis=3)
        self._first_core_cache = G0
        return G0

    def build_last_core(self) -> np.ndarray:
        if self._last_core_cache is not None:
            return self._last_core_cache

        Gp = np.concatenate([self._build_sample_last_core(l) for l in range(self.m)], axis=0)
        self._last_core_cache = Gp
        return Gp

    def build_middle_blocks(self, k: int) -> List[np.ndarray]:
        if not (1 <= k <= self.p - 2):
            raise ValueError(f"build_middle_blocks(k) requires a middle site, got k={k}")

        if k in self._middle_blocks_cache:
            return self._middle_blocks_cache[k]

        blocks_k = [self._build_sample_middle_block(l, k) for l in range(self.m)]
        self._middle_blocks_cache[k] = blocks_k
        return blocks_k

    def build_middle_core_dense(self, k: int) -> np.ndarray:
        if not (1 <= k <= self.p - 2):
            raise ValueError(f"k={k} is not a middle site index")

        blocks_k = self.build_middle_blocks(k)
        nk = self.local_dims[k]
        sample_ranks = [blk.shape[0] for blk in blocks_k]
        total_rank = sum(sample_ranks)

        G = np.zeros((total_rank, nk, nk, total_rank), dtype=float)
        start = 0
        for blk, ql in zip(blocks_k, sample_ranks):
            G[start:start + ql, :, :, start:start + ql] = blk
            start += ql
        return G

    def clear_cache(self) -> None:
        self._sample_cache.clear()
        self._first_core_cache = None
        self._last_core_cache = None
        self._middle_blocks_cache.clear()
        self.tg_cores = []















