"""
TgStiffnessOperator: TT representation of the (possibly diffusion-weighted)
stiffness/generator operator, unifying the constant and variable/samplewise
diffusion-tensor (Sigma) cases into a single band-based construction.

Design
------
The middle core of the TT operator is, before summing over the m samples,
block-diagonal: sample l occupies rows [l*r_left:(l+1)*r_left] and columns
[l*r_right:(l+1)*r_right]. Within that (r_left, r_right) block there is a
small, fixed, SAMPLE-INDEPENDENT set of nonzero positions ("bands"): for the
no-Sigma case the bands are (0,0)->e, (1,1)->e, (0,1)->f, always at rank 2.
For a real (constant or samplewise) Sigma there are more bands and the rank
grows by 2 with every dimension processed -- but the *set of (src, dst)
positions* at a given dimension is still exactly the same for every sample;
only the (n, n) content differs per sample.

So instead of ever building the huge (m*r_left, n, n, m*r_right) block-
diagonal array, ``prepare_middle_site`` builds, once per dimension, a list
of bands; each band stores just an (m, n, n) stack (one (n, n) matrix per
sample) plus its fixed (src, dst) position. The matvec in
``tensor_gedmd.algorithms.mat_vec_prod`` (``prepare_operator`` +
``tt_matrix_vector_product``) uses exactly this compact representation, so
the (m*r_left, n, n, m*r_right) array is never materialized for real use;
``build_cores()``/``to_dense()`` below still build it, but are intended only
for validation on small problems.

Sigma handling
--------------
- ``Sigma=None``: no diffusion tensor at all; every internal bond has the
  cheap fixed rank 2, with no cross-derivative terms (equivalent to an
  isotropic/unit diffusion with no dimension coupling).
- ``Sigma`` as a constant ``(p, p)`` matrix: still routed through the SAME
  general band construction as the samplewise case below (just reusing the
  same Sigma matrix for every sample), so a non-identity constant Sigma is
  now genuinely incorporated into the operator -- unlike a previous version
  of this class, which discarded a constant Sigma's actual values.
- ``Sigma`` as a samplewise ``(p, p, m)`` array (varies per sample): full
  general construction, factoring in every sample's own diffusion matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tensor_gedmd.reps.tensor_train import TT

ArrayLikeSequence = Sequence[np.ndarray]
SigmaLike = Optional[np.ndarray]


class TgStiffnessOperator:
    """
    Parameters
    ----------
    psi : list of np.ndarray
        psi[k] has shape (n_k, m), k = 0, ..., p-1 (0-indexed physical
        dimensions), matching the convention used throughout this package.
    dpsi : list of np.ndarray
        dpsi[k] has shape (n_k, m), (n_k, 1, m), or (n_k, m, 1); the latter
        two are normalized to (n_k, m) at construction time.
    Sigma : None, (p, p) array, or (p, p, m) array, optional
        Diffusion tensor. See the module docstring for how each case is
        handled.
    """

    def __init__(
        self,
        psi: ArrayLikeSequence,
        dpsi: ArrayLikeSequence,
        Sigma: SigmaLike = None,
    ) -> None:
        self.psi = [np.asarray(a, dtype=float) for a in psi]
        self.dpsi = [
            self._normalize_derivative_block(np.asarray(a, dtype=float), k)
            for k, a in enumerate(dpsi)
        ]

        self._validate_inputs()

        self.p: int = len(self.psi)
        self.m: int = self.psi[0].shape[1]
        self.local_dims: List[int] = [arr.shape[0] for arr in self.psi]

        self.has_sigma: bool = Sigma is not None
        if self.has_sigma:
            self._Sigma_kind, self.Sigma = self._prepare_diffusion(Sigma)
        else:
            self._Sigma_kind, self.Sigma = None, None

        # Match compute_A_r's / this package's existing convention: "constant"
        # covers both Sigma=None (implicit identity) and a real constant
        # (p, p) matrix; "variable" covers samplewise (p, p, m).
        self.sigma_mode: str = "variable" if self._Sigma_kind == "samplewise" else "constant"

        self.scale: float = -(1.0 / self.m) if not self.has_sigma else -(1.0 / (2.0 * self.m))

        self.tt_cores: List[np.ndarray] = []  # only populated by build_cores()

        # Caches (avoid recomputation across repeated matvecs / calls).
        self._site_weight_cache: Dict[int, List[dict]] = {}
        self._first_core_cache: Optional[np.ndarray] = None
        self._last_core_cache: Optional[np.ndarray] = None
        self._middle_site_cache: Dict[int, dict] = {}

    @property
    def Sigma_prepared(self) -> SigmaLike:
        """
        The validated/symmetrized diffusion tensor actually used internally:
        ``None`` if no Sigma was given, a constant ``(p, p)`` array, or a
        samplewise ``(p, p, m)`` array -- consistent with ``self.sigma_mode``.
        """
        return self.Sigma

    # -------------------------------------------------------------------------
    # Validation / setup
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

    def _prepare_diffusion(self, Sigma: np.ndarray) -> Tuple[str, np.ndarray]:
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.ndim == 2:
            if Sigma.shape != (self.p, self.p):
                raise ValueError(f"Sigma must have shape ({self.p}, {self.p}), got {Sigma.shape}.")
            Sigma = 0.5 * (Sigma + Sigma.T)
            return "constant", Sigma
        if Sigma.ndim == 3:
            if Sigma.shape != (self.p, self.p, self.m):
                raise ValueError(
                    f"Samplewise Sigma must have shape ({self.p}, {self.p}, {self.m}), "
                    f"got {Sigma.shape}."
                )
            Sigma = 0.5 * (Sigma + Sigma.transpose(1, 0, 2))
            return "samplewise", Sigma
        raise ValueError(f"Sigma must have ndim 2 or 3, got ndim={Sigma.ndim} with shape {Sigma.shape}.")

    def _sigma_at_sample(self, l: int) -> np.ndarray:
        if self._Sigma_kind == "constant":
            return self.Sigma
        return self.Sigma[:, :, l]

    # -------------------------------------------------------------------------
    # Structural (sample-independent) rank / band info.
    #
    # `t` below always means a BOND POSITION (a count of how many dimensions
    # have been processed so far), not a dimension label: t=0 is the left
    # boundary (before dimension 0), t=p is the right boundary (after
    # dimension p-1). Internal bonds sit at t=1,...,p-1.
    # -------------------------------------------------------------------------
    def local_rank(self, t: int) -> int:
        """Bond rank at position t (t=0,...,p; t=0 and t=p are always 1)."""
        if t == 0 or t == self.p:
            return 1
        if not self.has_sigma:
            return 2
        return 2 + 2 * t

    @staticmethod
    def _g_index(j: int) -> int:
        return 2 + j

    @staticmethod
    def _h_index(t: int, j: int) -> int:
        return 2 + t + j

    def middle_site_bands(self, k: int) -> List[Tuple[int, int, str, Optional[int]]]:
        """
        Structural list of (src_local, dst_local, kind, j) bands for the
        middle core at 0-indexed dimension k (1 <= k <= p-2). Sample-independent.
        kind in {'e', 'f', 'g', 'h', 'h_sigma', 'g_sigma'}.
        """
        bands: List[Tuple[int, int, str, Optional[int]]] = [
            (0, 0, 'e', None), (1, 1, 'e', None), (0, 1, 'f', None)
        ]
        if self.has_sigma:
            for j in range(0, k):
                gL = self._g_index(j)
                hL = self._h_index(k, j)
                hR = self._h_index(k + 1, j)
                bands.append((gL, gL, 'e', None))       # propagate open g_j
                bands.append((hL, hR, 'e', None))        # propagate open h_j (shifts)
                bands.append((gL, 1, 'h_sigma', j))       # close g_j x new h_k
                bands.append((hL, 1, 'g_sigma', j))       # close h_j x new g_k
            bands.append((0, self._g_index(k), 'g', None))          # open new g_k
            bands.append((0, self._h_index(k + 1, k), 'h', None))   # open new h_k
        return bands

    # -------------------------------------------------------------------------
    # Per-sample weight matrices at dimension k (cached across calls).
    # -------------------------------------------------------------------------
    def _sample_site_weights(self, k: int) -> List[dict]:
        if k in self._site_weight_cache:
            return self._site_weight_cache[k]

        out = []
        for l in range(self.m):
            psi_k = self.psi[k][:, l]
            dpsi_k = self.dpsi[k][:, l]
            e_k = np.outer(psi_k, psi_k)
            if self.has_sigma:
                Sigma_l = self._sigma_at_sample(l)
                f_k = Sigma_l[k, k] * np.outer(dpsi_k, dpsi_k)
                g_k = np.outer(dpsi_k, psi_k)
                h_k = np.outer(psi_k, dpsi_k)
            else:
                f_k = np.outer(dpsi_k, dpsi_k)
                g_k = h_k = None
            out.append({'e': e_k, 'f': f_k, 'g': g_k, 'h': h_k})

        self._site_weight_cache[k] = out
        return out

    # -------------------------------------------------------------------------
    # First / last cores: only O(m) to build, so built+concatenated directly.
    # -------------------------------------------------------------------------
    def first_core(self) -> np.ndarray:
        if self._first_core_cache is not None:
            return self._first_core_cache

        n0 = self.local_dims[0]
        r_right = self.local_rank(1)
        w = self._sample_site_weights(0)

        G0 = np.zeros((1, n0, n0, self.m * r_right), dtype=float)
        for l in range(self.m):
            off = l * r_right
            G0[0, :, :, off + 0] = w[l]['e']
            G0[0, :, :, off + 1] = w[l]['f']
            if self.has_sigma:
                G0[0, :, :, off + self._g_index(0)] = w[l]['g']
                G0[0, :, :, off + self._h_index(1, 0)] = w[l]['h']

        G0 *= self.scale
        self._first_core_cache = G0
        return G0

    def last_core(self) -> np.ndarray:
        if self._last_core_cache is not None:
            return self._last_core_cache

        n_last = self.local_dims[-1]
        p_last = self.p - 1
        r_left = self.local_rank(self.p - 1)
        w = self._sample_site_weights(p_last)

        Gp = np.zeros((self.m * r_left, n_last, n_last, 1), dtype=float)
        for l in range(self.m):
            off = l * r_left
            Gp[off + 0, :, :, 0] = w[l]['f']
            Gp[off + 1, :, :, 0] = w[l]['e']
            if self.has_sigma:
                Sigma_l = self._sigma_at_sample(l)
                for j in range(0, p_last):
                    gL = self._g_index(j)
                    hL = self._h_index(self.p - 1, j)
                    Gp[off + gL, :, :, 0] = Sigma_l[j, p_last] * w[l]['h']
                    Gp[off + hL, :, :, 0] = Sigma_l[p_last, j] * w[l]['g']

        self._last_core_cache = Gp
        return Gp

    # -------------------------------------------------------------------------
    # Middle dimension k (1 <= k <= p-2): band data, stacked (m, n, n) per
    # band. NEVER builds the (m*r_left, n, n, m*r_right) dense array.
    # -------------------------------------------------------------------------
    def prepare_middle_site(self, k: int) -> dict:
        if k in self._middle_site_cache:
            return self._middle_site_cache[k]

        bands = self.middle_site_bands(k)
        r_left = self.local_rank(k)
        r_right = self.local_rank(k + 1)
        nk = self.local_dims[k]

        w = self._sample_site_weights(k)
        Sigma_all = [self._sigma_at_sample(l) for l in range(self.m)] if self.has_sigma else None

        band_stacks = []
        for (src, dst, kind, j) in bands:
            if kind == 'e':
                M = np.stack([w[l]['e'] for l in range(self.m)], axis=0)
            elif kind == 'f':
                M = np.stack([w[l]['f'] for l in range(self.m)], axis=0)
            elif kind == 'g':
                M = np.stack([w[l]['g'] for l in range(self.m)], axis=0)
            elif kind == 'h':
                M = np.stack([w[l]['h'] for l in range(self.m)], axis=0)
            elif kind == 'h_sigma':
                M = np.stack(
                    [Sigma_all[l][j, k] * w[l]['h'] for l in range(self.m)], axis=0
                )
            elif kind == 'g_sigma':
                M = np.stack(
                    [Sigma_all[l][k, j] * w[l]['g'] for l in range(self.m)], axis=0
                )
            else:
                raise ValueError(f"unknown band kind {kind}")

            mask = ~(M == 0).all(axis=(1, 2))
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            # Stored as-is (axis1=out/row, axis2=in/col), matching the M@v
            # convention used by the first/last-core tensordot. e and f
            # happen to be symmetric so this coincides with M^T for those,
            # but g/h are NOT symmetric, so no transpose is applied here.
            B = np.ascontiguousarray(M)
            band_stacks.append((src, dst, B, idx))

        prepared = {
            'r_left': r_left,
            'r_right': r_right,
            'n': nk,
            'bands': band_stacks,
            'm': self.m,
        }
        self._middle_site_cache[k] = prepared
        return prepared

    # -------------------------------------------------------------------------
    # Dense reconstruction (validation / small problems only -- this DOES
    # materialize the full block-diagonal middle cores).
    # -------------------------------------------------------------------------
    def _dense_middle_core(self, k: int) -> np.ndarray:
        prepared = self.prepare_middle_site(k)
        r_left, r_right, nk = prepared['r_left'], prepared['r_right'], prepared['n']
        G = np.zeros((self.m * r_left, nk, nk, self.m * r_right), dtype=float)
        for (src, dst, B, idx) in prepared['bands']:
            for l in idx:
                off_l = l * r_left
                off_r = l * r_right
                G[off_l + src, :, :, off_r + dst] = B[l]
        return G

    def build(self) -> TT:
        """Convenience wrapper: ``TT(self.build_cores(), require_right_rank_one=True)``."""
        cores = self.build_cores()
        return TT(cores, require_right_rank_one=True)

    def build_cores(self) -> List[np.ndarray]:
        """
        Build FULL dense TT cores (block-diagonal middle cores included).
        Only use for validation / small (m, p, ranks); for real use, prefer
        ``tensor_gedmd.algorithms.mat_vec_prod.make_A_mv(op, ...)``, which
        never materializes these.
        """
        if self.p < 2:
            raise ValueError("We need at least 2 cores.")

        cores = [self.first_core()]
        for k in range(1, self.p - 1):
            cores.append(self._dense_middle_core(k))
        cores.append(self.last_core())
        self.tt_cores = cores
        return cores

    def get_sample_ranks(self) -> List[int]:
        """Bond rank at every internal position 1,...,p-1 (structural, same
        for every sample; total assembled rank at that bond = m * rank)."""
        return [self.local_rank(t) for t in range(1, self.p)]

    def to_dense(self) -> np.ndarray:
        if not self.tt_cores:
            self.build_cores()
        T = self.tt_cores[0]
        for k in range(1, self.p):
            T = np.tensordot(T, self.tt_cores[k], axes=([T.ndim - 1], [0]))
        T = np.squeeze(T, axis=(0, -1))
        axes_row = list(range(0, 2 * self.p, 2))
        axes_col = list(range(1, 2 * self.p, 2))
        T = np.transpose(T, axes_row + axes_col)
        row_dim = int(np.prod(self.local_dims))
        return T.reshape(row_dim, row_dim)

    def build_dense_direct(self) -> np.ndarray:
        """
        Ground-truth dense operator, built directly (no TT structure at
        all): O(m * p^2) work via explicit Kronecker products. Only for
        validating the TT construction on small problems.
        """
        N = int(np.prod(self.local_dims))
        A = np.zeros((N, N), dtype=float)
        for l in range(self.m):
            vs = []
            for i in range(self.p):
                local = []
                for k in range(self.p):
                    if k == i:
                        local.append(self.dpsi[k][:, l])
                    else:
                        local.append(self.psi[k][:, l])
                v = local[0]
                for arr in local[1:]:
                    v = np.kron(v, arr)
                vs.append(v)

            if self.has_sigma:
                Sigma_l = self._sigma_at_sample(l)
                for i in range(self.p):
                    for j in range(self.p):
                        A += Sigma_l[i, j] * np.outer(vs[i], vs[j])
            else:
                for i in range(self.p):
                    A += np.outer(vs[i], vs[i])

        A *= self.scale
        return 0.5 * (A + A.T)

    def clear_cache(self) -> None:
        self._site_weight_cache.clear()
        self._first_core_cache = None
        self._last_core_cache = None
        self._middle_site_cache.clear()
        self.tt_cores = []
