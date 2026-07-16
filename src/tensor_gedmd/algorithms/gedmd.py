"""
General end-to-end tensor-train gEDMD pipeline.

This module wires together the rest of the package into a single entry
point, ``run_gedmd_pipeline``, that goes from raw per-dimension samples all
the way to the leading eigenvalues/eigenvectors of the (whitened) reduced
generator matrix:

    X (raw samples)
      -> per-dimension basis evaluation (psi, dpsi)
      -> Transformed_Data_Tensor_TT  (TT data tensor, built lazily/cached)
      -> global_svd_tt               (reduced TT basis U, singular values)
      -> TgStiffnessOperator + compute_A_r  (Sigma-aware reduced generator)
      -> mean-free whitening         (removes the trivial constant eigenfunction)
      -> scipy.linalg.eigh + filter_ev

This mirrors the pipeline used across several experiment notebooks (e.g. a
2D/3D "Lemon Slice" potential with a constant diffusion tensor, and
TICA-reduced molecular trajectories with a samplewise diffusion tensor);
``Sigma`` here can be ``None``, a constant ``(p, p)`` matrix, or a samplewise
``(p, p, m)`` array, matching ``TgStiffnessOperator``'s own conventions.

This module intentionally does no plotting and no data loading -- callers
supply ``X`` (and optionally a basis and/or Sigma) and get back a
``GedmdResult`` with everything needed to inspect eigenvalues or build
eigenfunctions themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import eigh

from tensor_gedmd.basis_sets.basis_sets import BasisSet
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures
from tensor_gedmd.reps.transformed_data_tensor import Transformed_Data_Tensor_TT
from tensor_gedmd.reps.stiffness_tt import TgStiffnessOperator
from tensor_gedmd.algorithms.global_svd import global_svd_tt
from tensor_gedmd.algorithms.mat_vec_prod import compute_A_r
from tensor_gedmd.algorithms.util import filter_ev

SigmaLike = Union[None, np.ndarray]
LengthScaleLike = Union[float, Sequence[float]]


# ======================================================================================
# Basis construction / evaluation helpers
# ======================================================================================

def deterministic_rff_frequencies(n: int, length_scale: float) -> np.ndarray:
    """
    Deterministic (non-random) 1D RFF frequency vector.

    omega_k = k / length_scale, for k = 1, ..., n.

    Parameters
    ----------
    n : int
        Number of features.
    length_scale : float
        Larger values give lower (smoother) frequencies.

    Returns
    -------
    np.ndarray
        Frequency matrix of shape (n, 1), suitable for
        ``RandomFourierFeatures(omega=...)``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    if length_scale <= 0:
        raise ValueError(f"length_scale must be positive, got {length_scale}.")

    k = np.arange(1, n + 1, dtype=float)
    return (k / length_scale).reshape(-1, 1)


def build_deterministic_rff_basis(
    p: int, n_features: int, length_scale: LengthScaleLike
) -> List[RandomFourierFeatures]:
    """
    Build one deterministic-frequency RandomFourierFeatures basis per
    physical dimension, all with the same number of features.

    Parameters
    ----------
    p : int
        Number of physical dimensions.
    n_features : int
        Number of RFF features per dimension.
    length_scale : float or sequence of float
        Per-dimension RFF length scale. A single float is broadcast to all
        p dimensions.

    Returns
    -------
    list[RandomFourierFeatures]
        Length-p list of basis objects, one per physical dimension.
    """
    if p < 1:
        raise ValueError(f"p must be at least 1, got {p}.")

    if np.isscalar(length_scale):
        scales: List[float] = [float(length_scale)] * p
    else:
        scales = [float(s) for s in length_scale]
        if len(scales) != p:
            raise ValueError(
                f"length_scale must be a scalar or a sequence of length p={p}; "
                f"got length {len(scales)}."
            )

    return [
        RandomFourierFeatures(omega=deterministic_rff_frequencies(n_features, s))
        for s in scales
    ]


def evaluate_basis(
    X: np.ndarray, basis_list: Sequence[BasisSet]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Evaluate a per-dimension basis (and its derivative) at samples X.

    Parameters
    ----------
    X : np.ndarray, shape (p, m)
        Samples: p physical dimensions, m samples.
    basis_list : sequence of BasisSet
        One 1D basis per physical dimension (length p). Each basis must
        follow the (d, m) -> (n, m) / (n, d, m) convention used throughout
        this package (see ``tensor_gedmd.basis_sets.basis_sets.BasisSet``).

    Returns
    -------
    psi : list[np.ndarray]
        psi[k] has shape (n_k, m), k = 0, ..., p-1.
    dpsi : list[np.ndarray]
        dpsi[k] has shape (n_k, m), k = 0, ..., p-1.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (p, m); got {X.shape}.")

    p = X.shape[0]
    if len(basis_list) != p:
        raise ValueError(
            f"basis_list has {len(basis_list)} entries, expected p={p} "
            "(one per physical dimension of X)."
        )

    psi: List[np.ndarray] = []
    dpsi: List[np.ndarray] = []

    for k, basis in enumerate(basis_list):
        Xk = X[k:k + 1, :]  # (1, m)

        psi_k = np.asarray(basis(Xk))          # (n_k, m)
        dpsi_k = np.asarray(basis.gradient(Xk))  # (n_k, 1, m)

        if dpsi_k.ndim == 3:
            if dpsi_k.shape[1] != 1:
                raise ValueError(
                    f"basis_list[{k}].gradient(...) must return shape (n_k, 1, m) "
                    f"for a 1D basis; got {dpsi_k.shape}."
                )
            dpsi_k = dpsi_k[:, 0, :]

        psi.append(psi_k)
        dpsi.append(dpsi_k)

    return psi, dpsi


# ======================================================================================
# Mean-free whitening
# ======================================================================================

def mean_free_whitening(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the mean-free whitening transform for the retained basis Z.

    The generator always has the constant function as a trivial eigenfunction
    with eigenvalue 0. This computes the eigendecomposition of
    ``I - mean(Z) @ mean(Z).T`` and discards the (near-)zero eigenvalue
    (and its eigenvector), which corresponds to that trivial direction, so
    it doesn't need to be handled specially later.

    Parameters
    ----------
    Z : np.ndarray, shape (r, m)
        The retained basis (e.g. ``sqrt(m) * V_core[:, :, 0]`` from
        ``global_svd_tt``) evaluated at the m samples.

    Returns
    -------
    d_G : np.ndarray, shape (r - 1,)
        Eigenvalues of the mean-free Gram matrix (ascending).
    W_G : np.ndarray, shape (r, r - 1)
        Corresponding eigenvectors.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"Z must have shape (r, m); got {Z.shape}.")

    r = Z.shape[0]
    mean_z = np.mean(Z, axis=1)
    G1 = np.eye(r) - np.outer(mean_z, mean_z)

    d_G, W_G = eigh(G1)
    # Discard the (near-)zero eigenvalue: the trivial constant direction.
    d_G = d_G[1:]
    W_G = W_G[:, 1:]
    return d_G, W_G


# ======================================================================================
# Result container
# ======================================================================================

@dataclass
class GedmdResult:
    """
    Everything produced by ``run_gedmd_pipeline``.

    Attributes
    ----------
    eigenvalues : np.ndarray, shape (nev,)
        Leading eigenvalues, ordered by descending real part (largest/least
        negative first).
    eigenvectors : np.ndarray, shape (r - 1, nev)
        Corresponding eigenvectors, in mean-free-whitened reduced
        coordinates (matching ``Z_white``).
    U_cores : list[np.ndarray]
        The reduced TT basis from ``global_svd_tt``.
    Sigma_svd : np.ndarray, shape (r, r)
        Diagonal matrix of retained singular values from ``global_svd_tt``.
    Z : np.ndarray, shape (r, m)
        The retained basis evaluated at the input samples,
        ``sqrt(m) * V_core[:, :, 0]``.
    Z_white : np.ndarray, shape (r - 1, m)
        ``Z`` in mean-free-whitened coordinates.
    d_G, W_G : np.ndarray
        Mean-free whitening eigenvalues/eigenvectors (see
        ``mean_free_whitening``).
    A_r : np.ndarray, shape (r, r)
        The reduced (Galerkin-projected) generator matrix, before dividing
        out the singular values.
    reduced_matrix : np.ndarray, shape (r, r)
        ``inv(Sigma_svd) @ A_r @ inv(Sigma_svd)``, before mean-free
        whitening.
    op : TgStiffnessOperator
        The stiffness operator used to compute ``A_r``.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    U_cores: List[np.ndarray]
    Sigma_svd: np.ndarray
    Z: np.ndarray
    Z_white: np.ndarray
    d_G: np.ndarray
    W_G: np.ndarray
    A_r: np.ndarray
    reduced_matrix: np.ndarray
    op: TgStiffnessOperator

    def eigenfunctions(self) -> np.ndarray:
        """
        Evaluate the retained eigenfunctions at the input samples.

        Returns
        -------
        np.ndarray, shape (nev, m)
            Row j is the j-th eigenfunction evaluated at every sample.
        """
        return self.eigenvectors.T @ self.Z_white


# ======================================================================================
# Main pipeline
# ======================================================================================

def run_gedmd_pipeline(
    X: np.ndarray,
    basis_list: Optional[Sequence[BasisSet]] = None,
    *,
    n_features: int = 10,
    length_scale: LengthScaleLike = 1.0,
    Sigma: SigmaLike = None,
    rmax: Optional[int] = None,
    tol: float = 1e-5,
    nev: int = 10,
    eps1: float = -np.inf,
    eps2: float = np.inf,
    chunk_size: int = 100,
    n_workers: Optional[int] = None,
) -> GedmdResult:
    """
    Run the full TT-gEDMD pipeline from raw samples to leading eigenvalues.

    Parameters
    ----------
    X : np.ndarray, shape (p, m)
        Input samples: p physical dimensions, m samples (e.g. raw
        coordinates of a test potential, or TICA-reduced molecular
        coordinates).
    basis_list : sequence of BasisSet, optional
        One 1D basis per physical dimension (length p). If omitted, a
        deterministic-frequency ``RandomFourierFeatures`` basis is built
        automatically from ``n_features`` and ``length_scale`` (see
        ``build_deterministic_rff_basis``).
    n_features : int, default=10
        Only used when ``basis_list`` is None.
    length_scale : float or sequence of float, default=1.0
        Only used when ``basis_list`` is None.
    Sigma : None, (p, p) array, or (p, p, m) array, optional
        Diffusion tensor, forwarded to ``TgStiffnessOperator``. ``None`` or
        a constant ``(p, p)`` matrix are both "constant" mode; a
        ``(p, p, m)`` array is samplewise/variable. Both are used correctly
        by ``compute_A_r`` (which this pipeline uses), including a
        non-identity constant Sigma.
    rmax, tol : optional
        Truncation parameters forwarded to ``global_svd_tt``.
    nev : int, default=10
        Number of leading eigenvalues/eigenvectors to return -- as many as
        the caller wants to inspect/print. Capped to the number available
        after filtering.
    eps1, eps2 : float, optional
        Eigenvalue real-part filter bounds, forwarded to ``filter_ev``.
    chunk_size, n_workers : optional
        Forwarded to ``compute_A_r``.

    Returns
    -------
    GedmdResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (p, m); got {X.shape}.")
    p, m = X.shape

    if basis_list is None:
        basis_list = build_deterministic_rff_basis(p, n_features, length_scale)
    elif len(basis_list) != p:
        raise ValueError(f"basis_list has {len(basis_list)} entries, expected p={p}.")

    psi, dpsi = evaluate_basis(X, basis_list)

    # ------------------------------------------------------------------
    # TT data tensor + global SVD.
    #
    # Transformed_Data_Tensor_TT.build_core is 1-indexed (1..p+1); wrap it
    # as a 0-indexed callable to match global_svd_tt's core-getter convention.
    # ------------------------------------------------------------------
    data_tensor = Transformed_Data_Tensor_TT(psi=psi)

    def core_getter(k: int) -> np.ndarray:
        return data_tensor.build_core(k + 1)

    U_tt, Sigma_svd, V_core = global_svd_tt(
        core_getter, num_cores=p + 1, rmax=rmax, tol=tol
    )
    U_cores = U_tt.cores
    r = U_cores[-1].shape[2]

    Z = np.sqrt(m) * V_core[:, :, 0]  # (r, m)

    # ------------------------------------------------------------------
    # Mean-free whitening transform (removes the trivial constant
    # eigenfunction ahead of the final eigendecomposition).
    # ------------------------------------------------------------------
    d_G, W_G = mean_free_whitening(Z)

    # ------------------------------------------------------------------
    # Sigma-aware reduced generator matrix.
    # ------------------------------------------------------------------
    op = TgStiffnessOperator(psi=psi, dpsi=dpsi, Sigma=Sigma)
    A_r = compute_A_r(op, U_cores, r, chunk_size=chunk_size, n_workers=n_workers)

    Sigma_svd_inv = np.linalg.inv(Sigma_svd)
    reduced_matrix = Sigma_svd_inv @ A_r @ Sigma_svd_inv

    # ------------------------------------------------------------------
    # Whiten and eigendecompose.
    # ------------------------------------------------------------------
    Dg_inv_sqrt = np.diag(d_G ** (-0.5))
    R_white = Dg_inv_sqrt @ W_G.T @ reduced_matrix @ W_G @ Dg_inv_sqrt
    R_white = 0.5 * (R_white + R_white.T)

    d_tt, W_tt = eigh(R_white)
    d_tt, W_tt = filter_ev(d_tt, W_tt, eps1=eps1, eps2=eps2)

    nev = min(nev, d_tt.shape[0])
    d_tt = d_tt[-nev:][::-1]
    W_tt = W_tt[:, -nev:][:, ::-1]

    Z_white = Dg_inv_sqrt @ (W_G.T @ Z)

    return GedmdResult(
        eigenvalues=d_tt,
        eigenvectors=W_tt,
        U_cores=U_cores,
        Sigma_svd=Sigma_svd,
        Z=Z,
        Z_white=Z_white,
        d_G=d_G,
        W_G=W_G,
        A_r=A_r,
        reduced_matrix=reduced_matrix,
        op=op,
    )
