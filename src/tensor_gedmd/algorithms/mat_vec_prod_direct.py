"""
mat_vec_prod_direct: reduced generator matrix A_r, computed directly.

This is a sibling of ``tensor_gedmd.algorithms.mat_vec_prod`` -- same family
(matrix/vector product style computations against the stiffness operator),
different method:

- ``mat_vec_prod``: builds/applies the TT stiffness operator to a TT vector
  one core at a time, using CUR-based truncation (``truncate_tt_core_cur``)
  at every step to keep bond dimensions manageable. General-purpose (works
  for any TT vector, any number of applications), but pays for that
  generality every time it's used.
- ``mat_vec_prod_direct`` (this module): skips building/applying the
  operator entirely. It streams through the samples once, directly
  assembling the small (r, r) reduced generator matrix ``A_r = U^T A U``
  via the same per-sample sweep used by ``mat_vec_prod.compute_A_r``
  (``_process_chunk``, imported unchanged -- single source of truth for
  that part). No TT operator is ever formed and no CUR truncation happens
  anywhere in this module; see the module-level note below for why.

What's actually new here versus ``mat_vec_prod.compute_A_r``: the final
Sigma-weighting step, ``A_r = -1/(2m) * W^T (Sigma W)``. Instead of always
doing a general ``(p, p)`` contraction (via ``np.tensordot``/``np.einsum``),
it first classifies Sigma's structure and picks the cheapest equivalent
computation:

    Sigma is None (identity)          -> SW = W                      (free)
    Sigma constant, scalar * I        -> SW = scalar * W              (one multiply)
    Sigma constant, diagonal          -> SW = W * diag[None, :, None] (elementwise)
    Sigma constant, general dense     -> SW = tensordot(Sigma, W)     (same as compute_A_r)
    Sigma samplewise, diagonal at every sample
                                       -> SW = W * diag[:, :, None]   (elementwise)
    Sigma samplewise, general dense   -> SW = einsum(...)            (same as compute_A_r)

All six paths are verified (see tests/test_mat_vec_prod_direct.py) to produce
identical results to ``mat_vec_prod.compute_A_r`` -- this module is purely
a performance optimization on top of the exact same math, not a different
algorithm. ``TgStiffnessOperator`` (from ``tensor_gedmd.reps.stiffness_tt``)
needs no changes to be used here: it already exposes everything this
function reads (``p``, ``m``, ``psi``, ``dpsi``, ``local_dims``,
``sigma_mode``, ``Sigma_prepared``).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Tuple

import numpy as np

from tensor_gedmd.algorithms.mat_vec_prod import _fmt, _process_chunk, _truncate_core
from tensor_gedmd.operations.tt_operations import TTLike, _as_tt_cores

__all__ = ["compute_A_r", "classify_sigma"]


# ======================================================================================
# Sigma structure classification
# ======================================================================================

def classify_sigma(op: Any, atol: float = 1e-12) -> Tuple[str, Any]:
    """
    Classify ``op``'s diffusion tensor into the cheapest applicable case.

    Returns
    -------
    kind : str
        One of "identity", "scalar", "diagonal", "dense_constant",
        "diag_samplewise", "dense_samplewise".
    payload : Any
        Whatever the corresponding fast path in ``compute_A_r`` needs:
        None for "identity"; a float for "scalar"; a ``(p,)`` array for
        "diagonal"; a ``(p, p)`` array for "dense_constant"; a ``(p, m)``
        array for "diag_samplewise"; a ``(p, p, m)`` array for
        "dense_samplewise".
    """
    if op.sigma_mode == "constant":
        Sigma = op.Sigma_prepared
        if Sigma is None:
            return "identity", None

        Sigma = np.asarray(Sigma, dtype=float)
        p = Sigma.shape[0]
        diag = np.diag(Sigma).copy()
        off_diag = Sigma - np.diag(diag)

        if np.allclose(off_diag, 0.0, atol=atol):
            if np.allclose(diag, diag[0], atol=atol):
                return "scalar", float(diag[0])
            return "diagonal", diag
        return "dense_constant", Sigma

    # sigma_mode == "variable": Sigma is (p, p, m)
    Sigma = np.asarray(op.Sigma_prepared, dtype=float)
    p = Sigma.shape[0]
    off_diag_mask = ~np.eye(p, dtype=bool)

    if np.allclose(Sigma[off_diag_mask], 0.0, atol=atol):
        diag_vals = np.stack([Sigma[i, i, :] for i in range(p)], axis=0)  # (p, m)
        return "diag_samplewise", diag_vals

    return "dense_samplewise", Sigma


def _apply_sigma(kind: str, payload: Any, W_full: np.ndarray) -> np.ndarray:
    """Apply the (implicit) Sigma to W_full = (m, p, r) using the cheapest path."""
    if kind == "identity":
        return W_full
    if kind == "scalar":
        return payload * W_full
    if kind == "diagonal":
        # payload: (p,) -> broadcast over (m, p, r)
        return W_full * payload[None, :, None]
    if kind == "dense_constant":
        SW = np.tensordot(payload, W_full, axes=([1], [1]))  # (p, m, r)
        return np.moveaxis(SW, 0, 1)                          # (m, p, r)
    if kind == "diag_samplewise":
        # payload: (p, m) -> (m, p) -> broadcast over (m, p, r)
        return W_full * payload.T[:, :, None]
    if kind == "dense_samplewise":
        return np.einsum('abl,lbj->laj', payload, W_full, optimize=True)
    raise ValueError(f"Unknown Sigma classification: {kind!r}")


# ======================================================================================
# Main entry point
# ======================================================================================

def compute_A_r(
    op: Any,
    U_cores: TTLike,
    r: int,
    chunk_size: int = 100,
    n_workers: Optional[int] = None,
    r_cap: Optional[int] = None,
) -> np.ndarray:
    """
    Faster drop-in replacement for
    ``tensor_gedmd.algorithms.mat_vec_prod.compute_A_r``.

    Computes the same reduced (Galerkin-projected) generator matrix

        A_r[i, j] = -1/(2m) * sum_l sum_{c,c'} Sigma_{c,c'}(l) *
                    d/dx_c u_i(x_l) * d/dx_{c'} u_j(x_l)

    from a stiffness operator ``op`` (``tensor_gedmd.reps.stiffness_tt.
    TgStiffnessOperator``, unchanged -- no new attributes required) and a
    reduced TT basis ``U_cores`` (typically from ``global_svd_tt``). The
    per-sample sweep is identical to ``mat_vec_prod.compute_A_r``; only the
    final Sigma-weighting step is specialized based on Sigma's structure
    (see ``classify_sigma``), which is free for Sigma=None or a scalar
    multiple of the identity, cheap for a diagonal Sigma (constant or
    samplewise), and falls back to the same general contraction otherwise.

    Parameters
    ----------
    op : TgStiffnessOperator
        Provides ``p``, ``m``, ``psi``, ``dpsi``, ``local_dims``,
        ``sigma_mode``, and ``Sigma_prepared``.
    U_cores : TT or list of np.ndarray
        The reduced basis, with exactly ``op.p`` cores.
    r : int
        Retained rank of ``U_cores`` (the last core's right bond dimension).
    chunk_size : int, default=100
        Number of samples processed per chunk/task.
    n_workers : int, optional
        Number of worker threads. Defaults to ``min(4, cpu_count() // 2)``.
    r_cap : int, optional
        If given and smaller than ``r``, hard-caps every core's bond
        dimensions to at most ``r_cap`` before computing (cheap/approximate).

    Returns
    -------
    A_r : np.ndarray
        Symmetrized reduced generator matrix of shape (r, r).
    """
    t0 = time.perf_counter()
    p, m = op.p, op.m

    if p < 2:
        raise ValueError("compute_A_r requires at least 2 physical dimensions.")

    U_cores = _as_tt_cores(U_cores)
    if len(U_cores) != p:
        raise ValueError(f"Expected {p} U_cores, got {len(U_cores)}")

    if r_cap is not None and r_cap < r:
        U_cores = [_truncate_core(c, r_cap) for c in U_cores]
        r = U_cores[-1].shape[2]

    if n_workers is None:
        n_workers = min(4, max(1, (os.cpu_count() or 2) // 2))

    bonds = [c.shape[0] for c in U_cores] + [r]

    kind, payload = classify_sigma(op)
    print(
        f"  [mat_vec_prod_direct] p={p}  m={m}  r={r}  chunk={chunk_size}  "
        f"workers={n_workers}  sigma_kind={kind}"
    )

    W_full = np.zeros((m, p, r), dtype=np.float64)
    chunks = [(s, min(s + chunk_size, m)) for s in range(0, m, chunk_size)]
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_chunk, s, e, op, U_cores, p, bonds, r): (s, e)
            for s, e in chunks
        }
        for fut in as_completed(futures):
            l_start, l_end, W_chunk = fut.result()
            W_full[l_start:l_end] = W_chunk
            completed += 1
            if completed % 5 == 0 or completed == len(chunks):
                print(
                    f"    chunks {completed}/{len(chunks)} ({100 * l_end / m:.0f}%)  "
                    f"elapsed {_fmt(time.perf_counter() - t0)}",
                    flush=True,
                )

    SW = _apply_sigma(kind, payload, W_full)

    # No Sigma at all ("identity" case) uses TgStiffnessOperator's own
    # convention of scale -1/m; a real Sigma matrix (any other kind) uses
    # -1/(2m). These are NOT interchangeable -- "identity" means "no Sigma
    # was given", not "Sigma is literally the identity matrix".
    prefactor = -(1.0 / m) if kind == "identity" else -(1.0 / (2.0 * m))

    W2 = W_full.reshape(-1, r)
    SW2 = SW.reshape(-1, r)
    A_r = W2.T @ SW2
    A_r *= prefactor
    A_r = 0.5 * (A_r + A_r.T)

    print(f"  [mat_vec_prod_direct] TOTAL {_fmt(time.perf_counter() - t0)}")
    return A_r
