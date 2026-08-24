"""
Chignolin -- plot_figure_chignolin_pcca_matrix_vs_tensor_multi_dim.py

Loads results/chignolin_pcca_multi_dim_results.npz (produced by
../run_calculation_pcca_multi_dim.py) and reproduces the 2x2 soft-membership
(RGB blend) PCCA figure:
    upper-left  : Tensor PCCA (TICA dim=3)
    upper-right : Matrix PCCA (TICA dim=3)
    lower-left  : Tensor PCCA (TICA dim=6)
    lower-right : Tensor PCCA (TICA dim=10)

No calculation happens here -- if results/ doesn't exist yet, run
../run_calculation_pcca_multi_dim.py first.

Usage
-----
    python plot_figure_chignolin_pcca_matrix_vs_tensor_multi_dim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "chignolin_pcca_multi_dim_results.npz"
CALC_SCRIPT_HINT = "examples/publication/chignolin/run_calculation_pcca_multi_dim.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "figure_chignolin_pcca_matrix_vs_tensor_multi_dim.pdf"


# ============================================================
# PCCA helpers (verbatim math from the source notebook)
# ============================================================

def _pcca_connected_isa(eigenvectors, n_clusters):
    n, m = eigenvectors.shape
    if n_clusters > m:
        raise ValueError(f"Cannot cluster ({n}x{m}) to {n_clusters} clusters.")
    diffs = np.abs(np.max(eigenvectors, axis=0) - np.min(eigenvectors, axis=0))
    assert diffs[0] < 1e-6, "First eigenvector is not constant."
    assert diffs[1] > 1e-6, "Second eigenvector is constant."
    c = eigenvectors[:, :n_clusters].copy()
    ind = np.zeros(n_clusters, dtype=np.int32)
    ind[0] = int(np.argmax(np.linalg.norm(c, axis=1)))
    ortho = c - c[ind[0], None, :]
    for k in range(1, n_clusters):
        temp = ortho[ind[k - 1]].copy()
        tnorm = np.linalg.norm(temp)
        if tnorm > 0:
            temp /= tnorm
        proj = ortho - (ortho @ temp)[:, None] * temp
        dists = np.linalg.norm(proj, axis=1)
        dists[ind[:k]] = -np.inf
        ind[k] = int(np.argmax(dists))
        ortho = proj
    return c @ np.linalg.inv(c[ind, :]), np.linalg.inv(c[ind, :])


def pcca_from_slow_modes(slow_modes, K):
    m = slow_modes.shape[0]
    evec = np.hstack([np.ones((m, 1)), np.real(slow_modes)])
    evec[:, 0] = 1.0
    for k in range(1, K):
        s = np.std(evec[:, k])
        if s > 0:
            evec[:, k] /= s
    chi, rot = _pcca_connected_isa(evec, K)
    chi = np.clip(chi, 0.0, None)
    rowsum = chi.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    chi /= rowsum
    return chi, np.argmax(chi, axis=1), rot


def build_tensor_slow_modes(reduced_matrix, V_core, m, K, nev=40):
    Z = np.sqrt(m) * V_core[:, :, 0]
    r = Z.shape[0]
    mean_b = np.mean(Z, axis=1)
    G1 = np.eye(r) - np.outer(mean_b, mean_b)
    d_G, W_G = eigh(G1)
    d_G, W_G = d_G[1:], W_G[:, 1:]
    Dmhalf = np.diag(d_G ** (-0.5))
    Z_white = Dmhalf @ (W_G.T @ Z)
    if reduced_matrix.shape[0] == Z_white.shape[0]:
        R_use, Z_eval = reduced_matrix, Z_white
    elif reduced_matrix.shape[0] == Z.shape[0]:
        R_use = Dmhalf @ (W_G.T @ reduced_matrix @ W_G) @ Dmhalf
        Z_eval = Z_white
    else:
        raise ValueError(f"Shape mismatch: {reduced_matrix.shape} vs {Z.shape}")
    d_tt, W_tt = eigh(R_use)
    nev_eff = min(nev, len(d_tt))
    d_tt = d_tt[-nev_eff:][::-1]
    W_tt = W_tt[:, -nev_eff:][:, ::-1]
    slow = np.column_stack([np.real(W_tt[:, j] @ Z_eval) for j in range(K - 1)])
    return slow, d_tt, Z_eval


def align_labels(labels_src, labels_ref, K):
    C = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            C[i, j] = np.sum((labels_src == i) & (labels_ref == j))
    _, col = linear_sum_assignment(-C)
    return col[labels_src], col


def plot_soft_blend(ax, X, chi, title, state_colors=None, state_labels=None, s=10, alpha=0.85):
    K = chi.shape[1]
    if state_colors is None:
        state_colors = ['tab:blue', 'tab:orange', 'tab:green',
                        'tab:red', 'tab:purple', 'tab:brown'][:K]
    if state_labels is None:
        state_labels = [f'State {k}' for k in range(K)]

    rgb_states = np.array([mcolors.to_rgb(c) for c in state_colors])
    rgb_points = np.clip(chi @ rgb_states, 0, 1)

    ax.scatter(X[0], X[1], color=rgb_points, s=s, alpha=alpha)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"TICA $x_1$")
    ax.set_ylabel(r"TICA $x_2$")

    patches = [Patch(color=state_colors[k], label=state_labels[k]) for k in range(K)]
    ax.legend(handles=patches, fontsize=8, loc='upper right')


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)
    K = int(results["K"])

    Xlist_3 = results["Xlist_3"]
    reduced_matrix_3 = results["reduced_matrix_3"]
    V_core_3 = results["V_core_3"]
    m_3 = int(results["m_3"])
    Wdata = results["Wdata"]

    Xlist_6 = results["Xlist_6"]
    reduced_matrix_6 = results["reduced_matrix_6"]
    V_core_6 = results["V_core_6"]
    m_6 = int(results["m_6"])

    Xlist_10 = results["Xlist_10"]
    reduced_matrix_10 = results["reduced_matrix_10"]
    V_core_10 = results["V_core_10"]
    m_10 = int(results["m_10"])

    slow3, d_tt3, _ = build_tensor_slow_modes(reduced_matrix_3, V_core_3, m_3, K)
    chi3_t, lab3_t, _ = pcca_from_slow_modes(slow3, K)

    slow_m3 = np.real(Wdata[:K - 1, :].T)
    chi3_m, lab3_m, _ = pcca_from_slow_modes(slow_m3, K)

    lab3_t_al, perm3 = align_labels(lab3_t, lab3_m, K)
    chi3_t_al = chi3_t[:, perm3]

    slow6, d_tt6, _ = build_tensor_slow_modes(reduced_matrix_6, V_core_6, m_6, K)
    chi6_t, lab6_t, _ = pcca_from_slow_modes(slow6, K)

    slow10, d_tt10, _ = build_tensor_slow_modes(reduced_matrix_10, V_core_10, m_10, K)
    chi10_t, lab10_t, _ = pcca_from_slow_modes(slow10, K)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    state_labels = ['Misfolded', 'Unfolded', 'Folded']
    plot_soft_blend(axes[0, 0], Xlist_3, chi3_t_al, 'Tensor PCCA (TICA_dim=3)', state_labels=state_labels)
    plot_soft_blend(axes[0, 1], Xlist_3, chi3_m, 'Matrix PCCA (TICA_dim=3)', state_labels=state_labels)
    plot_soft_blend(axes[1, 0], Xlist_6, chi6_t, 'Tensor PCCA (TICA_dim=6)', state_labels=state_labels)
    plot_soft_blend(axes[1, 1], Xlist_10, chi10_t, 'Tensor PCCA (TICA_dim=10)', state_labels=state_labels)

    plt.tight_layout()
    save_figure(fig, OUTPUT_PATH, also_png=True)
    plt.show()

    print("\nSoft membership statistics (mean +/- std per state):")
    for tag, chi_arr in [
        ("Tensor dim=3  (aligned)", chi3_t_al),
        ("Matrix dim=3           ", chi3_m),
        ("Tensor dim=6           ", chi6_t),
        ("Tensor dim=10          ", chi10_t),
    ]:
        means = chi_arr.mean(axis=0)
        stds = chi_arr.std(axis=0)
        print(f"  {tag}: mean={np.round(means, 3)}  std={np.round(stds, 3)}")


if __name__ == "__main__":
    main()
