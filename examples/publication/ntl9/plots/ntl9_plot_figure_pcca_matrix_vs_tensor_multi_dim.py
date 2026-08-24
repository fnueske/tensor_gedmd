"""
NTL9 -- plot_figure_ntl9_pcca_matrix_vs_tensor_multi_dim.py

Loads results/ntl9_pcca_multi_dim_results.npz (produced by
../run_calculation_pcca_multi_dim.py) and reproduces the 2x2 3D soft-PCCA
comparison figure:
    Matrix PCCA, TICA_dimension=3   |  Tensor PCCA, TICA_dimension=3
    Tensor PCCA, TICA_dimension=6   |  Tensor PCCA, TICA_dimension=8

Colors are consistent across panels: Tensor labels at each dimension are
aligned to the Matrix TICA=3 reference via a Hungarian-algorithm match, and
the rare high-TICA-x3 state is auto-detected and given its own color,
separate from the green/orange/red trio used for the other three states.

No calculation happens here -- if results/ doesn't exist yet, run
../run_calculation_pcca_multi_dim.py first.

Usage
-----
    python plot_figure_ntl9_pcca_matrix_vs_tensor_multi_dim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from results_io import load_results  # noqa: E402
from plotting import set_publication_style, save_figure  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "ntl9_pcca_multi_dim_results.npz"
CALC_SCRIPT_HINT = "examples/publication/ntl9/run_calculation_pcca_multi_dim.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "figure_ntl9_pcca_matrix_vs_tensor_multi_dim.pdf"

SOFT_PCCA_CUT = 0.6
FIRST_N_EIGENVALUES = 4
RARE_STATE_COLOR_TOP = "tab:purple"
RARE_STATE_COLOR_BOTTOM = "tab:blue"
RARE_STATE_NAME_TOP = "Rare"
RARE_STATE_NAME_BOTTOM = "Intermediate"
MAIN_STATE_COLORS = ["tab:green", "tab:orange", "tab:red"]
MAIN_STATE_NAMES = ["Folded", "Unfolded", "Misfolded"]


# ============================================================
# PCCA helpers (verbatim math from the source notebook)
# ============================================================

def _pcca_connected_isa(eigenvectors, n_clusters):
    n, m = eigenvectors.shape
    if n_clusters > m:
        raise ValueError(f"Cannot cluster eigenvector matrix of shape {eigenvectors.shape} into {n_clusters} clusters.")
    c = eigenvectors[:, :n_clusters].copy()
    ind = np.zeros(n_clusters, dtype=np.int32)
    ind[0] = int(np.argmax(np.linalg.norm(c, axis=1)))
    ortho = c - c[ind[0], None, :]
    for k in range(1, n_clusters):
        temp = ortho[ind[k - 1]].copy()
        tnorm = np.linalg.norm(temp)
        if tnorm > 0:
            temp /= tnorm
        dots = ortho @ temp
        proj = ortho - dots[:, None] * temp
        dists = np.linalg.norm(proj, axis=1)
        dists[ind[:k]] = -np.inf
        ind[k] = int(np.argmax(dists))
        ortho = proj
    rot_mat = np.linalg.pinv(c[ind, :])
    chi = c @ rot_mat
    return chi, rot_mat, ind


def pcca_from_slow_modes(slow_modes, K, normalize_modes=True):
    m = slow_modes.shape[0]
    evec = np.hstack([np.ones((m, 1)), np.real(slow_modes)])
    if normalize_modes:
        for j in range(1, K):
            s = np.std(evec[:, j])
            if s > 0:
                evec[:, j] /= s
    chi, rot, ind = _pcca_connected_isa(evec, K)
    chi = np.clip(np.real(chi), 0.0, None)
    rowsum = chi.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    chi /= rowsum
    labels = np.argmax(chi, axis=1)
    return chi, labels, rot


def tensor_slow_modes_from_reduced_matrix(reduced_matrix, V_core, K, nev=40):
    Z = V_core[:, :, 0]
    m = Z.shape[1]
    Z = np.sqrt(m) * Z
    r = Z.shape[0]

    mean_basis = np.mean(Z, axis=1)
    G1 = np.eye(r) - np.outer(mean_basis, mean_basis)
    d_G, W_G = eigh(G1)
    d_G, W_G = d_G[1:], W_G[:, 1:]
    Dmhalf = np.diag(d_G ** (-0.5))
    Z_white = Dmhalf @ (W_G.T @ Z)

    if reduced_matrix.shape[0] == Z_white.shape[0]:
        R_use = reduced_matrix
    elif reduced_matrix.shape[0] == Z.shape[0]:
        R_use = Dmhalf @ (W_G.T @ reduced_matrix @ W_G) @ Dmhalf
    else:
        raise ValueError(f"Shape mismatch: reduced_matrix={reduced_matrix.shape}, Z={Z.shape}")

    d_tt, W_tt = eigh(R_use)
    nev_eff = min(nev, len(d_tt))
    d_tt = d_tt[-nev_eff:][::-1]
    W_tt = W_tt[:, -nev_eff:][:, ::-1]

    zero_tol = 1e-6
    keep = np.abs(d_tt) > zero_tol
    d_tt, W_tt = d_tt[keep], W_tt[:, keep]

    if W_tt.shape[1] < K - 1:
        raise ValueError(f"Only {W_tt.shape[1]} nontrivial tensor slow modes available, but K={K} needs {K - 1}.")

    slow_modes = [np.real(W_tt[:, j].T @ Z_white) for j in range(K - 1)]
    return np.column_stack(slow_modes), d_tt, W_tt, Z_white


def align_pcca_chi_by_labels(chi_src, labels_src, labels_ref, K):
    overlap = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            overlap[i, j] = np.sum((labels_src == i) & (labels_ref == j))
    row_ind, col_ind = linear_sum_assignment(-overlap)
    old_to_ref = np.zeros(K, dtype=int)
    old_to_ref[row_ind] = col_ind
    inv_perm = np.argsort(old_to_ref)
    chi_aligned = chi_src[:, inv_perm]
    labels_aligned = np.argmax(chi_aligned, axis=1)
    return chi_aligned, labels_aligned, old_to_ref, overlap


def plot_soft_pcca_3d(ax, X, chi, title, K, color_map, name_map, cut=0.6):
    if X.shape[0] < 3:
        raise ValueError(f"{title}: X must have at least 3 rows for 3D plotting.")
    X3 = X[:3, :]

    max_chi = np.max(chi, axis=1)
    residual_mask = max_chi < cut

    ax.scatter(X3[0, residual_mask], X3[1, residual_mask], X3[2, residual_mask],
               c="white", edgecolors="0.6", linewidths=0.3, s=5, alpha=0.6,
               label="Residual", depthshade=True)

    state_counts = []
    for k in range(K):
        mask = chi[:, k] >= cut
        state_counts.append(int(np.sum(mask)))
        ax.scatter(X3[0, mask], X3[1, mask], X3[2, mask],
                   color=color_map[k], s=7, alpha=0.9, label=name_map[k], depthshade=True)

    ax.set_title(title)
    ax.set_xlabel(r"TICA $x_1$")
    ax.set_ylabel(r"TICA $x_2$")
    ax.set_zlabel(r"TICA $x_3$")

    return state_counts, int(np.sum(residual_mask))


def main() -> None:
    set_publication_style()
    results = load_results(RESULTS_PATH, calc_script_hint=CALC_SCRIPT_HINT)
    K = int(results["K"])

    Xlist_3 = results["Xlist_3"]
    reduced_matrix_3 = results["reduced_matrix_3"]
    V_core_3 = results["V_core_3"]
    Wdata = results["Wdata"]
    d_matrix = results["d_matrix"]

    Xlist_6 = results["Xlist_6"]
    reduced_matrix_6 = results["reduced_matrix_6"]
    V_core_6 = results["V_core_6"]

    Xlist_8 = results["Xlist_8"]
    reduced_matrix_8 = results["reduced_matrix_8"]
    V_core_8 = results["V_core_8"]

    # ---- Matrix PCCA, TICA=3 (color/label reference) ----
    slow_matrix_3 = np.real(Wdata[:K - 1, :].T)
    chi_matrix_3, labels_matrix_3, _ = pcca_from_slow_modes(slow_matrix_3, K, normalize_modes=True)

    # ---- Tensor PCCA, TICA=3/6/8 ----
    slow_tensor_3, eig_tensor_3, W_tt_3, Z_white_3 = tensor_slow_modes_from_reduced_matrix(
        reduced_matrix_3, V_core_3, K)
    chi_tensor_3, labels_tensor_3, _ = pcca_from_slow_modes(slow_tensor_3, K, normalize_modes=True)

    slow_tensor_6, eig_tensor_6, W_tt_6, Z_white_6 = tensor_slow_modes_from_reduced_matrix(
        reduced_matrix_6, V_core_6, K)
    chi_tensor_6, labels_tensor_6, _ = pcca_from_slow_modes(slow_tensor_6, K, normalize_modes=True)

    slow_tensor_8, eig_tensor_8, W_tt_8, Z_white_8 = tensor_slow_modes_from_reduced_matrix(
        reduced_matrix_8, V_core_8, K)
    chi_tensor_8, labels_tensor_8, _ = pcca_from_slow_modes(slow_tensor_8, K, normalize_modes=True)

    # ---- align Tensor 3/6/8 labels to Matrix TICA=3 ----
    chi_tensor_3_aligned, labels_tensor_3_aligned, perm_3, _ = align_pcca_chi_by_labels(
        chi_tensor_3, labels_tensor_3, labels_matrix_3, K)
    chi_tensor_6_aligned, labels_tensor_6_aligned, perm_6, _ = align_pcca_chi_by_labels(
        chi_tensor_6, labels_tensor_6, labels_matrix_3, K)
    chi_tensor_8_aligned, labels_tensor_8_aligned, perm_8, _ = align_pcca_chi_by_labels(
        chi_tensor_8, labels_tensor_8, labels_matrix_3, K)

    print("Tensor TICA=3 old state -> Matrix TICA=3 state:", perm_3)
    print("Tensor TICA=6 old state -> Matrix TICA=3 state:", perm_6)
    print("Tensor TICA=8 old state -> Matrix TICA=3 state:", perm_8)

    # ---- consistent color map: detect the rare high-TICA-x3 state ----
    mean_x3 = np.full(K, -np.inf)
    pop_size = np.zeros(K, dtype=int)
    for k in range(K):
        mask = chi_matrix_3[:, k] >= SOFT_PCCA_CUT
        pop_size[k] = int(np.sum(mask))
        if pop_size[k] > 0:
            mean_x3[k] = np.mean(Xlist_3[2, mask])
    rare_state = int(np.argmax(mean_x3))

    remaining_states = [k for k in range(K) if k != rare_state]
    remaining_states.sort(key=lambda k: pop_size[k], reverse=True)

    color_map_base = [None] * K
    name_map_base = [None] * K
    for color, name, k in zip(MAIN_STATE_COLORS, MAIN_STATE_NAMES, remaining_states):
        color_map_base[k] = color
        name_map_base[k] = name

    color_map_top = list(color_map_base)
    color_map_top[rare_state] = RARE_STATE_COLOR_TOP
    name_map_top = list(name_map_base)
    name_map_top[rare_state] = RARE_STATE_NAME_TOP

    color_map_bottom = list(color_map_base)
    color_map_bottom[rare_state] = RARE_STATE_COLOR_BOTTOM
    name_map_bottom = list(name_map_base)
    name_map_bottom[rare_state] = RARE_STATE_NAME_BOTTOM

    print(f"\nRare/Intermediate state: State index {rare_state}")
    print(f"  Row 1 (TICA_dimension=3): color={RARE_STATE_COLOR_TOP}, name='{RARE_STATE_NAME_TOP}'")
    print(f"  Row 2 (TICA_dimension=6/8): color={RARE_STATE_COLOR_BOTTOM}, name='{RARE_STATE_NAME_BOTTOM}'")
    for color, name, k in zip(MAIN_STATE_COLORS, MAIN_STATE_NAMES, remaining_states):
        print(f"State index {k}: population {pop_size[k]} -> {name} ({color})")

    print("\nFirst matrix eigenvalues, TICA=3:", np.real(d_matrix[:FIRST_N_EIGENVALUES]))
    print("First tensor eigenvalues, TICA=3:", np.real(eig_tensor_3[:FIRST_N_EIGENVALUES]))
    print("First tensor eigenvalues, TICA=6:", np.real(eig_tensor_6[:FIRST_N_EIGENVALUES]))
    print("First tensor eigenvalues, TICA=8:", np.real(eig_tensor_8[:FIRST_N_EIGENVALUES]))

    # ---- 2x2 figure ----
    fig = plt.figure(figsize=(14, 11))
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    cases = [
        ("Matrix PCCA, TICA_dimension=3", Xlist_3, chi_matrix_3, color_map_top, name_map_top),
        ("Tensor PCCA, TICA_dimension=3", Xlist_3, chi_tensor_3_aligned, color_map_top, name_map_top),
        ("Tensor PCCA, TICA_dimension=6", Xlist_6, chi_tensor_6_aligned, color_map_bottom, name_map_bottom),
        ("Tensor PCCA, TICA_dimension=8", Xlist_8, chi_tensor_8_aligned, color_map_bottom, name_map_bottom),
    ]

    soft_counts = {}
    for ax, (title, X_case, chi_case, case_color_map, case_name_map) in zip(axes, cases):
        counts, residual_count = plot_soft_pcca_3d(
            ax, X_case, chi_case, title, K, case_color_map, case_name_map, cut=SOFT_PCCA_CUT
        )
        soft_counts[title] = {"state_counts_chi_ge_cut": counts, "residual_count": residual_count,
                              "name_map": case_name_map}

    # Merge legend entries from a top panel (has "Rare") and a bottom panel
    # (has "Intermediate") so both show up, with no duplicates, in a fixed order.
    handles_top, labels_top = axes[0].get_legend_handles_labels()
    handles_bottom, labels_bottom = axes[2].get_legend_handles_labels()
    combined = dict(zip(labels_top, handles_top))
    combined.update(dict(zip(labels_bottom, handles_bottom)))

    preferred_order = ["Folded", "Unfolded", "Misfolded", "Rare", "Intermediate", "Residual"]
    ordered_labels = [l for l in preferred_order if l in combined]
    ordered_handles = [combined[l] for l in ordered_labels]
    fig.legend(ordered_handles, ordered_labels, loc="center right", markerscale=1.5, title="PCCA assignment")

    plt.tight_layout(rect=(0, 0, 0.88, 0.96))
    fig.subplots_adjust(bottom=0.06, top=0.94)
    save_figure(fig, OUTPUT_PATH, also_png=True)
    plt.show()

    print(f"\nSoft PCCA assignment counts with cut={SOFT_PCCA_CUT}:")
    for title, item in soft_counts.items():
        print(f"\n{title}")
        for k, count in enumerate(item["state_counts_chi_ge_cut"]):
            print(f"  {item['name_map'][k]}: {count} points with chi >= {SOFT_PCCA_CUT:.3f}")
        print(f"  Residual: {item['residual_count']} points with max chi < {SOFT_PCCA_CUT:.3f}")


if __name__ == "__main__":
    main()
