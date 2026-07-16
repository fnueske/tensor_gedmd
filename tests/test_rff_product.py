import numpy as np

from tensor_gedmd.basis_sets.basis_sets import BasisSet
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures
from tensor_gedmd.basis_sets.product_basis import ProductBasis


def test_product_basis_with_rff():
    # ----- define two 1D RFF bases (2 frequencies each) -----
    omega_x = np.array([[1.0], [2.0]])
    b_x = np.array([0.0, np.pi / 4])

    omega_y = np.array([[0.5], [1.5]])
    b_y = np.array([np.pi / 2, np.pi / 3])

    rff_x = RandomFourierFeatures(omega=omega_x, b=b_x)
    rff_y = RandomFourierFeatures(omega=omega_y, b=b_y)

    # ----- product basis -----
    product_basis = ProductBasis([rff_x, rff_y])

    # ----- sample inputs (d=2, m=2) -----
    X = np.array([
        [np.pi / 2, np.pi],   # dimension 0 samples
        [np.pi, np.pi / 2],   # dimension 1 samples
    ])

    x1 = X[0:1, :]  # (1, m)
    x2 = X[1:2, :]  # (1, m)

    # ----- evaluate 1D RFF features -----
    psi_x = rff_x(x1)  # (n, m) = (2, 2)
    psi_y = rff_y(x2)  # (n, m) = (2, 2)

    # ----- explicit 4 frequency combinations, per sample -----
    comb_00 = psi_x[0, :] * psi_y[0, :]
    comb_01 = psi_x[0, :] * psi_y[1, :]
    comb_10 = psi_x[1, :] * psi_y[0, :]
    comb_11 = psi_x[1, :] * psi_y[1, :]

    direct = np.stack([comb_00, comb_01, comb_10, comb_11], axis=0)  # (4, m)

    # ----- ProductBasis result -----
    out = product_basis(X)  # (4, m)

    assert out.shape == direct.shape == (4, 2)
    assert np.allclose(direct, out)


def test_product_basis_gradient_matches_finite_differences():
    omega_x = np.array([[1.0], [2.0]])
    b_x = np.array([0.0, np.pi / 4])

    omega_y = np.array([[0.5], [1.5]])
    b_y = np.array([np.pi / 2, np.pi / 3])

    rff_x = RandomFourierFeatures(omega=omega_x, b=b_x)
    rff_y = RandomFourierFeatures(omega=omega_y, b=b_y)
    product_basis = ProductBasis([rff_x, rff_y])

    X = np.array([[0.3, 0.9], [0.6, 1.1]])  # (d=2, m=2)
    grad = product_basis.gradient(X)  # (n, d, m) = (4, 2, 2)
    assert grad.shape == (4, 2, 2)

    eps = 1e-6
    for dim in range(2):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[dim, :] += eps
        X_minus[dim, :] -= eps

        fd = (product_basis(X_plus) - product_basis(X_minus)) / (2 * eps)
        assert np.allclose(grad[:, dim, :], fd, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_product_basis_with_rff()
    test_product_basis_gradient_matches_finite_differences()
    print("ok")
