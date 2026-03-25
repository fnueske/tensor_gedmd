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

    # ----- sample inputs (m=2, d=2) -----
    x = np.array([
        [np.pi / 2, np.pi],
        [np.pi, np.pi / 2],
    ])

    # split coordinates
    x1 = x[:, [0]]
    x2 = x[:, [1]]

    # ----- evaluate 1D RFF features -----
    psi_x = rff_x(x1)  # (m,2)
    psi_y = rff_y(x2)  # (m,2)


    # ----- explicit 4 frequency combinations -----
    comb_00 = psi_x[:, 0] * psi_y[:, 0]
    comb_01 = psi_x[:, 0] * psi_y[:, 1]
    comb_10 = psi_x[:, 1] * psi_y[:, 0]
    comb_11 = psi_x[:, 1] * psi_y[:, 1]

    direct = np.column_stack([comb_00, comb_01, comb_10, comb_11])

    print("\nDirect omega combinations:\n", direct)

    # ----- ProductBasis result -----
    out = product_basis(x)

    print("\nProductBasis output:\n", out)

    # ----- check equality -----
    print("\nCheck:", np.allclose(direct, out))


if __name__ == "__main__":
    test_product_basis_with_rff()
    
