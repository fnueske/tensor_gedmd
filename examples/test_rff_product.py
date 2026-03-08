import numpy as np
<<<<<<< HEAD
=======

>>>>>>> tgedmd
from tensor_gedmd.basis_sets.basis_sets import BasisSet
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures
from tensor_gedmd.basis_sets.product_basis import ProductBasis

<<<<<<< HEAD
def test_random_fourier_features():
    # Test with 2D input and 3 basis functions
    omega = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([0.0, np.pi / 4, np.pi / 2])
    rff = RandomFourierFeatures(omega=omega, b=b)

    x = np.array([[np.pi / 2, np.pi / 2], [np.pi, np.pi]])  # (2, 2)
    output = rff(x)
    print("Random Fourier Features Output:\n", output)

def test_product_basis():
    # Create two simple basis sets for 1D input
    class SimpleBasis(BasisSet):
        def __init__(self, n):
            super().__init__()
            self.n = n
            self.d = 1

        def __call__(self, x):
            x_arr = self._format_input(x, expected_dim=1)
            return np.hstack([x_arr ** i for i in range(self.n)])

        def gradient(self, x):
            x_arr = self._format_input(x, expected_dim=1)
            return np.hstack([i * x_arr ** (i - 1) if i > 0 else np.zeros_like(x_arr) for i in range(self.n)])

    basis_list = [SimpleBasis(n=2), SimpleBasis(n=2)]
    product_basis = ProductBasis(basis_list)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    output = product_basis(x)
    print("Product Basis Output:\n", output)

if __name__ == "__main__":
    test_random_fourier_features()
    test_product_basis()   

    
=======



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
    






d
>>>>>>> tgedmd
