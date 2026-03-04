import numpy as np
from tensor_gedmd.basis_sets.basis_sets import BasisSet
from tensor_gedmd.basis_sets.random_fourier_features import RandomFourierFeatures
from tensor_gedmd.basis_sets.product_basis import Product_Basis

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

    basis_list = [SimpleBasis(n=3), SimpleBasis(n=2)]
    product_basis = Product_Basis(basis_list)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    output = product_basis(x)
    print("Product Basis Output:\n", output)

if __name__ == "__main__":
    test_random_fourier_features()
    test_product_basis()   

    