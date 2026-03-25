import numpy as np

from tensor_gedmd.reps.Tensor_Train import TT
from tensor_gedmd.algorithms.Tranformed_data_tensor import Transformed_Data_Tensor_TT


def main():
    np.random.seed(0)

    # -------------------------------------------------
    # Example feature maps
    # psi_j shape = (n_j, m)
    # -------------------------------------------------
    m = 5
    psi = {
        1: np.random.randn(3, m),
        2: np.random.randn(4, m),
        3: np.random.randn(2, m),
    }

    # -------------------------------------------------
    # Build transformed data tensor TT
    # -------------------------------------------------
    builder = Transformed_Data_Tensor_TT(psi=psi)
    cores = builder.build_tt_cores()

    print("Builder object:")
    print(builder)
    print()

    # -------------------------------------------------
    # Core shapes
    # -------------------------------------------------
    print("Core shapes:")
    print(builder.core_shapes())
    print()

    # -------------------------------------------------
    # Create TT object
    # -------------------------------------------------
    psi_tt = builder.to_tt()

    print("TT object:")
    print(psi_tt)
    print("TT ranks:", psi_tt.tt_ranks())
    print("Mode sizes:", psi_tt.mode_sizes())
    print()

    # -------------------------------------------------
    # Dense reconstruction
    # -------------------------------------------------
    dense = builder.reconstruct_dense()
    print("Dense tensor shape:", dense.shape)
    print()

    # -------------------------------------------------
    # Check sample tensors
    # -------------------------------------------------
    print("Checking sample tensors:")
    for k in range(m):
        sample_tensor = builder.sample_tensor(k)
        dense_slice = dense[..., k]

        err = np.linalg.norm(sample_tensor - dense_slice)
        print(f"Sample {k}: error = {err:.2e}")

    print()
    print("All tests finished.")


if __name__ == "__main__":
    main()