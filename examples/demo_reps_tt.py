import numpy as np

from tensor_gedmd.reps.tensor_train import TT


cores = [
    np.random.rand(1, 4, 3),
    np.random.rand(3, 5, 2),
    np.random.rand(2, 6, 1),
]

T = TT(cores)
print(T)
print(T.is_operator)   # False
print(T.tt_ranks())    # [1, 3, 2, 1]
print(T.mode_sizes())  # [4, 5, 6]


cores_1 = [
    np.random.rand(1, 4, 4, 3),
    np.random.rand(3, 5, 5, 2),
    np.random.rand(2, 6, 6, 1),
]

A = TT(cores_1)
print(A)
print(A.is_operator)   # True
print(A.tt_ranks())    # [1, 3, 2, 1]
print(A.mode_sizes())  # [(4, 4), (5, 5), (6, 6)]






