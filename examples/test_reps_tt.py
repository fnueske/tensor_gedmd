import numpy as np

cores = [
    np.random.rand(1, 4, 4, 3),
    np.random.rand(3, 5, 5, 2),
    np.random.rand(2, 6, 6, 1),
]

A_tt = TT(cores)

print(A_tt)                 # summary via __repr__
print(A_tt.tt_ranks())      # TT ranks
print(A_tt.get_core(1).shape)
print(A_tt.mode_sizes())
print(A_tt.is_operator())    