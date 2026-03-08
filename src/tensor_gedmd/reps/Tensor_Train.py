# -*- coding: utf-8 -*-

from typing import List, Sequence
import numpy as np


class TT:
    """
    Minimal Tensor Train (TT) container.

    A tensor train is represented by a list of 4D cores:
        core[k].shape = (r_k, n_k, m_k, r_{k+1})

    where
        r_k, r_{k+1} : TT ranks
        n_k          : row mode size
        m_k          : column mode size

    Notes
    -----
    - A standard tensor TT has m_k = 1 for all k.
    - A TT operator / MPO has at least one m_k > 1.
    - This class is a lightweight container and validator for TT cores.
    """

    def __init__(self, cores: Sequence[np.ndarray]):
        if not isinstance(cores, (list, tuple)) or len(cores) == 0:
            raise TypeError("cores must be a non-empty list or tuple of numpy arrays.")

        self._cores: List[np.ndarray] = [np.asarray(core) for core in cores]
        self._validate_cores()

        self.order: int = len(self._cores)
        self.row_dims: List[int] = [core.shape[1] for core in self._cores]
        self.col_dims: List[int] = [core.shape[2] for core in self._cores]
        self.ranks: List[int] = [self._cores[0].shape[0]] + [core.shape[3] for core in self._cores]

    def _validate_cores(self) -> None:
        """Validate TT core shapes and rank compatibility."""
        for i, core in enumerate(self._cores):
            if not isinstance(core, np.ndarray):
                raise TypeError(f"Core {i} must be a numpy.ndarray.")
            if core.ndim != 4:
                raise ValueError(
                    f"Core {i} must be 4-dimensional, got shape {core.shape}."
                )

        for i in range(len(self._cores) - 1):
            if self._cores[i].shape[3] != self._cores[i + 1].shape[0]:
                raise ValueError(
                    f"Rank mismatch between core {i} and core {i+1}: "
                    f"{self._cores[i].shape[3]} != {self._cores[i + 1].shape[0]}"
                )

        if self._cores[0].shape[0] != 1:
            raise ValueError(
                f"First TT rank must be 1, got {self._cores[0].shape[0]}."
            )

        if self._cores[-1].shape[3] != 1:
            raise ValueError(
                f"Last TT rank must be 1, got {self._cores[-1].shape[3]}."
            )

    def __len__(self) -> int:
        """Return TT order."""
        return self.order

    def __getitem__(self, k: int) -> np.ndarray:
        """Return the k-th core."""
        return self.get_core(k)

    def get_core(self, k: int) -> np.ndarray:
        """Return the k-th TT core."""
        if not 0 <= k < self.order:
            raise IndexError(f"Core index out of range: {k}")
        return self._cores[k]

    def tt_ranks(self) -> List[int]:
        """Return TT ranks [r0, r1, ..., rd]."""
        return self.ranks.copy()

    def mode_sizes(self) -> List[tuple]:
        """
        Return mode sizes as a list of (row_dim, col_dim) pairs.
        """
        return list(zip(self.row_dims, self.col_dims))

    def is_operator(self) -> bool:
        """
        Return True if this TT represents an operator (MPO),
        i.e. at least one column mode is greater than 1.
        """
        return any(c > 1 for c in self.col_dims)

    @property
    def cores(self) -> List[np.ndarray]:
        """Return the list of TT cores."""
        return self._cores

    def __repr__(self) -> str:
        kind = "operator" if self.is_operator() else "tensor"
        return (
            f"TT({kind}, order={self.order}, "
            f"row_dims={self.row_dims}, "
            f"col_dims={self.col_dims}, "
            f"ranks={self.ranks})"
        )
    



