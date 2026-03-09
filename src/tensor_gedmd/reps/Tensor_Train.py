# -*- coding: utf-8 -*-

from typing import List, Sequence, Union
import numpy as np


class TT:
    """
    Minimal Tensor Train (TT) container.

    This class supports both:
    - Tensor TT: cores of shape (r_k, n_k, r_{k+1})
    - Operator TT: cores of shape (r_k, n_k, m_k, r_{k+1})

    Notes
    -----
    - All cores must have the same order: either all 3D or all 4D.
    - Tensor TT:
          core[k].shape = (r_k, n_k, r_{k+1})
    - Operator TT:
          core[k].shape = (r_k, n_k, m_k, r_{k+1})
    """

    def __init__(self, cores: Sequence[np.ndarray]):
        if not isinstance(cores, (list, tuple)) or len(cores) == 0:
            raise TypeError("cores must be a non-empty list or tuple of numpy arrays.")

        self._cores: List[np.ndarray] = [np.asarray(core) for core in cores]
        self._ndim: int = self._validate_cores()

        self.order: int = len(self._cores)
        self.is_operator: bool = (self._ndim == 4)

        if self.is_operator:
            self.row_dims: List[int] = [core.shape[1] for core in self._cores]
            self.col_dims: List[int] = [core.shape[2] for core in self._cores]
            self.ranks: List[int] = [self._cores[0].shape[0]] + [core.shape[3] for core in self._cores]
        else:
            self.mode_dims: List[int] = [core.shape[1] for core in self._cores]
            self.ranks: List[int] = [self._cores[0].shape[0]] + [core.shape[2] for core in self._cores]

    def _validate_cores(self) -> int:
        """
        Validate TT cores and return their common order (3 or 4).
        """
        for i, core in enumerate(self._cores):
            if not isinstance(core, np.ndarray):
                raise TypeError(f"Core {i} must be a numpy.ndarray.")
            if core.ndim not in (3, 4):
                raise ValueError(
                    f"Core {i} must be either 3D or 4D, got shape {core.shape}."
                )

        ndim = self._cores[0].ndim

        for i, core in enumerate(self._cores):
            if core.ndim != ndim:
                raise ValueError(
                    "All TT cores must have the same order. "
                    f"Core 0 is {ndim}D, but core {i} is {core.ndim}D."
                )

        if ndim == 3:
            for i in range(len(self._cores) - 1):
                if self._cores[i].shape[2] != self._cores[i + 1].shape[0]:
                    raise ValueError(
                        f"Rank mismatch between core {i} and core {i + 1}: "
                        f"{self._cores[i].shape[2]} != {self._cores[i + 1].shape[0]}"
                    )

            if self._cores[0].shape[0] != 1:
                raise ValueError(
                    f"First TT rank must be 1, got {self._cores[0].shape[0]}."
                )

            if self._cores[-1].shape[2] != 1:
                raise ValueError(
                    f"Last TT rank must be 1, got {self._cores[-1].shape[2]}."
                )

        else:  # ndim == 4
            for i in range(len(self._cores) - 1):
                if self._cores[i].shape[3] != self._cores[i + 1].shape[0]:
                    raise ValueError(
                        f"Rank mismatch between core {i} and core {i + 1}: "
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

        return ndim

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

    def mode_sizes(self) -> Union[List[int], List[tuple]]:
        """
        Return mode sizes.

        Returns
        -------
        list[int]
            For tensor TT: [n1, n2, ..., nd]
        list[tuple[int, int]]
            For operator TT: [(n1, m1), (n2, m2), ..., (nd, md)]
        """
        if self.is_operator:
            return list(zip(self.row_dims, self.col_dims))
        return self.mode_dims.copy()

    @property
    def cores(self) -> List[np.ndarray]:
        """Return the list of TT cores."""
        return self._cores

    def __repr__(self) -> str:
        if self.is_operator:
            return (
                f"TT(operator, order={self.order}, "
                f"row_dims={self.row_dims}, "
                f"col_dims={self.col_dims}, "
                f"ranks={self.ranks})"
            )

        return (
            f"TT(tensor, order={self.order}, "
            f"mode_dims={self.mode_dims}, "
            f"ranks={self.ranks})"
        )












