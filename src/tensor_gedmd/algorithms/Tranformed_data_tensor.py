
#-*- coding: utf-8 -*-

from typing import Dict, List, Mapping, Optional, Sequence, Union
import numpy as np

from tensor_gedmd.reps.Tensor_Train import TT


PsiInput = Union[Mapping[int, np.ndarray], Sequence[np.ndarray]]


class Transformed_Data_Tensor_TT:
    """
    Construct the transformed data tensor in Tensor Train (TT) format.

    This class builds TT cores for the transformed data tensor

        Psi(X) = sum_{k=1}^m (psi_1(x_k) ⊗ ... ⊗ psi_p(x_k)) ⊗ e_k,

    where e_k is the k-th standard basis vector in R^m.

    Parameters
    ----------
    psi : dict[int, np.ndarray] or sequence[np.ndarray]
        Feature map evaluations. Supported formats are:

        - Dictionary with 1-based keys:
          psi[j] has shape (n_j, m), for j = 1, ..., p

        - Sequence of arrays:
          psi[j-1] has shape (n_j, m), for j = 1, ..., p

    p : int, optional
        Number of feature dimensions. If omitted, it is inferred from psi.

    normalize_first_core : bool, default=True
        If True, the first core is scaled by 1 / sqrt(m).

    Attributes
    ----------
    psi : dict[int, np.ndarray]
        Internal normalized 1-based dictionary of feature matrices.
    p : int
        Number of feature dimensions.
    m : int
        Number of samples.
    n : dict[int, int]
        Mode sizes, where n[j] = psi[j].shape[0].
    tt_cores : list[np.ndarray]
        Constructed TT cores after calling build_tt_cores().

    Notes
    -----
    The constructed TT has p + 1 cores:

    - First core:
      G1.shape = (1, n1, m)

    - Intermediate cores:
      Gj.shape = (m, n_j, m) for j = 2, ..., p,
      diagonal in the TT rank indices

    - Final core:
      G_last.shape = (m, m, 1),
      equal to the identity in the sample index

    The dense tensor represented by this TT has shape

        (n1, n2, ..., np, m).
    """

    def __init__(
        self,
        psi: PsiInput,
        p: Optional[int] = None,
        normalize_first_core: bool = True,
    ):
        self.psi: Dict[int, np.ndarray] = self._normalize_psi_input(psi, p)
        self.p: int = len(self.psi)
        self.normalize_first_core: bool = normalize_first_core

        self._validate_psi()

        self.m: int = self.psi[1].shape[1]
        self.n: Dict[int, int] = {
            j: self.psi[j].shape[0] for j in range(1, self.p + 1)
        }
        self.dtype = np.result_type(*[self.psi[j].dtype for j in range(1, self.p + 1)])

        self.tt_cores: List[np.ndarray] = []

    @staticmethod
    def _normalize_psi_input(
        psi: PsiInput,
        p: Optional[int],
    ) -> Dict[int, np.ndarray]:
        """
        Normalize psi input into a 1-based dictionary.

        Parameters
        ----------
        psi : dict[int, np.ndarray] or sequence[np.ndarray]
            Input feature matrices.
        p : int, optional
            Expected number of modes.

        Returns
        -------
        dict[int, np.ndarray]
            Dictionary with keys 1, ..., p.

        Raises
        ------
        TypeError
            If psi has unsupported type.
        ValueError
            If p is invalid or inconsistent with psi.
        """
        if p is not None:
            if not isinstance(p, int):
                raise TypeError(f"p must be an int or None, got {type(p).__name__}.")
            if p < 1:
                raise ValueError(f"p must be at least 1, got {p}.")

        if isinstance(psi, Mapping):
            psi_dict = {int(k): np.asarray(v) for k, v in psi.items()}
            inferred_p = len(psi_dict) if p is None else p

            expected_keys = set(range(1, inferred_p + 1))
            actual_keys = set(psi_dict.keys())

            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys

            if missing:
                raise ValueError(f"psi is missing keys: {sorted(missing)}.")
            if extra:
                raise ValueError(
                    f"psi contains unexpected keys outside 1..{inferred_p}: {sorted(extra)}."
                )

            return {j: psi_dict[j] for j in range(1, inferred_p + 1)}

        if isinstance(psi, Sequence) and not isinstance(psi, (str, bytes)):
            if len(psi) == 0:
                raise ValueError("psi sequence must be non-empty.")

            inferred_p = len(psi) if p is None else p
            if len(psi) != inferred_p:
                raise ValueError(
                    f"Length of psi sequence ({len(psi)}) does not match p={inferred_p}."
                )

            return {j + 1: np.asarray(psi[j]) for j in range(inferred_p)}

        raise TypeError(
            "psi must be either a mapping {1..p -> ndarray} or a sequence of ndarrays."
        )

    def _validate_psi(self) -> None:
        """
        Validate feature map inputs.

        Raises
        ------
        ValueError
            If an input array has invalid shape or inconsistent sample count.
        """
        if len(self.psi) == 0:
            raise ValueError("psi must contain at least one feature matrix.")

        if self.psi[1].ndim != 2:
            raise ValueError(
                f"psi[1] must be 2D of shape (n_1, m), got shape {self.psi[1].shape}."
            )

        m = self.psi[1].shape[1]

        if self.psi[1].shape[0] < 1:
            raise ValueError(
                f"psi[1] must have at least one row, got shape {self.psi[1].shape}."
            )
        if m < 1:
            raise ValueError(
                f"psi[1] must have at least one sample, got shape {self.psi[1].shape}."
            )

        for j in range(1, len(self.psi) + 1):
            arr = self.psi[j]

            if arr.ndim != 2:
                raise ValueError(
                    f"psi[{j}] must be 2D of shape (n_{j}, m), got shape {arr.shape}."
                )

            n_j, m_j = arr.shape

            if n_j < 1:
                raise ValueError(
                    f"psi[{j}] must have at least one row, got shape {arr.shape}."
                )

            if m_j != m:
                raise ValueError(
                    f"All psi[j] must have the same number of samples m. "
                    f"psi[1].shape[1]={m}, but psi[{j}].shape[1]={m_j}."
                )

    def build_tt_cores(self) -> List[np.ndarray]:
        """
        Build and store the TT cores of the transformed data tensor.

        Returns
        -------
        list[np.ndarray]
            List of TT cores of length p + 1.

        Notes
        -----
        Core shapes:
        - First core: (1, n1, m)
        - Intermediate cores: (m, n_j, m)
        - Final core: (m, m, 1)
        """
        m = self.m
        cores: List[np.ndarray] = []

        # First core: shape (1, n1, m)
        n1 = self.n[1]
        G1 = np.empty((1, n1, m), dtype=self.dtype)
        G1[0, :, :] = self.psi[1]

        if self.normalize_first_core:
            G1 = G1 / np.sqrt(m)

        cores.append(G1)

        # Intermediate cores: shape (m, n_j, m), diagonal in rank indices
        diag_idx = np.arange(m)
        for j in range(2, self.p + 1):
            nj = self.n[j]
            Gj = np.zeros((m, nj, m), dtype=self.dtype)
            Gj[diag_idx, :, diag_idx] = self.psi[j].T
            cores.append(Gj)

        # Final core: shape (m, m, 1)
        G_last = np.zeros((m, m, 1), dtype=self.dtype)
        G_last[:, :, 0] = np.eye(m, dtype=self.dtype)
        cores.append(G_last)

        self.tt_cores = cores
        return self.tt_cores

    def to_tt(self) -> TT:
        """
        Return the transformed data tensor as a TT object.

        Returns
        -------
        TT
            Tensor Train object built from the TT cores.

        Notes
        -----
        The returned TT is closed on the right, so
        require_right_rank_one=True is used.
        """
        if not self.tt_cores:
            self.build_tt_cores()
        return TT(self.tt_cores, require_right_rank_one=True)

    def reconstruct_dense(self) -> np.ndarray:
        """
        Reconstruct the TT as a dense tensor.

        Returns
        -------
        np.ndarray
            Dense tensor of shape (n1, n2, ..., np, m).

        Raises
        ------
        RuntimeError
            If TT cores have not been built.

        Warning
        -------
        This operation may be memory-expensive for large problems.
        """
        if not self.tt_cores:
            raise RuntimeError("Call build_tt_cores() first.")

        T = self.tt_cores[0]
        for core in self.tt_cores[1:]:
            T = np.tensordot(T, core, axes=([-1], [0]))

        return np.squeeze(T, axis=(0, -1))

    def sample_tensor(self, k: int) -> np.ndarray:
        """
        Return the dense p-way tensor for sample k.

        This computes

            psi_1(x_k) ⊗ ... ⊗ psi_p(x_k).

        Parameters
        ----------
        k : int
            Zero-based sample index.

        Returns
        -------
        np.ndarray
            Dense tensor of shape (n1, ..., np).

        Raises
        ------
        TypeError
            If k is not an integer.
        IndexError
            If k is out of range.
        """
        if not isinstance(k, int):
            raise TypeError(f"k must be an int, got {type(k).__name__}.")
        if not 0 <= k < self.m:
            raise IndexError(f"k out of range: {k}. Valid range is 0 <= k < {self.m}.")

        T = self.psi[1][:, k]
        for j in range(2, self.p + 1):
            T = np.multiply.outer(T, self.psi[j][:, k])

        if self.normalize_first_core:
            T = T / np.sqrt(self.m)

        return T

    def core_shapes(self) -> List[tuple]:
        """
        Return the shapes of the currently built TT cores.

        Returns
        -------
        list[tuple]
            Core shapes.

        Raises
        ------
        RuntimeError
            If TT cores have not been built.
        """
        if not self.tt_cores:
            raise RuntimeError("Call build_tt_cores() first.")
        return [core.shape for core in self.tt_cores]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}, "
            f"m={self.m}, "
            f"n={self.n}, "
            f"normalize_first_core={self.normalize_first_core}, "
            f"built={bool(self.tt_cores)})"
        )














