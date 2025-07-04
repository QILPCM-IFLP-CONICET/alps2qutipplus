"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

# function used to safely and robustly map K-states to states


def slice_times(tlist: NDArray, tcuts: List[float]) -> List[NDArray]:
    """
    Divides a time list (`tlist`) into slices based on a sequence of cutoff
    times (`tcuts`).

    Parameters
    ----------
    tlist : np.array
        A NumPy array of time values to be sliced.
    tcuts : List[float]
        A list or array of cutoff times used to define the time slices.

    Returns
    -------
    List[ndarray]
        A list of NumPy arrays, where each array corresponds to a segment of
        `tlist`
        based on the intervals defined by `tcuts`.
        - The first slice includes times up to `tcuts[1]`.
        - Subsequent slices include times between `tcuts[d-1]` and `tcuts[d]`.
        - If there are remaining times beyond `tcuts[-1]`, they are included
        in the last slice.

    """

    sliced_times = [np.array([t for t in tlist if t <= tcuts[1]])]

    for d in range(2, len(tcuts)):
        local_tlist = np.array([t for t in tlist if tcuts[d - 1] <= t <= tcuts[d]])
        sliced_times.append(local_tlist)

    if tlist[-1] > tcuts[-1]:
        sliced_times.append(np.array([t for t in tlist if t >= tcuts[-1]]))

    return sliced_times


def m_th_partial_sum(phi: NDArray, m: int) -> float:
    """
    Computes the $m$-th partial sum of the squared magnitudes of the last `m`
    coefficients of `phi`.

    Parameters
    ----------
    phi : ndarray
        A NumPy array containing coefficients.
    m : int
        An integer specifying how many of the last coefficients to include
        in the sum.

    Returns
    -------
    float
        The partial sum of the squared magnitudes of the last `m` coefficients
        in `phi`.

    """
    if m >= len(phi) or m < -len(phi):
        return sum(abs(phi_n) ** 2 for phi_n in phi)
    return sum(abs(phi_n) ** 2 for phi_n in phi[-m:])
