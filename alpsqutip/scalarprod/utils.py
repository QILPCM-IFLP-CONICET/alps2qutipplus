from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from alpsqutip.settings import ALPSQUTIP_TOLERANCE


def find_linearly_independent_rows(
    mat: NDArray, tol: float = ALPSQUTIP_TOLERANCE
) -> Tuple[int]:
    """
    Find indices of a maximal subset of linearly independent columns of the matrix.
    """
    tol = min(tol, min(row[i] for i, row in enumerate(mat)) * 0.25)
    r, inds = qr(mat, mode="r", pivoting=True)
    rank = np.linalg.matrix_rank(r, tol=tol)
    # The first `rank` indices are linearly independent columns
    return tuple(sorted(inds[:rank]))
