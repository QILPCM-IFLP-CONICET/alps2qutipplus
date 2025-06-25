from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr


def find_linearly_independent_rows(mat: NDArray, tol: float = 1e-6) -> Tuple[int]:
    """
    Find indices of a maximal subset of linearly independent columns of the matrix.
    """
    _, inds = qr(mat, mode="r", pivoting=True)
    rank = np.linalg.matrix_rank(mat, tol=tol)
    # The first `rank` indices are linearly independent columns
    return tuple(sorted(inds[:rank]))
