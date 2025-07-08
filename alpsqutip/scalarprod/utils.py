from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr


def find_linearly_independent_rows(mat: NDArray, tol: float = 1e-6) -> Tuple[int]:
    """
    Find indices of a maximal subset of linearly independent columns of the matrix.
    """
    (mat,) = qr(mat, mode="r")
    weights = abs(np.diag(mat))
    result = tuple(idx for idx, val in enumerate(abs(weights)) if val > tol)
    print(result)
    return result
