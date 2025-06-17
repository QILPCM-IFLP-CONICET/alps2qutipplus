"""
Routines to build the Gram's matrix associated to a scalar product and a basis.
"""

# from datetime import datetime
from typing import Callable

import numpy as np

# ### Generic functions depending on the SP ###


def gram_matrix(basis, sp: Callable):
    """
    Computes the Gram matrix of a given operator basis using a scalar product.

    The Gram matrix is symmetric and defined as:
        Gij = sp(op1, op2)
    where `sp` is the scalar product function and `op1, op2` are operators from
    the basis.

    Parameters:
        basis: A list of basis operators.
        sp: A callable that defines a scalar product function between two
        operators.

    Returns:
        A symmetric NumPy array representing the Gram matrix, with entries
        rounded to 14 decimal places.
    """
    size = len(basis)
    result = np.zeros([size, size], dtype=float)

    for i, op1 in enumerate(basis):
        for j, op2 in enumerate(basis):
            if j < i:
                continue  # Use symmetry: Gij = Gji.
            entry = np.real(sp(op1, op2))
            if i == j:
                result[i, i] = entry  # Diagonal elements.
            else:
                result[i, j] = result[j, i] = entry  # Off-diagonal elements.

    return result.round(14)
