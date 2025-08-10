"""
Routines to build the Gram's matrix associated to a scalar product and a basis.
"""

import logging
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

import alpsqutip.settings as alpsqutip_settings
from alpsqutip.scalarprod.utils import find_linearly_independent_rows

# from datetime import datetime


MAX_WORKERS = alpsqutip_settings.PARALLEL_MAX_WORKERS
USE_THREADS = alpsqutip_settings.PARALLEL_USE_THREADS

if alpsqutip_settings.USE_PARALLEL:
    try:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        from functools import partial

        logging.info("using parallel routines to build Gram's matrices.")
    except ModuleNotFoundError:
        alpsqutip_settings.USE_PARALLEL = False
        logging.warning(
            "ProcessPoolExecutor/ThreadPoolExecutor cannot be loaded. Using serial routines."
        )
else:
    logging.info("using serial routines to build Gram's matrices.")


# ### Generic functions depending on the SP ###
def _sp_worker(pair, basis, sp):
    """
    Compute the real-valued part of the scalar product between two SumOperators belonging to some basis,
    provided these are hermitian.

    Parameters:
        pair (tuple): A tuple (i, j) indicating indices in the basis list.
        basis (list): A list of operator objects forming a basis.
        sp (callable): A scalar product function sp(op1, op2) -> real.

    Returns:
        tuple: A tuple (i, j, val) where val is the real part of sp(basis[i], basis[j]).
    """
    try:
        i, j = pair
        val = float(np.real(sp(basis[i], basis[j])))
        return (i, j, val)
    except Exception as exc_val:
        logging.error(f"Error computing Gram's matrix entry ({i},{j}):{exc_val}")
        return (i, j, np.nan)


def gram_matrix_parallel(basis, sp, num_workers=MAX_WORKERS, use_threads=USE_THREADS):
    """
    Compute the Gram matrix of a set of operators in parallel using a scalar product.

    This function evaluates all inner products ⟨b_i | b_j⟩ for i, j in `basis`
    and returns a real symmetric Gram matrix. Parallelization can be done via
    threads or processes.

    Parameters:
        basis (List[Operator]): List of basis operators.
        sp (Callable): Scalar product function taking two operators and returning a scalar.
                       Must be a top-level, pickleable function if using processes.
                       Notice that sp must be a top-level function, or in general,
                       an object that can be stored with pickle. This does not include
                       lambda functions.
        num_workers (int or None): Number of worker threads or processes to use.
                                   Defaults to the number of cores.
        use_threads (bool): If True, uses threading instead of multiprocessing.
                            Useful when the scalar product is I/O-bound or GIL-friendly.

    Returns:
        np.ndarray: Symmetric real-valued Gram matrix of shape (len(basis), len(basis)).
    """
    size = len(basis)
    result = np.zeros((size, size), dtype=float)

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    index_pairs = list(
        (
            i,
            j,
        )
        for i in range(size)
        for j in range(i + 1)
    )

    # Pre-bind basis and sp to _sp_worker
    worker = partial(_sp_worker, basis=basis, sp=sp)

    with executor_cls(max_workers=num_workers) as executor:
        for i, j, val in executor.map(worker, index_pairs):
            result[i, j] = val
            if i != j:
                result[j, i] = val

    return result.round(14)


def gram_matrix_serial(basis, sp: Callable):
    """
    Computes the Gram matrix of a given operator basis using a scalar product.

    The Gram matrix is symmetric and defined as:
        Gij = sp(op1, op2)
    where `sp` is the scalar product function and `op1, op2` are operators from
    the basis.

    Parameters
    ----------

        basis: A list of basis operators.
        sp: A callable that defines a scalar product function between two
        operators.

    Return
    ------
        A symmetric NumPy array representing the Gram matrix, with entries
        rounded to 14 decimal places.
    """
    if hasattr(sp, "compute_gram_matrix"):  # and all(b_i.isherm for b_i in basis):
        return sp.compute_gram_matrix(basis)

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


gram_matrix = (
    gram_matrix_parallel if alpsqutip_settings.USE_PARALLEL else gram_matrix_serial
)


def merge_gram_matrices(
    g11: NDArray, g11_inv: NDArray, g22: NDArray, g12: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray, Tuple[int]]:
    """
    Build the gram and gram_inv tensors from the available information.

    Given two basis, their Gram matrices, the inverse
    of the first and the cross gram matrix, this function finds
    a set of linearly independent basis elements that span
    the union of both basis, the Gram matrix associated to the new
    basis and its inverse, and the Gram matrices of the reduced blocks.

    Parameters
    ==========

    g11: NDArray
         The upper-left block of the Gram Matrix
    g11_inv: NDArray
         The inverse of the upper-left block of the Gram Matrix
    g22: NDArray
         The bottom-right block of the Gram Matrix
    g22: NDArray
         The cross block of the Gram Matrix

    Return values
    =============

    gram_full: NDArray
        The gram matrix of the reduced basis

    gram_full_inv: NDArray
         The inverse of the gram matrix of the reduced basis

    g11, g22: NDArray
         The sub-blocks of g11 and g22 associated to the
         reduced common basis

    li_indices: Tuple[int]
       the list of indices of the elements in the original basis
       that span the linearly independent common basis.

    """

    # Build the new Gram matrix
    n_1 = len(g11)
    n_2 = len(g22)
    n_total = n_1 + n_2

    g21 = g12.T
    gram_full = np.block([[g11, g12], [g21, g22]])

    # If gram is singular, reduce it and remove the
    # linearly dependent elements.
    li_indices = find_linearly_independent_rows(gram_full)
    li_1_indices = tuple(i for i in li_indices if i < n_1)
    if len(li_1_indices) != n_1:
        raise ValueError("It looks like basis_1 were singular.")

    if len(li_indices) != n_total:
        n_total = len(li_indices)
        if n_total == n_1:
            return g11, g11_inv, g11, g11, li_indices
    n_2 = n_total - n_1
    gram_full = gram_full[li_indices, :][:, li_indices]
    g12 = gram_full[:n_1, n_1:]
    g21 = g12.T
    g22 = gram_full[n_1:, n_1:]

    # --- Gram inverse (block inversion, Schur complement) ---
    # Should not be singular, because we ensure that gram is not
    # singular...
    shur = g22 - g21 @ g11_inv @ g12

    try:
        shur_inv = np.linalg.inv(shur)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Gram matrix is singular after merging bases") from e

    # Build the inverse
    top_left = g11_inv + g11_inv @ g12 @ shur_inv @ g21 @ g11_inv
    top_right = -g11_inv @ g12 @ shur_inv
    bottom_left = -shur_inv @ g21 @ g11_inv
    bottom_right = shur_inv
    gram_full_inv = np.block([[top_left, top_right], [bottom_left, bottom_right]])

    return gram_full, gram_full_inv, g11, g22, li_indices
