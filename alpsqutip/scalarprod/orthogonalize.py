"""
Routines to compute generalized scalar products over the algebra of operators.
"""

# from datetime import datetime
from typing import Callable

import numpy as np
from numpy.linalg import cholesky, inv, norm, svd
from scipy.linalg import sqrtm

from alpsqutip.scalarprod.gram import gram_matrix


def build_hermitician_basis(basis, sp=lambda x, y: ((x.dag() * y).tr())):
    """
    Build a basis of independent hermitician operators
    from a set of operators, and the coefficients for the expansion
    of basis in terms of the new orthogonal basis.
    """
    # First, find a basis of hermitician operators that generates
    # basis.
    new_basis = []
    indx = 0
    # indices is a list that keeps the connection between the original
    # basis and the hermitician basis
    indices = []
    for b in basis:
        indices.append([])
        if b.isherm:
            if b:
                new_basis.append(b)
                indices[-1].append(
                    (
                        indx,
                        1.0,
                    )
                )
                indx += 1
        else:
            op = b + b.dag()  # .simplify()
            if op:
                new_basis.append(op)
                indices[-1].append(
                    (
                        indx,
                        0.5,
                    )
                )
                indx += 1
            op = 1j * b - 1j * b.dag()  # .simplify()
            if op:
                new_basis.append(1j * (b - b.dag()))
                indices[-1].append(
                    (
                        indx,
                        -0.5j,
                    )
                )
                indx += 1

    # Now, we work with the hermitician basis.
    # The first step is to build the Gram's matrix
    gram_mat = gram_matrix(new_basis, sp)

    # Now, we construct the SVD of the Gram's matrix
    u_mat, s_mat, vd_mat = svd(gram_mat, full_matrices=False, hermitian=True)
    # And find a change of basis to an orthonormalized basis
    t = np.array([row * s ** (-0.5) for row, s in zip(vd_mat, s_mat) if s > 1e-10])
    # and build the hermitician, orthogonalized basis
    new_basis = [
        sum(c * op for c, op in zip(row, new_basis)) for row, s in zip(t, s_mat)
    ]
    # Then, we build the change back to the hermitician basis
    q = np.array([row * s ** (0.5) for row, s in zip(u_mat.T, s_mat) if s > 1e-10]).T
    # Finally, we apply the change of basis to the original (non-hermitician)
    # basis
    q = np.array([sum(spec[1] * q[spec[0]] for spec in row) for row in indices])

    return new_basis, q


def operator_components(op, orthogonal_basis, sp: Callable):
    """
    Get the components of the projection of an operator onto an orthogonal
    basis using a scalar product.

    This computes the components of the orthogonal projection of `op`
    over the basis `orthogonal_basis` with respect to the scalar product `sp`.

    Parameters:
        op: The operator to be projected (e.g., a matrix or quantum operator).
        orthogonal_basis: A list of orthogonalized operators to serve as the
        projection basis.
        sp: A callable that defines the scalar product function between
        two operators.

    Returns:
        A NumPy array containing the projection coefficients, where the i-th
        coefficient represents the projection of `op` onto the i-th element
        of `orthogonal_basis`.
    """
    return np.array([sp(op2, op) for op2 in orthogonal_basis])


def orthogonalize_basis(basis, sp: callable, tol=1e-5):
    """
    Orthogonalize a given basis of operators using the default method.

    Parameters:
        basis: A list of operators (or matrices) to be orthogonalized.
        sp: A callable that defines the scalar product function between two
        operators.
        tol: A tolerance value (default: 1e-5) for verifying the orthogonality
        of the resulting basis.

    Returns:
        orth_basis: A list of orthogonalized operators, normalized with respect
        to the scalar product `sp`.

    Raises:
        AssertionError: If the orthogonalized basis does not satisfy
        orthonormality within the specified tolerance.
    """
    return orthogonalize_basis_gs(basis, sp, tol)


def orthogonalize_basis_gs(basis, sp: callable, tol=1e-5):
    """
    Orthogonalizes a given basis of operators using a scalar product and the
    Gram-Schmidt method.

    Parameters:
        basis: A list of operators (or matrices) to be orthogonalized.
        sp: A callable that defines the scalar product function between two
        operators.
        tol: A tolerance value (default: 1e-5) for verifying the orthogonality
        of the resulting basis.

    Returns:
        orth_basis: A list of orthogonalized operators, normalized with respect
        to the scalar product `sp`.

    Raises:
        AssertionError: If the orthogonalized basis does not satisfy
        orthonormality within the specified tolerance.
    """
    orth_basis = []
    for op_orig in basis:
        norm: float = abs(sp(op_orig, op_orig)) ** 0.5
        if norm < tol:
            continue
        changed = False
        new_op = op_orig / norm
        for prev_op in orth_basis:
            overlap = sp(prev_op, new_op)
            if abs(overlap) > tol:
                new_op -= prev_op * overlap
                changed = True
        if changed:
            norm = np.real(sp(new_op, new_op) ** 0.5)
            if norm < tol:
                continue
            new_op = new_op / norm
        orth_basis.append(new_op)
    return orth_basis


def orthogonalize_basis_cholesky(basis, sp: callable, tol=1e-5):
    """
    Orthogonalizes a given basis of operators using a scalar product and the
    Cholesky decomposition
    method.

    Parameters:
        basis: A list of operators (or matrices) to be orthogonalized.
        sp: A callable that defines the scalar product function between two
        operators.
        tol: A tolerance value (default: 1e-5) for verifying the orthogonality
        of the resulting basis.

    Returns:
        orth_basis: A list of orthogonalized operators, normalized with respect
        to the scalar product `sp`.

    Raises:
        AssertionError: If the orthogonalized basis does not satisfy
        orthonormality within the specified tolerance.
    """
    local_basis = basis

    # Compute the inverse Gram matrix for the given basis
    cholesky_gram_matrix = cholesky(gram_matrix(basis=local_basis, sp=sp), lower=False)
    linv_t = inv(cholesky_gram_matrix).transpose()

    # Construct the orthogonalized basis by linear combinations of
    # the original basis
    orth_basis = [
        sum(local_basis[s] * linv_t[i, s] for s in range(i + 1))
        for i in range(len(local_basis))
    ]

    # Verify the orthogonality by checking that the Gram matrix is
    # approximately the identity matrix
    assert (
        norm(gram_matrix(basis=orth_basis, sp=sp) - np.identity(len(orth_basis))) < tol
    ), "Error: Basis not correctly orthogonalized"

    return orth_basis


def orthogonalize_basis_svd(basis, sp: callable, tol=1e-5):
    """
    Orthogonalizes a given basis of operators using a scalar product and the
    svd decomposition method.

    Parameters:
        basis: A list of operators (or matrices) to be orthogonalized.
        sp: A callable that defines the scalar product function between two
        operators.
        tol: A tolerance value (default: 1e-5) for verifying the orthogonality
        of the resulting basis.

    Returns:
        orth_basis: A list of orthogonalized operators, normalized with respect
        to the scalar product `sp`.

    Raises:
        AssertionError: If the orthogonalized basis does not satisfy
        orthonormality within the specified tolerance.
    """
    local_basis = basis

    # Compute the inverse Gram matrix for the given basis
    inv_gram_matrix = inv(gram_matrix(basis=local_basis, sp=sp))

    # Construct the orthogonalized basis by linear combinations of
    # the original basis
    orth_basis = [
        sum(
            sqrtm(inv_gram_matrix)[j][i] * local_basis[j]
            for j in range(len(local_basis))
        )
        for i in range(len(local_basis))
    ]

    # Verify the orthogonality by checking that the Gram matrix is
    # approximately the identity matrix
    assert (
        norm(gram_matrix(basis=orth_basis, sp=sp) - np.identity(len(orth_basis))) < tol
    ), "Error: Basis not correctly orthogonalized"

    return orth_basis
