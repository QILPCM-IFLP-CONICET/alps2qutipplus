"""
Routines to compute generalized scalar products over the algebra of operators.
"""

# from datetime import datetime
from typing import Callable

import numpy as np
from numpy import real
from numpy.linalg import cholesky, inv, norm, svd
from scipy.linalg import sqrtm

from alpsqutip.operators import Operator
from alpsqutip.operators.functions import anticommutator

#  ### Functions that build the scalar products ###


def fetch_kubo_scalar_product(sigma: Operator, threshold=0) -> Callable:
    """
    Build a KMB scalar product function
    associated to the state `sigma`
    """
    evals_evecs = sorted(zip(*sigma.eigenstates()), key=lambda x: -x[0])
    w = 1
    for i, val_vec in enumerate(evals_evecs):
        p = val_vec[0]
        w -= p
        if w < threshold or p <= 0:
            evals_evecs = evals_evecs[: i + 1]
            break

    def ksp(op1, op2):
        result = sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * (p1 if p1 == p2 else (p1 - p2) / np.log(p1 / p2))
            )
            for p1, v1 in evals_evecs
            for p2, v2 in evals_evecs
            if (p1 > 0 and p2 > 0)
        )

        #    stored[key] = result
        return result

    return ksp


def fetch_kubo_int_scalar_product(sigma: Operator) -> Callable:
    """
    Build a KMB scalar product function
    associated to the state `sigma`, from
    its integral form.
    """

    evals, evecs = sigma.eigenstates()

    def return_func(op1, op2):
        return 0.01 * sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * ((p1) ** (1.0 - tau))
                * ((p1) ** (tau))
            )
            for p1, v1 in zip(evals, evecs)
            for p2, v2 in zip(evals, evecs)
            for tau in np.linspace(0.0, 1.0, 100)
            if (p1 > 0.0 and p2 > 0.0)
        )

    return return_func


def fetch_covar_scalar_product(sigma: Operator) -> Callable:
    """
    Returns a scalar product function based on the covariance of a density
    operator.

    The scalar product for two operators op1 and op2 is defined as:
        0.5 * Tr(sigma * {op1†, op2}),
    where sigma is a density operator, {op1†, op2} is the anticommutator of
    the Hermitian conjugate of op1 and op2, and Tr denotes the trace.

    Parameters:
        sigma: The density operator (quantum state) used to define the scalar
        product.

    Returns:
        A function that takes two operators (op1, op2) and computes their
        covariance-based scalar product.
    """

    def sp_(op1: Operator, op2: Operator):
        """Correlation scalar product between
        two operators"""
        op1_herm = op1.isherm
        op2_herm = op2.isherm
        if op1_herm:
            if op2_herm:
                return real(sigma.expect(op1 * op2))
            op1_dag = op1
        else:
            op1_dag = op1.dag()
        if op1_dag is op2:
            return sigma.expect((op1_dag * op2).simplify())
        else:
            return 0.5 * sigma.expect(anticommutator(op1_dag, op2))

    return sp_


def fetch_HS_scalar_product() -> Callable:
    """
    Build a HS scalar product function
    """
    return lambda op1, op2: (op1.dag() * op2).tr()


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
