"""
Routines to compute generalized scalar products over the algebra of operators.
"""

# from datetime import datetime
from typing import Callable

import numpy as np

from alpsqutip.operators import Operator

#  ### Functions that build the scalar products ###


def fetch_kubo_scalar_product(sigma: Operator, threshold=0):
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
    stored = {}

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


def fetch_kubo_int_scalar_product(sigma: Operator):
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


def fetch_corr_scalar_product(sigma: Operator):
    """
    Build a correlation scalar product function
    associated to the state `sigma`
    """

    def sp_(op1: Operator, op2: Operator):
        """Correlation scalar product between
        two operators"""
        op1_herm = op1.isherm
        op2_herm = op2.isherm
        if op1_herm:
            if op2_herm:
                return sigma.expect(op1 * op2).real
            op1_dag = op1
        else:
            op1_dag = op1.dag()
        if op1_dag is op2:
            w = (op1_dag * op2).simplify()
        else:
            w = (op1_dag * op2 + op2 * op1_dag).simplify()

        return 0.5 * sigma.expect(w)

    return sp_


def fetch_HS_scalar_product():
    """
    Build a HS scalar product function
    """
    return lambda op1, op2: (op1.dag() * op2).tr()


# ### Generic functions depending on the SP ###


def gram_matrix(basis: list, sp: Callable):
    """
    Build the Gramm matrix for the scalar
    product `sp` and the operators in  `basis`
    """
    size = len(basis)
    result = np.zeros([size, size], dtype=float)

    for i, op1 in enumerate(basis):
        for j, op2 in enumerate(basis):
            if j < i:
                continue
            entry = np.real(sp(op1, op2))
            if i == j:
                result[i, i] = entry
            else:
                result[i, j] = entry
                result[j, i] = entry

    return result


def orthogonalize_basis(basis: list, sp: Callable, idop: Operator = None):
    """
    Orthogonalize a `basis` of operators regarding
    the scalar product `sp`, by looking at the eigenvalues
    of the Gramm's matrix.

    If `idop` is given, ensures that the ortogonalized basis
    has elements orthogonal to the identity operator.
    """

    if idop:
        idop = idop * sp(idop, idop) ** (-0.5)
        basis = [idop] + [op - sp(idop, op) * idop for op in basis]

    normalizations = [np.real(sp(op, op)) for op in basis]
    basis = [
        op / (norm) ** 0.5 for op, norm in zip(basis, normalizations) if norm > 1.0e-100
    ]

    for i in range(1):
        gs = gram_matrix(basis, sp)
        lvecs, evals, rvecs = np.linalg.svd(gs)
        coeffs = [(vec) / (val**0.5) for vec, val in zip(rvecs, evals) if val > 1e-20]
        basis = [sum(c * op for c, op in zip(w, basis)) for w in coeffs]
    return basis


def project_op(op: Operator, orthogonal_basis: list, sp: Callable):
    """
    Compute the components of the orthogonal projection of
    `op` over the basis `orthogonal_basis` regarding the scalar
    product `sp`.
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
    for m, b in enumerate(basis):
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
