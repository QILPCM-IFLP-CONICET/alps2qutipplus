"""
Functions for operators.
"""

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple
from typing import Tuple

from numpy import imag, ndarray, real

from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.qutip import QutipOperator

# from alpsqutip.operators.simplify import simplify_sum_operator


def anticommutator(op1, op2):
    """
    Computes the anticommutator of two operators, defined as {op1, op2} = op1 * op2 + op2 * op1.

    Parameters:
        op1, op2: operators (can be a matrix or a quantum operator object).

    Returns:
        The anticommutator of op1 and op2.
    """
    return op1 * op2 + op2 * op1


def commutator(op_1: Operator, op_2: Operator) -> Operator:
    """
    The commutator of two operators
    """
    system = op_1.system or op_2.system
    if isinstance(op_1, SumOperator):
        return SumOperator(
            tuple((commutator(term, op_2) for term in op_1.terms)), system
        ).simplify()
    if isinstance(op_2, SumOperator):
        return SumOperator(
            tuple((commutator(op_1, term) for term in op_2.terms)), system
        ).simplify()

    acts_over_1, acts_over_2 = op_1.acts_over(), op_2.acts_over()
    if acts_over_1 is not None:
        if len(acts_over_1) == 0:
            return ScalarOperator(0, system)
        if acts_over_2 is not None:
            if len(acts_over_2) == 0 or len(acts_over_1.intersection(acts_over_2)) == 0:
                return ScalarOperator(0, system)
    return (op_1 * op_2 - op_2 * op_1).simplify()


def compute_dagger(operator):
    """
    Compute the adjoint of an `operator.
    If `operator` is a number, return its complex conjugate.
    """
    if isinstance(operator, (int, float)):
        return operator
    if isinstance(operator, complex):
        if operator.imag == 0:
            return operator.real
        return operator.conj()
    return operator.dag()


def eigenvalues(
    operator: Operator,
    sparse: bool = False,
    sort: str = "low",
    eigvals: int = 0,
    tol: float = 0.0,
    maxiter: int = 100000,
) -> ndarray:
    """Compute the eigenvalues of operator"""

    qutip_op = operator.to_qutip() if isinstance(operator, Operator) else operator
    if eigvals > 0 and qutip_op.data.shape[0] < eigvals:
        sparse = False
        eigvals = 0

    return qutip_op.eigenenergies(sparse, sort, eigvals, tol, maxiter)


def hermitian_and_antihermitian_parts(operator: Operator) -> Tuple[Operator, Operator]:
    """Decompose an operator Q as A + i B with
    A and B self-adjoint operators
    """
    from alpsqutip.operators.quadratic import QuadraticFormOperator

    system = operator.system
    if operator.isherm:
        return operator, ScalarOperator(0, system)

    if isinstance(operator, QuadraticFormOperator):
        weights = operator.weights
        basis = operator.basis
        system = operator.system
        offset = operator.offset
        if offset is None:
            real_offset, imag_offset = (None, None)
        else:
            real_offset, imag_offset = hermitian_and_antihermitian_parts(offset)
        weights_re, weights_im = tuple((real(w) for w in weights)), tuple(
            (imag(w) for w in weights)
        )
        return (
            QuadraticFormOperator(
                basis, weights_re, system=system, offset=real_offset
            ).simplify(),
            QuadraticFormOperator(
                basis, weights_im, system=system, offset=imag_offset
            ).simplify(),
        )

    if isinstance(operator, ProductOperator):
        sites_op = operator.sites_op
        system = operator.system
        if len(operator.sites_op) == 1:
            site, loc_op = next(iter(sites_op.items()))
            loc_op = loc_op * 0.5
            loc_op_dag = loc_op.dag()
            return (
                LocalOperator(site, loc_op + loc_op_dag, system),
                LocalOperator(site, loc_op * 1j - loc_op_dag * 1j, system),
            )

    elif isinstance(operator, (LocalOperator, OneBodyOperator, QutipOperator)):
        operator = operator * 0.5
        op_dagger = compute_dagger(operator)
        return (
            (operator + op_dagger).simplify(),
            (op_dagger - operator).simplify() * 1j,
        )

    operator = operator * 0.5
    operator_dag = compute_dagger(operator)
    return (
        SumOperator(
            (
                operator,
                operator_dag,
            ),
            system,
            isherm=True,
        ).simplify(),
        SumOperator(
            (
                operator_dag * 1j,
                operator * (-1j),
            ),
            system,
            isherm=True,
        ).simplify(),
    )


def spectral_norm(operator: Operator) -> float:
    """
    Compute the spectral norm of the operator `op`
    """

    if isinstance(operator, LocalOperator):
        return max(operator.operator.eigenenergies() ** 2) ** 0.5
    if isinstance(operator, ProductOperator):
        result = operator.prefactor
        for loc_op in operator.sites_op.values():
            result *= max(loc_op.eigenenergies() ** 2) ** 0.5
        return real(result)

    return max(eigenvalues(operator) ** 2) ** 0.5


def log_op(operator: Operator) -> Operator:
    """The logarithm of an operator"""

    if hasattr(operator, "logm"):
        return operator.logm()
    return operator.to_qutip_operator().logm()


def relative_entropy(rho: Operator, sigma: Operator) -> float:
    """Compute the relative entropy"""

    log_rho = log_op(rho)
    log_sigma = log_op(sigma)
    delta_log = log_rho - log_sigma
    # TODO: fix rho.expect
    # return real(rho.expect(delta_log))
    return real((rho * delta_log).tr())
