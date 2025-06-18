"""
Spectral-related functions for operators.
"""

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple

from numpy import ndarray, real

from alpsqutip.operators.arithmetic import OneBodyOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)

# from alpsqutip.operators.simplify import simplify_sum_operator


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


def spectral_norm(operator: Operator) -> float:
    """
    Compute the spectral norm of the operator `op`
    """

    if isinstance(operator, ScalarOperator):
        return abs(operator.prefactor)
    if isinstance(operator, LocalOperator):
        if operator.isherm:
            return max(abs(operator.operator.eigenenergies()))
        op_qutip = operator.operator
        return max(abs((op_qutip.dag() * op_qutip).eigenenergies())) ** 0.5
    if isinstance(operator, ProductOperator):
        result = abs(operator.prefactor)
        for loc_op in operator.sites_op.values():
            if loc_op.isherm:
                result *= max(abs(loc_op.eigenenergies()))
            else:
                result *= max((loc_op.dag() * loc_op).eigenenergies()) ** 0.5
        return real(result)

    if operator.isherm:
        if isinstance(operator, OneBodyOperator):
            operator = operator.simplify()
            return sum(spectral_norm(term) for term in operator.terms)
        return max(abs(eigenvalues(operator)))
    return max(eigenvalues(operator.dag() * operator)) ** 0.5


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

    return real(rho.expect(delta_log))
