"""
Functions for operators.
"""

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple

from numpy import array as np_array, log, real

from alpsqutip.model import Operator
from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
)


def commutator(A: Operator, B: Operator) -> Operator:
    """
    The commutator of two operators
    """
    return A * B - B * A


def eigenvalues(
    operator: Operator,
    sparse: bool = False,
    sort: str = "low",
    eigvals: int = 0,
    tol: float = 0.0,
    maxiter: int = 100000,
) -> np_array:
    """Compute the eigenvalues of operator"""
    return operator.to_qutip().eigenenergies(sparse, sort, eigvals, tol, maxiter)


def spectral_norm(operator: Operator) -> float:
    """
    Compute the spectral norm of the operator `op`
    """

    if isinstance(operator, LocalOperator):
        return max(operator.operator.eigenenergies() ** 2) ** 0.5
    if isinstance(operator, ProductOperator):
        result = operator.prefactor
        for loc_op in operator.sites_ops.values():
            result *= max(loc_op.eigenenergies() ** 2) ** 0.5
        return result

    return max(eigenvalues(operator) ** 2) ** 0.5


def log_op(operator: Operator) -> Operator:
    """The logarithm of an operator"""

    def qutip_log(local_op):
        if isinstance(local_op, (int, float, complex)):
            return log(local_op)

        evals, evecs = operator.operator.eigenstates()
        return sum(
            v * v.dag() * log(x if x > 0 else x + 1.0e-12j).real
            for x, v in zip(evals, evecs)
        )

    system = operator.system
    prefactor = operator.prefactor
    if isinstance(operator, LocalOperator):
        return LocalOperator(operator.site, qutip_log(operator.operator), system)
    if isinstance(operator, ProductOperator):
        terms = []

        for site, loc_op in operator.sites_op.items():
            terms.append(LocalOperator(site, qutip_log(loc_op), system))
        if prefactor != 1:
            terms.append(ProductOperator({}, log(prefactor), system))

        return OneBodyOperator(terms, system, False)

    return QutipOperator(qutip_log(operator.to_qutip()), system, None, 1)


def relative_entropy(rho: Operator, sigma: Operator) -> float:
    """Compute the relative entropy"""
    return real(rho.expect(log_op(rho) - log_op(sigma)))
