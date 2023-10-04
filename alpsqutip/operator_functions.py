"""
Functions for operators.
"""

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple
from numbers import Number
from typing import Tuple

from numpy import array as np_array, real

from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.scalarprod import orthogonalize_basis


def commutator(op_1: Operator, op_2: Operator) -> Operator:
    """
    The commutator of two operators
    """
    system = op_1.system or op_2.system
    if isinstance(op_1, SumOperator):
        return SumOperator([commutator(term, op_2) for term in op_1.terms], system)
    if isinstance(op_2, SumOperator):
        return SumOperator([commutator(op_1, term) for term in op_2.terms], system)

    act_over_1, act_over_2 = op_1.act_over(), op_2.act_over()
    if act_over_1 is not None:
        if len(act_over_1) == 0:
            return ScalarOperator(0, system)
        if act_over_2 is not None:
            if len(act_over_2) == 0 or len(act_over_1.intersection(act_over_2)) == 0:
                return ScalarOperator(0, system)

    return simplify_sum_operator(op_1 * op_2 - op_2 * op_1)


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
) -> np_array:
    """Compute the eigenvalues of operator"""
    return operator.to_qutip().eigenenergies(sparse, sort, eigvals, tol, maxiter)


def hermitian_and_antihermitian_parts(operator) -> Tuple[Operator]:
    """Decompose an operator Q as A + i B with
    A and B self-adjoint operators
    """
    system = operator.system
    if operator.isherm:
        return operator, ScalarOperator(0, system)

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
        return (operator + op_dagger, (op_dagger - operator) * 1j)

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
        ),
        SumOperator(
            (
                operator_dag * 1j,
                operator * (-1j),
            ),
            system,
            isherm=True,
        ),
    )


def reduce_by_orthogonalization(operator_list):
    """
    From a list of operators whose sum spans another operator,
    produce a new list with linear independent terms
    """

    def scalar_product(op_1, op_2):
        return (op_1.dag() * op_2).tr()

    basis = orthogonalize_basis(operator_list, sp=scalar_product)
    if len(basis) > len(operator_list):
        return operator_list
    coeffs = [
        sum(scalar_product(op_b, term) for term in operator_list) for op_b in basis
    ]

    return [op_b * coeff for coeff, op_b in zip(coeffs, basis)]


def simplify_sum_operator(operator):
    """
    Try to simplify a sum of operators by flatten it,
    classifying the terms according to which subsystem acts,
    reducing the partial sums.
    """

    if not isinstance(operator, SumOperator):
        return operator.simplify()

    operator_terms = operator.terms
    if len(operator_terms) < 2:
        return operator.simplify()

    system = operator.system
    isherm = operator._isherm

    null_subsystem = tuple()
    terms_by_subsystem = {
        null_subsystem: [0.0],
        None: [],
    }

    def process_term(term):
        """
        Flatten the list of terms and classify them
        according to over which subsystem act.
        """
        if isinstance(term, Number):
            terms_by_subsystem.setdefault(null_subsystem, []).append(term)
        if isinstance(term, SumOperator):
            for sub_term in term.terms:
                process_term(sub_term)
            return
        sites = tuple(term.act_over())
        terms_by_subsystem.setdefault(sites, []).append(term)

    # Flatten and classify the terms
    for term in operator_terms:
        term = term.simplify()
        process_term(term)

    # Reduce the partial sums
    new_terms = []
    one_body_terms = []
    scalar_term = 0
    for subsystem, terms in terms_by_subsystem.items():
        if subsystem is None:
            new_terms.extend(terms)
            continue
        if len(subsystem) > 1:
            terms = reduce_by_orthogonalization(terms)
            new_terms.extend(terms)
            continue
        if len(subsystem) == 0:
            scalar_term = sum(terms)
            continue
        assert len(subsystem) == 1
        one_body_terms.extend(terms)

    # One-body terms are put together and added as a OneBodyOperator term
    if one_body_terms:
        one_body_term = (
            one_body_terms[0]
            if len(one_body_terms) == 1
            else OneBodyOperator(tuple(one_body_terms), system)
        )
        if scalar_term:
            one_body_term = one_body_term + scalar_term
        new_terms.append(one_body_term)
    elif scalar_term:
        if isinstance(scalar_term, Number):
            scalar_term = ScalarOperator(scalar_term, system)
        new_terms.append(scalar_term)

    # Build the return value
    if new_terms:
        if len(new_terms) == 1:
            return new_terms[0]

        if not isherm:
            isherm = None
        return SumOperator(tuple(new_terms), system, isherm)
    return ScalarOperator(0.0, system)


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

    if hasattr(operator, "logm"):
        return operator.logm()
    return operator.to_qutip_operator().logm()


def relative_entropy(rho: Operator, sigma: Operator) -> float:
    """Compute the relative entropy"""

    log_rho = log_op(rho)
    log_sigma = log_op(sigma)

    return real(rho.expect(log_rho - log_sigma))
