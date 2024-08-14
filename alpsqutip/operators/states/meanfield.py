"""
Module that implements a meanfield approximation of a Gibbsian state
"""

import numpy as np

from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.states.states import ProductDensityOperator


def one_body_from_qutip_operator(operator, sigma0=None):
    """

    decompose an operator K (operator)
    given in the sparse Qutip form,
    into a sum of two operators
    K = K_0 + Delta K
    with K_0 a OneBodyOperator and
    DeltaK s.t.
    Tr[DeltaK sigma] = 0

    By default, sigma0 is the maximally mixed state.
    """
    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=operator.system)
        sigma0 = sigma0 / sigma0.tr()

    system = sigma0.system
    tr_value = (operator * sigma0.to_qutip_operator()).tr()
    operator = operator - tr_value
    average_term = ScalarOperator(tr_value, system)
    local_states = {
        name: sigma0.partial_trace([name]).to_qutip() for name in system.dimensions
    }
    local_terms = [average_term]
    for name in local_states:
        sigma_compl = ProductOperator(
            {
                name: s_loc
                for name_loc, s_loc in local_states.items()
                if name != name_loc
            },
            system=system,
        )
        loc_op = (sigma_compl * operator).partial_trace([name])
        local_terms.append(LocalOperator(name, loc_op.to_qutip(), system))

    one_body_term = OneBodyOperator(tuple(local_terms), system=system)

    remaining = (operator - one_body_term).to_qutip_operator()
    return SumOperator(
        tuple(
            (
                average_term,
                one_body_term,
                remaining,
            )
        ),
        system,
    )


def project_meanfield(operator, sigma0=None, **kwargs):
    """
    Build a self-consistent meand field approximation
    of -log(exp(-operator)) as a OneBodyOperator
    """

    # Operators that are already "self consistent
    if type(operator) in (LocalOperator, ScalarOperator, OneBodyOperator):
        return operator

    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=operator.system)

    # cache already computed mean values
    current_sigma = kwargs.get("current_sigma", None)
    if sigma0 is not current_sigma:
        kwargs["current_sigma"] = sigma0
        kwargs["meanvalues"] = {}
    meanvalues = kwargs["meanvalues"]

    if isinstance(operator, SumOperator):
        return sum(project_meanfield(term, sigma0, **kwargs) for term in operator.terms)

    def get_meanvalue(name, op_l):
        """compute the local mean values regarding sigma0"""
        key = (name, id(op_l))
        result = meanvalues.get(key, None)
        if result is None:
            sigma_local = sigma0.partial_trace([name]).to_qutip()
            result = (op_l * sigma_local).tr()
            meanvalues[key] = result
        return result

    if isinstance(operator, ProductOperator):
        system = operator.system
        # Compute the mean values of each factor
        factors = {
            name: get_meanvalue(name, factor)
            for name, factor in operator.sites_op.items()
        }

        # The projection for a product operator
        # o = o_1 ... o_i ... o_j
        # is given by
        # prod_i <op_i> + sum_i  (op_i-<op_i> ) * prod_{j\neq i} <op_j>

        # Here we collect these terms. The first term
        # is the expectation value of the operator
        terms = [np.prod(list(factors.values()))]

        for name, local_op in operator.sites_op.items():
            prefactor = np.prod([f for name_f, f in factors.items() if name_f != name])
            loc_mv = factors[name]
            terms.append(
                LocalOperator(name, prefactor * (local_op - loc_mv), system=system)
            )
        return sum(terms)

    raise TypeError(f"Unsupported operator type '{type(operator)}'")


def self_consistent_meanfield(operator, sigma0=None, max_it=100) -> ProductOperator:
    """
    Build a self-consistent approximation of
    rho \\propto \\exp(-operator)
    as a product operator.

    If sigma0 is given, it is used as the first step.
    """
    operator = operator.flat()
    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=operator.system)
        sigma0 = sigma0 / sigma0.tr()

    curr_it = max_it
    while curr_it:
        curr_it -= 1
        kappa = project_meanfield(operator, sigma0)
        sigma0 = (-kappa).expm()
        sigma0 = sigma0 / sigma0.tr()

    return sigma0
