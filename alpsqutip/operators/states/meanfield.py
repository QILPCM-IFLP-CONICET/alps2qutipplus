"""
Module that implements a meanfield approximation of a Gibbsian state
"""

from typing import Optional, Union

import numpy as np
from qutip import Qobj

from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states.states import (
    DensityOperatorMixin,
    ProductDensityOperator,
)


def one_body_from_qutip_operator(
    operator: Union[Operator, Qobj], sigma0: Optional[DensityOperatorMixin] = None
) -> SumOperator:
    """
    Decompose a qutip operator as a sum of an scalar term,
    a one-body term and a remainder, with
    the one-body term and the reamainder having zero mean
    regarding sigma0.

    Parameters
    ----------
    operator : Union[Operator, qutip.Qobj]
        the operator to be decomposed.
    sigma0 : DensityOperatorMixin, optional
        A Density matrix. If None (default) it is assumed to be
        the maximally mixed state.

    Returns
    -------
    SumOperator
        A sum of a Scalar Operator (the expectation value of `operator`
       w.r.t `sigma0`), a LocalOperator and a QutipOperator.

    """

    system = sigma0.system if sigma0 is not None else None

    if isinstance(operator, Qobj):
        operator = QutipOperator(operator, system)

    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=operator.system)
        sigma0 = sigma0 / sigma0.tr()
        system = sigma0.system

    local_states = {
        name: sigma0.partial_trace([name]).to_qutip() for name in system.dimensions
    }

    local_terms = []
    averages = 0
    for name in local_states:
        # Build a product operator Sigma_compl
        # s.t. Tr_{i}Sigma_i =Tr_i sigma0
        #      Tr{/i} Sigma_i = Id
        # Then, for any local operators q_i, q_j s.t.
        # Tr[q_i sigma0]= Tr[q_j sigma0]=0,
        # Tr_{/i}[q_i  Sigma_compl] = q_i
        # Tr_{/i}[q_j  Sigma_compl] = 0
        # Tr_{/i}[q_i q_j Sigma_compl] = 0

        sigma_compl_factors = {
            name_loc: s_loc
            for name_loc, s_loc in local_states.items()
            if name != name_loc
        }
        sigma_compl = ProductOperator(
            sigma_compl_factors,
            system=system,
        )
        local_term = (sigma_compl * operator).partial_trace([name])
        # Split the zero-average part from the average
        local_average = (local_term * local_states[name]).tr()
        averages += local_average
        local_term = local_term - local_average
        local_terms.append(LocalOperator(name, local_term.to_qutip(), system))

    average_term = ScalarOperator(averages / len(local_terms), system)
    one_body_term = OneBodyOperator(tuple(local_terms), system=system)
    remaining = (operator - one_body_term - average_term).to_qutip_operator()
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
