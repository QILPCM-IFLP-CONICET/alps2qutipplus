"""
Utility functions for alpsqutip.operators.states

"""

from typing import Dict

import numpy as np
from qutip import tensor as qutip_tensor

from alpsqutip.operators.arithmetic import SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states.basic import ProductDensityOperator
from alpsqutip.operators.states.qutip import QutipDensityOperator
from alpsqutip.qutip_tools.tools import (
    safe_exp_and_normalize as safe_exp_and_normalize_qobj,
)


def k_by_site_from_operator(k: Operator) -> Dict[str, Operator]:
    """
    Return a tuple of a dictionary mapping site names
    to local generators, and a scalar offset.
    """
    if isinstance(k, ScalarOperator):
        system = k.system
        site = next(iter(system.dimensions))
        return {site: k.prefactor * system.site_identity(site)}
    if isinstance(k, LocalOperator):
        return {getattr(k, "site"): getattr(k, "operator")}
    if isinstance(k, ProductOperator):
        prefactor = getattr(k, "prefactor")
        if prefactor == 0:
            return {}
        sites_op = getattr(k, "sites_op")
        if len(sites_op) > 1:
            raise ValueError(
                "k must be a sum of one-body operators, but has a term acting on {k.acts_over()}"
            )
        if len(sites_op) == 0:
            system = k.system
            site = next(iter(system.dimensions))
            return {site: prefactor * system.site_identity(site)}
        if prefactor == 1:
            return sites_op
        return {site: op * prefactor for site, op in sites_op.items()}
    if isinstance(k, SumOperator):
        result = {}
        offset = 0
        for term in getattr(k, "terms"):
            if isinstance(term, LocalOperator):
                site = term.site
                result[site] = term.operator
            elif isinstance(term, ScalarOperator):
                offset += term.prefactor
            elif isinstance(term, SumOperator):
                sub_terms = k_by_site_from_operator(term)
                for sub_site, sub_term in sub_terms.items():
                    if sub_site in result:
                        result[sub_site] += sub_term
                    else:
                        result[sub_site] = sub_term
            else:
                raise TypeError(f"term of {type(term)} not allowed.")

        if offset:
            if result:
                site = next(iter(result))
                result[site] += offset
            return k_by_site_from_operator(ScalarOperator(offset, k.system))
        return result
    raise TypeError(f"k of {type(k)} not allowed.")


def safe_exp_and_normalize_localop(operator: LocalOperator):
    system = operator.system
    site = operator.site
    loc_rho, log_z = safe_exp_and_normalize_qobj(operator.operator)
    logz = sum(
        (
            np.log(dim)
            for site_factor, dim in system.dimensions.items()
            if site != site_factor
        ),
        log_z,
    )
    local_states = {
        site_factor: (
            loc_rho
            if site == site_factor
            else system.site_identity(site_factor) / system.dimensions[site_factor]
        )
        for site_factor in system.sites
    }
    return (
        ProductDensityOperator(
            local_states=local_states,
            system=system,
            normalize=False,
        ),
        logz,
    )


def safe_exp_and_normalize_sumop(operator: SumOperator):
    operator = operator.simplify()
    if not isinstance(operator, SumOperator):
        return safe_exp_and_normalize(operator)
    terms = operator.terms
    acts_over_terms = [term.acts_over() for term in terms]
    if any(len(acts_over) > 1 for acts_over in acts_over_terms):
        return safe_exp_and_normalize_qutip_operator(operator.to_qutip_operator())

    system = operator.system
    local_generators = dict()
    logz = 0
    for acts_over, term in zip(acts_over_terms, terms):
        if len(acts_over) == 0:
            logz += term.prefactor
            continue
        site = next(iter(acts_over))
        op_qutip = term.to_qutip((site,))
        if site in local_generators:
            local_generators[site] = local_generators[site] + op_qutip
        else:
            local_generators[site] = op_qutip

    local_states = {}
    for site, factor_qutip in local_generators.items():
        local_rho, local_f = safe_exp_and_normalize_qobj(factor_qutip)
        local_states[site] = local_rho
        logz += local_f
    for site in system.sites:
        if site not in local_states:
            dim = system.dimensions[site]
            logz += np.log(dim)
            local_states[site] = system.site_identity(site) / dim

    return (
        ProductDensityOperator(
            local_states=local_states,
            system=system,
            normalize=False,
        ),
        logz,
    )


def safe_exp_and_normalize_qutip_operator(operator):

    system = operator.system
    if isinstance(operator, ScalarOperator):
        ln_z = sum((np.log(dim) for dim in system.dimensions.values()))
        return (ScalarOperator(np.exp(-ln_z), system), ln_z + operator.prefactor)

    site_names = operator.site_names
    block = tuple(sorted(site_names, key=lambda x: site_names[x]))
    rho_qutip, logz = safe_exp_and_normalize_qobj(operator.operator)
    rest = tuple(sorted(site for site in system.sites if site not in block))
    operator = qutip_tensor(
        rho_qutip,
        *(system.site_identity(site) / system.dimensions[site] for site in rest),
    )
    logz = logz + sum(np.log(system.dimensions[site]) for site in rest)
    return (
        QutipDensityOperator(
            operator,
            names={site: pos for pos, site in enumerate(block + rest)},
            system=system,
        ),
        logz,
    )


def safe_exp_and_normalize(operator):
    """
    Compute the decomposition of exp(operator) as rho*exp(f)
    with f = Tr[exp(operator)], for operator a Qutip operator.

    operator: Operator | Qobj

    result: Tuple[Operator|Qobj, float]
         (exp(operator)/f , f)

    """

    if isinstance(operator, ScalarOperator):
        system = operator.system
        ln_z = sum((np.log(dim) for dim in system.dimensions.values()))
        return (ScalarOperator(np.exp(-ln_z), system), ln_z + operator.prefactor)
    if isinstance(operator, LocalOperator):
        return safe_exp_and_normalize_localop(operator)
    if isinstance(operator, SumOperator):
        return safe_exp_and_normalize_sumop(operator)
    if isinstance(operator, QutipOperator):
        return safe_exp_and_normalize_qutip_operator(operator)
    if isinstance(operator, Operator):
        return safe_exp_and_normalize_qutip_operator(operator.to_qutip_operator())

    # assume Qobj or any other class with a compatible interface.
    return safe_exp_and_normalize_qobj(operator)
