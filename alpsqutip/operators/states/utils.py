"""
Utility functions for alpsqutip.operators.states

"""

from typing import Dict

import numpy as np

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


def safe_exp_and_normalize(operator):
    """Compute `expm(operator)/Z` and `log(Z)`.
    `Z=expm(operator).tr()` in a safe way.
    """
    from alpsqutip.operators.functions import eigenvalues

    k_0 = max(np.real(eigenvalues(operator, sparse=True, sort="high", eigvals=3)))
    op_exp = (operator - k_0).expm()
    op_exp_tr = op_exp.tr()
    op_exp = op_exp * (1.0 / op_exp_tr)
    k_0 = np.log(op_exp_tr) + k_0
    if isinstance(op_exp, LocalOperator):
        loc_op = op_exp.operator
        tr_loc_op = loc_op.tr()
        k_0 += np.log(tr_loc_op)
        return (
            ProductDensityOperator(
                local_states={op_exp.site: loc_op / tr_loc_op},
                system=op_exp.system,
                normalize=False,
            ),
            k_0,
        )
    if isinstance(op_exp, ProductOperator):
        loc_ops = op_exp.sites_op
        tr_ops = {site: l_op.tr() for site, l_op in loc_ops.items()}
        loc_ops = {site: l_op / tr_ops[site] for site, l_op in loc_ops.items()}
        k_0 += sum(np.log(tr_op) for tr_op in tr_ops.values())
        return (
            ProductDensityOperator(
                local_states=loc_ops,
                system=op_exp.system,
                normalize=False,
            ),
            k_0,
        )
    if isinstance(op_exp, QutipOperator):
        return (
            QutipDensityOperator(
                op_exp.operator,
                op_exp.system,
                op_exp.site_names,
                prefactor=1,
            ),
            k_0,
        )

    return op_exp, k_0
