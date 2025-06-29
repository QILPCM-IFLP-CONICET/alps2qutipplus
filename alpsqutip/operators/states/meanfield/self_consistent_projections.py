"""
Module that implements a meanfield approximation of a Gibbsian state
"""

import logging
from typing import Callable, Optional, Tuple, cast

import numpy as np

from alpsqutip.operators import Operator
from alpsqutip.operators.states import DensityOperatorMixin, ProductDensityOperator
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.operators.states.meanfield.projections import (
    project_operator_to_m_body,
    project_to_n_body_operator,
)


def self_consistent_project_meanfield(
    k_op: Operator,
    sigma: Optional[ProductDensityOperator | GibbsProductDensityOperator] = None,
    max_it: int = 100,
    proj_func: Callable = project_operator_to_m_body,
) -> Tuple[Operator, DensityOperatorMixin]:
    """
    Iteratively computes the one-body component from a QuTip operator and state
    using a self-consistent Mean-Field Projection (MF).

    Parameters:
        k_op: The initial operator, a QuTip.Qobj, to be decomposed into
        one-body components.
        sigma: The referential state to be used as the initial guess
               in the calculations.
        k_0: if given, the logarithm of sigma.
        max_it: Maximum number of iterations.

    Returns:
        A tuple (K_one_body, sigma_one_body):
        - K_one_body: The one-body component of the operator K, an
        AlpsQuTip.one_body_operator object.
        - sigma_one_body: The one-body state normalized through the
        MFT process.
    """
    it: int
    curr_sigma: GibbsProductDensityOperator
    new_sigma: GibbsProductDensityOperator
    opt_sigma: GibbsProductDensityOperator
    k_one_body: Operator
    rel_s: float
    rel_s_new: float

    if sigma is None:
        curr_sigma = GibbsProductDensityOperator(k={}, system=k_op.system)
        k_one_body = -(curr_sigma.logm())
    else:
        k_one_body = -(cast(GibbsProductDensityOperator, sigma).logm())
        if not isinstance(sigma, GibbsProductDensityOperator):
            curr_sigma = GibbsProductDensityOperator(k_one_body)
        else:
            curr_sigma = sigma

    rel_s = np.real(cast(complex, curr_sigma.expect(k_op - k_one_body)))
    opt_sigma = curr_sigma
    # print("self consistent loop using", proj_func)
    sigma = None
    for it in range(max_it):
        # k_one_body = project_operator_to_m_body(k_op, 1, sigma)
        k_one_body = project_to_n_body_operator(k_op, 1, curr_sigma).simplify()
        new_sigma = GibbsProductDensityOperator(k_one_body)
        rel_s_new = np.real(cast(complex, curr_sigma.expect(k_op + new_sigma.logm())))
        rel_entropy_txt = f"     S(curr||target)={rel_s_new}"
        logging.debug(rel_entropy_txt)
        # print(it, "->", rel_entropy_txt)
        if it > 20 and rel_s_new > 2 * rel_s:
            # print("  rel_s_new is worst than the optimal. Give up.")
            break

        if rel_s_new < rel_s:
            rel_s = rel_s_new
            opt_sigma = new_sigma
        curr_sigma = new_sigma

    k_one_body = project_to_n_body_operator(k_op, 1, opt_sigma).simplify()
    return k_one_body, opt_sigma
