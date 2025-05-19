"""
Module that implements a meanfield approximation of a Gibbsian state
"""

import logging
from typing import Tuple

import numpy as np
from alpsqutip.operators import Operator

from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.operators.states.meanfield.projections import project_operator_to_m_body, project_to_n_body_operator


def self_consistent_project_meanfield(
    k_op, sigma=None, max_it=100, proj_func=project_operator_to_m_body
) -> Tuple[Operator, Operator]:
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
    if sigma is None:
        sigma = GibbsProductDensityOperator(k={}, system=k_op.system)
        neg_log_sigma = -sigma.logm()
    else:
        neg_log_sigma = -sigma.logm()
        if not isinstance(sigma, GibbsProductDensityOperator):
            sigma = GibbsProductDensityOperator(neg_log_sigma)

    rel_s = 10000
    opt_sigma = sigma

    for it in range(max_it):
        # k_one_body = project_operator_to_m_body(k_op, 1, sigma)
        k_one_body = project_to_n_body_operator(k_op, 1, sigma).simplify()
        new_sigma = GibbsProductDensityOperator(k_one_body)

        log_k_one_body = new_sigma.logm()
        rel_s_new = np.real(sigma.expect(k_op + log_k_one_body))
        rel_entropy_txt = f"     S(curr||target)={rel_s_new}"
        logging.debug(rel_entropy_txt)
        # print(rel_entropy_txt)
        if it > 20 and rel_s_new > 2 * rel_s:
            break

        if rel_s_new < rel_s:
            rel_s = rel_s_new
            opt_sigma = new_sigma
            sigma = new_sigma
        else:
            sigma = new_sigma

    k_one_body = project_to_n_body_operator(k_op, 1, opt_sigma).simplify()
    return k_one_body, opt_sigma
