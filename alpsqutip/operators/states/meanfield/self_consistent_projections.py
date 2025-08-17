"""
Module that implements a meanfield approximation of a Gibbsian state
"""

import logging
from typing import Optional, Tuple, cast

import numpy as np

from alpsqutip.operators import Operator
from alpsqutip.operators.states import DensityOperatorMixin, ProductDensityOperator
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.operators.states.meanfield.projections import (
    ProjectingOperatorFunction,
    n_body_projection,
)


def self_consistent_project_meanfield(
    k_op: Operator,
    sigma: Optional[ProductDensityOperator | GibbsProductDensityOperator] = None,
    max_it: int = 100,
    tol: float = 1e-12,
    proj_func: Optional[ProjectingOperatorFunction] = n_body_projection,
) -> Tuple[Operator, DensityOperatorMixin]:
    """
    Iteratively computes the one-body component from a QuTip operator and state
    using a self-consistent Mean-Field Projection (MF).


    Parameters
    ----------
    k_op : Operator
        The operator to be projected.
    sigma : Optional[ProductDensityOperator or GibbsProductDensityOperator], optional
        Initial guess for the density operator.
    max_it : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance for relative entropy.
    proj_func : ProjectingOperatorFunction, optional
        Function to project operator to n-body.
    verbose : bool, optional
        If True, prints iteration information.

    Returns
    -------
    k_one_body : Operator
        The projected one-body operator.
    opt_sigma : DensityOperatorMixin
        The optimized one-body density operator.
    """
    converged: bool
    it: int
    sigma_curr: GibbsProductDensityOperator
    sigma_new: GibbsProductDensityOperator
    sigma_opt: GibbsProductDensityOperator
    k_one_body: Operator
    rel_s: float
    rel_s_new: float

    assert isinstance(k_op, Operator)
    assert sigma is None or isinstance(
        sigma, (ProductDensityOperator, GibbsProductDensityOperator)
    )
    assert tol > 0

    if sigma is None:
        sigma_curr = GibbsProductDensityOperator(k={}, system=k_op.system)
        k_one_body = -(sigma_curr.logm())
    else:
        k_one_body = -(cast(GibbsProductDensityOperator, sigma).logm())
        if not isinstance(sigma, GibbsProductDensityOperator):
            sigma_curr = GibbsProductDensityOperator(k_one_body)
        else:
            sigma_curr = sigma

    rel_s = np.real(cast(complex, sigma_curr.expect(k_op - k_one_body)))
    sigma_opt = sigma_curr
    # print("self consistent loop using", proj_func)
    converged = False
    for it in range(max_it):
        # k_one_body = project_operator_to_m_body(k_op, 1, sigma)
        k_one_body = n_body_projection(k_op, 1, sigma_curr).simplify()
        if not k_one_body.isherm:
            k_one_body = (k_one_body + k_one_body.dag()).simplify()

        assert k_one_body.isherm, f"k_one_body is not herm at iteration = {it}"

        sigma_new = GibbsProductDensityOperator(k_one_body)
        rel_s_new = np.real(cast(complex, sigma_curr.expect(k_op + sigma_new.logm())))
        rel_entropy_txt = f"     S(curr||target)={rel_s_new}"
        logging.debug(rel_entropy_txt)
        # print(it, "->", rel_entropy_txt)
        if it > 20:
            if (rel_s_new - rel_s) < tol:
                converged = True
                break
            if rel_s_new > 2 * rel_s:
                break

        if rel_s_new < rel_s:
            rel_s = rel_s_new
            sigma_opt = sigma_new
        sigma_curr = sigma_new

    if converged:
        logging.debug(
            f"  convergence achieved after {it} iterations with SR = {rel_s}."
        )
    else:
        logging.debug(
            f"  rel_s_new {rel_s_new} is much worst than the optimal {rel_s}. Give up."
        )

    return k_one_body, sigma_opt
