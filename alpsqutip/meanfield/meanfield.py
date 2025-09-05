"""
Module that implements a meanfield approximation of a Gibbsian state
"""

from typing import Callable, Optional

from alpsqutip.meanfield.self_consistent_projections import (
    self_consistent_project_meanfield,
)
from alpsqutip.operators import Operator
from alpsqutip.operators.states import DensityOperatorMixin, ProductDensityOperator
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.projections import project_operator_to_m_body


def project_meanfield(
    k_op,
    sigma0: Optional[ProductDensityOperator | GibbsProductDensityOperator] = None,
    max_it: int = 100,
    proj_func: Callable = project_operator_to_m_body,
) -> Operator:
    """
    Look for a one-body operator kmf s.t
    Tr (k_op-kmf)exp(-kmf)=0

    following a self-consistent, iterative process
    assuming that exp(-kmf)~sigma0

    If sigma0 is not provided, sigma0 is taken as the
    maximally mixed state.

    """
    sigma: DensityOperatorMixin = self_consistent_project_meanfield(
        k_op, sigma0, max_it, proj_func=proj_func
    )[1]
    result = proj_func(k_op, 1, sigma).simplify()
    # result = n_body_projection(k_op, 1, sigma0).simplify()
    return result
