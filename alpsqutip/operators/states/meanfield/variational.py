"""
Variational Mean-field

Build variational approximations to a Gibbsian statate.

"""

import logging
from typing import Callable, Optional

import numpy as np
from numpy.random import random_sample
from scipy.optimize import minimize

from alpsqutip.operators import OneBodyOperator, Operator
from alpsqutip.operators.quadratic import build_quadratic_form_from_operator
from alpsqutip.operators.states import DensityOperatorMixin
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

from .projections import project_to_n_body_operator


def variational_quadratic_mfa(
    ham: Operator,
    numfields: int = 1,
    sigma_ref: DensityOperatorMixin = None,
    its: int = 0,
    method: Optional[str] = None,
    callback: Callable = None,
):
    r"""
    Find the Mean field approximation for the exponential
    of a quadratic form.

    Decompose ham as a quadratic form

    ```
    ham = sum_a w_a Q_a^2 + L + delta_ham
    ```
    Then keep `numfields` terms of the sum with maximal weights,
    and look for a variational mean field state
    ```
    sigma \propto exp(-\sum_a phi_a Q_a + L)
    ```
    for real values of `phi_a`.

    Returns `sigma` as a GibbsProductOperator.
    """

    ham_proj = project_to_n_body_operator(ham, nmax=2, sigma=sigma_ref)
    qf_op = build_quadratic_form_from_operator(ham_proj)
    generators_and_weights = sorted(
        zip(qf_op.weights, qf_op.basis), key=lambda x: x[0]
    )[:numfields]
    generators = [
        -weight * base_op.tidyup()
        for weight, base_op in generators_and_weights
        if weight < 0
    ]
    k0 = qf_op.linear_term.tidyup() or None

    def build_test_state(coeffs):
        terms = tuple((coef * gen for coef, gen in zip(coeffs, generators)))
        if k0 is not None:
            terms = (k0,) + terms
        k = OneBodyOperator(terms, qf_op.system).tidyup().simplify()
        return GibbsProductDensityOperator(k)

    def test_state_re(coeffs):
        test_state = build_test_state(coeffs)
        return np.real(test_state.expect(ham + test_state.logm()))

    phis = 2 * random_sample(numfields) - 1
    # phis = np.zeros(numfields)
    result = minimize(test_state_re, phis, method=method, callback=callback)
    sigma_ref = build_test_state(result.x)
    error = abs(sigma_ref.expect(qf_op - ham_proj)) if ham_proj is not ham else 0.
    if its == 0:
        logging.info(f"<delta H>={error}")
        return sigma_ref

    if error < ALPSQUTIP_TOLERANCE**0.5:
        return sigma_ref

    return variational_quadratic_mfa(
        ham,
        numfields=numfields,
        sigma_ref=sigma_ref,
        its=its - 1,
        method=method,
        callback=callback,
    )
