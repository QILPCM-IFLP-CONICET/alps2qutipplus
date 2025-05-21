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
    of an operator using a variational algorithm.





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
    if ham_proj is ham:
        print("trivial projection")

    if isinstance(ham, OneBodyOperator):
        return GibbsProductDensityOperator(ham)

    qf_op = build_quadratic_form_from_operator(ham_proj)
    min_weight = min(sorted(qf_op.weights)[numfields], 0)

    generators_and_weights = sorted(
        (pair for pair in zip(qf_op.weights, qf_op.basis) if pair[0] <= min_weight),
        key=lambda x: x[0],
    )
    generators = [
        -weight * base_op.tidyup() for weight, base_op in generators_and_weights
    ]
    print("using ", len(generators), "generators")
    # TODO: if the generators do not acts over the whole system, consider
    # adding more generators.

    k0 = qf_op.linear_term
    if k0:
        k0 = k0.tidyup() or None

    # If there are not any two-body terms,
    # try a self-consistent step:
    if len(generators) == 0:
        if k0:
            # Self consistent step
            sigma_ref = GibbsProductDensityOperator(k0)
            if its:
                return variational_quadratic_mfa(
                    ham, numfields, sigma_ref, its - 1, method, callback
                )
            return sigma_ref
        return ProductDensityOperator({}, system=ham.system)

    def build_test_state(coeffs):
        terms = tuple((coef * gen for coef, gen in zip(coeffs, generators)))
        if k0 is not None:
            terms = (k0,) + terms
        k = OneBodyOperator(terms, qf_op.system).tidyup().simplify()
        sigma_k = GibbsProductDensityOperator(k)
        return sigma_k

    def compute_rel_entropy(state):
        return np.real(state.expect(ham + state.logm()))

    def test_state_re(coeffs):
        test_state = build_test_state(coeffs)
        return compute_rel_entropy(test_state)

    phis = 2 * random_sample(len(generators)) - 1
    # phis = np.zeros(numfields)
    result = minimize(test_state_re, phis, method=method, callback=callback)
    sigma_ref = build_test_state(result.x)
    rel_entropy = compute_rel_entropy(sigma_ref)
    print("**", rel_entropy)
    for i in range(10):
        gen_sc = project_to_n_body_operator(ham, nmax=1, sigma=sigma_ref)
        sigma_sc = GibbsProductDensityOperator(gen_sc)
        new_rel_entropy = compute_rel_entropy(sigma_sc)
        print("**", new_rel_entropy)
        if new_rel_entropy >= rel_entropy:
            break
        rel_entropy = new_rel_entropy
        sigma_ref = sigma_sc

    if its == 0:
        print(f"<delta H>={rel_entropy}")
        return sigma_ref

    if rel_entropy < ALPSQUTIP_TOLERANCE**0.5:
        print(f"<rel_entropy>={rel_entropy} OK!")
        return sigma_ref
    print(f"<rel_entropy>={rel_entropy} iterate")

    return variational_quadratic_mfa(
        ham,
        numfields=numfields,
        sigma_ref=sigma_ref,
        its=its - 1,
        method=method,
        callback=callback,
    )
