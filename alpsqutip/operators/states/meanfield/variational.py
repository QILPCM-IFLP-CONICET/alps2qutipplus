"""
Variational Mean-field

Build variational approximations to a Gibbsian state.

"""

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.random import random_sample
from scipy.optimize import minimize

from alpsqutip.operators import OneBodyOperator, Operator
from alpsqutip.operators.quadratic import (
    QuadraticFormOperator,
    build_quadratic_form_from_operator,
)
from alpsqutip.operators.states import DensityOperatorMixin
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

from .projections import project_to_n_body_operator


def compute_rel_entropy(
    state: GibbsProductDensityOperator, ham: Operator
) -> np.float64:
    """
    Compute the relative entropy relative to the gibbs state exp(-ham)
    from `state`.

    Parameters
    ----------
    state: GibbsProductDensityOperator
         the reference state
    ham: Operator
         the generator of rho=exp(-ham)

    Returns
    -------
    float64
    The relative entropy S(sigma|exp(-ham))
    """
    return np.real(state.expect(ham + state.logm()))


def mf_quadratic_form_exponential(
    qf_op: QuadraticFormOperator,
    num_fields: int = 1,
    method: Optional[str] = None,
    callback_optimizer: Callable = None,
    ham: Optional[Operator] = None,
) -> GibbsProductDensityOperator:
    """
    Approximate `exp(-qf_op)` as `exp(-h_mf)`
    with h_mf = k_0 + sum_a phi_a q_a

    Parameters
    ----------
    qf_op : QuadraticFormOperator
        The generator of the target state state exp(-qf_op).
    num_fields : int, optional
        The number of terms of terms of qf_op to be kept in the approximation.
        The default is 1.
    method : Optional[str], optional
        The numerical optimization method. T
        he default is None.
    callback_optimizer : Callable, optional
        The callback function to be called on each evaluation.
        The default is None.

    ham: Operator
        The operator to be used as the reference.

    Returns
    -------
    GibbsProductDensityOperator
        The approximated state.

    """

    def build_test_state(coeffs: np.ndarray) -> GibbsProductDensityOperator:
        """
        Build the test state from the coefficients.
        """
        terms = tuple((coef * gen for coef, gen in zip(coeffs, generators)))
        if k0 is not None:
            terms = (k0,) + terms
        k = OneBodyOperator(terms, qf_op.system).tidyup().simplify()
        sigma_k = GibbsProductDensityOperator(k)
        return sigma_k

    def test_state_re(coeffs: np.ndarray) -> np.float64:
        """
        Target function. Computes the relative entropy
        relative to the gibbs state exp(-ham)
        """
        test_state = build_test_state(coeffs)
        return compute_rel_entropy(test_state, ham)

    # Trim negative terms and keep at least num_fields of the remaining
    # terms, in a way that exp(qf_op')~ exp(qf_op)

    qf_op = reduced_quadratic_form_operator(-qf_op, num_fields)
    if ham is None:
        ham = -qf_op.as_sum_of_products()

    # Linear term
    k0 = qf_op.linear_term
    if k0:
        k0 = k0.tidyup() or None

    generators = qf_op.basis
    logging.info("using %s generators", len(generators))
    if len(generators) == 0:
        logging.info(
            ("No 2-body terms found. " "Using the linear term as reference state.")
        )
        if k0:
            sigma_ref = GibbsProductDensityOperator(k0)
        return GibbsProductDensityOperator({}, system=qf_op.system)

    # Now, optimize the relative entropy over states of the form
    # sigma = exp(-k0 - sum_a phi_a Q_a)

    # Generate a initial guess for the coefficients.
    phis = 2 * random_sample(len(generators)) - 1
    try:
        result = minimize(
            test_state_re, phis, method=method, callback=callback_optimizer
        )
        phis = result.x
    except ValueError as val_exc:
        logging.info("Optimization failed with exception %s", val_exc)

    sigma_ref = build_test_state(phis)
    return sigma_ref


def reduced_quadratic_form_operator(
    qf_op: QuadraticFormOperator, num_terms: int
) -> QuadraticFormOperator:
    """
    Build a new quadratic form operator keeping only positive weights.

    Parameters
    ----------
    qf_op: QuadraticFormOperator
         the quadratic form.
    num_terms: int
         number of generators to keep.

    Returns
    -------
    QuadraticFormOperator
    A new `QuadraticFormOperator` with all its weights equal to 1.

    """
    assert num_terms > 0, f"num_terms must be an integer number >0. Got {num_terms}."
    weights, basis = qf_op.weights, qf_op.basis
    if len(weights) == 0:
        return qf_op
    num_terms = min(len(weights), num_terms)
    min_weight = max(0, sorted(weights)[-num_terms])
    generators = tuple(
        (
            basis_op * (weight**0.5)
            for weight, basis_op in zip(weights, basis)
            if weight >= min_weight
        )
    )
    return QuadraticFormOperator(
        generators,
        tuple((1 for i in generators)),
        qf_op.system,
        qf_op.linear_term,
        qf_op.offset,
    )


def self_consistent_mf(
    ham: Operator,
    sigma_ref: Optional[GibbsProductDensityOperator] = None,
    max_steps: int = 10,
    callback: Callable = None,
) -> Tuple[GibbsProductDensityOperator, float]:
    """
    Starting from `sigma_ref` compute an approximation of
    exp(-ham) following a self-consistent algorithm.

    Parameters
    ----------
    ham : Operator
        The generator of the exact state rho=exp(-ham).
    sigma_ref : DensityOperatorMixin, optional
        The initial state to begin the self-consistent loop.
        The default is None. In that case, the initial state is
        the fully mixed state.
    max_steps : int, optional
        Maximum number of self-consistent steps used to improve the solution.
        The default is 10.
    callback: Callable, optional
        Function called on each self-consistent round. The default is None.

    Returns
    -------
    Tuple[GibbsProductDensityOperator, float]
        A tuple of the Gibbs product operators that approximates exp(-ham),
    and the corresponding relative entropy.

    """
    if sigma_ref is None:
        sigma_ref = GibbsProductDensityOperator({}, system=ham.system)

    rel_entropy = compute_rel_entropy(sigma_ref, ham)

    for curr_step in range(max_steps):
        gen_sc = project_to_n_body_operator(ham, nmax=1, sigma=sigma_ref)
        sigma_sc = GibbsProductDensityOperator(gen_sc)
        new_rel_entropy = compute_rel_entropy(sigma_sc, ham)
        if callback is not None:
            callback(sigma_ref, rel_entropy, curr_step)
        if new_rel_entropy >= rel_entropy:
            break
        rel_entropy = new_rel_entropy
        sigma_ref = sigma_sc

    return sigma_ref, rel_entropy


def variational_quadratic_mfa(
    ham: Operator, numfields: int = 1, sigma_ref: DensityOperatorMixin = None, **kwargs
) -> GibbsProductDensityOperator:
    r"""
    Find the Mean field approximation for the exponential
    of an operator using a variational algorithm.

    At the end, improve the solution in a self-consistent
    way, looking for a fixed point of the one-body projection.

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

    Parameters
    ----------
    ham : Operator
        The generator of the exact state rho=exp(-ham).
    numfields : int, optional
        The minimal number of *fields* $\phi_a$ to be included in the
        optimization. If there are several generators of the quadratic form
        with the same weight, numfields is extended to include all of them.
        The default is 1.
    sigma_ref : DensityOperatorMixin, optional
        The initial reference state to project `ham` to a quadratic form.
        The default is None.
    its : int, optional
        Maximum number of recursive rounds. If the operator is already a
        2-body operator, its is set to 0. The default is 0.
    method : Optional[str], optional
        The method used in the numeric optimization. The default is None.
    callback_optimizer : Callable, optional
        Callback function called on each evaluation of the optimizer.
        The default is None.
    max_self_consistent_steps : int, optional
        Maximum number of self-consistent steps used to improve the solution.
        The default is 10.
    callback_self_consistent_step : Callable, optional
        Function called on each self-consistent round. The default is None.

    Returns
    -------
    GibbsProductDensityOperator
        A Gibbs product operators that approximates exp(-ham).

    """

    its: int = kwargs.get("its", 1)
    method: Optional[str] = kwargs.get("method", None)
    callback_optimizer: Callable = kwargs.get("callback_optimizer", None)
    max_self_consistent_steps: int = kwargs.get("max_self_consistent_steps", 10)
    callback_self_consistent_step: Callable = kwargs.get(
        "callback_self_consistent_step", None
    )

    current_rel_entropy = None
    if isinstance(ham, OneBodyOperator):
        return GibbsProductDensityOperator(ham)

    for _ in range(its):
        # We start by projecting the generator `ham` to the two-body sector
        # relative to `sigma_ref`:

        ham_proj = project_to_n_body_operator(ham, nmax=2, sigma=sigma_ref)
        if isinstance(ham_proj, OneBodyOperator):
            sigma_ref = GibbsProductDensityOperator(ham_proj)
        else:
            # Now, write the projected operator as a QuadraticFormOperator
            # ham_proj = k_0 + sum_a w_a Q_a^2
            # with |Q_a|_{infty}=1 and
            # w_1 <= w_2 <=... <=w_l < 0 <= w_{k+1} <= ... w_n
            qf_op = build_quadratic_form_from_operator(ham_proj)
            sigma_ref = mf_quadratic_form_exponential(
                qf_op, numfields, method, callback_optimizer, ham
            )

        if current_rel_entropy is None:
            current_rel_entropy = compute_rel_entropy(sigma_ref, ham)

        # Improve the solution by a self-consistent round
        sigma_ref, rel_s = self_consistent_mf(
            ham,
            sigma_ref,
            max_steps=max_self_consistent_steps,
            callback=callback_self_consistent_step,
        )

        # If the relative entropy have not improved, or
        # the ham==ham_proj
        if (current_rel_entropy - rel_s + ALPSQUTIP_TOLERANCE) > 0 or ham_proj is ham:
            break
        current_rel_entropy = rel_s

    return sigma_ref
