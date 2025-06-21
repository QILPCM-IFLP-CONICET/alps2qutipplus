"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np

from alpsqutip.meanfield import (
    self_consistent_project_meanfield,
)
from alpsqutip.operators import Operator
from alpsqutip.operators.states import GibbsProductDensityOperator
from alpsqutip.projections import n_body_projection
from alpsqutip.scalarprod import HierarchicalOperatorBasis, fetch_covar_scalar_product

# function used to safely and robustly map K-states to states


def occupation_factor(phi):
    """
    Compute an estimation of how spread is the operator over the basis.
    """
    partial_sums = np.array(
        [sum(np.abs(phi[:i]) ** 2) ** 0.5 for i in range(1, len(phi))]
    )
    partial_sums = partial_sums / partial_sums[-1]
    for idx, val in enumerate(partial_sums):
        if val > 0.995:
            return idx + 1
    return len(phi)


def adaptative_projected_evolution(
    ham,
    k0,
    t_span,
    order,
    n_body: int = -1,
    tol=1e-3,
    update_basis_callback: Optional[Callable] = None,
) -> List[Operator]:
    """
    Compute the solution of the MaxEnt projected Schrödinger equation

    dk
    -- = -i [H, k]
    dt

    as a linear combination of a an operator basis

    k = sum phi_a(t) Q_a

    chosen adaptatively along the evolution.

    Parameters
    ----------
    ham : Operator
        The Hamiltonian operator
    k0 : Operator
        The initial condition
    t_span: np.array
        the times for with the evolution is computed
    order:
        the order of the solution

    n_body: int
        if non-negative, build a solution projected on
        the subspace of n_body operators.

    tol: float
        the maximum induced distance between the projected solution
        and the exact solution.

    update_basis_callback: Callable[dict], optional
        if not None, this function is called each time the basis is rebuilt.

    Returns
    -------
    List[Operator]:
        A list with the solution at times t_span

    """

    def update_basis(k, sigma):
        k_ref_new, sigma = self_consistent_project_meanfield(
            k, sigma, proj_func=n_body_projection
        )
        return (
            HierarchicalOperatorBasis(
                k,
                ham,
                order,
                fetch_covar_scalar_product(sigma),
                n_body_projection=lambda op_b: n_body_projection(
                    op_b, nmax=n_body, sigma=sigma
                ),
            ),
            sigma,
        )

    t_max = t_span[-1]
    max_error_speed = tol / t_max
    logging.info(f"max_error_speed:{max_error_speed}")

    k_t = k0
    basis, sigma_ref = update_basis(k_t, GibbsProductDensityOperator({}, k_t.system))
    phi_0 = basis.coefficient_expansion(k_t)
    logging.info(f"phi_0={phi_0}")
    result = [k_t]
    last_t = t_ref = 0

    for t in t_span[1:]:
        delta_t = t - t_ref
        phi, error = basis.evolve(delta_t, phi_0)
        oc_factor = occupation_factor(phi)
        if error > max_error_speed * delta_t:
            logging.info(
                (
                    f"At time {t} the estimated error {error} "
                    f"is beyond the expected limit{max_error_speed * delta_t}. Updating basis."
                )
            )
            start_basis_time = datetime.now()
            basis, sigma_ref = update_basis(k_t, sigma_ref)
            build_basis_time_cost = datetime.now() - start_basis_time
            phi_0 = basis.coefficient_expansion(k_t)
            t_ref = last_t
            delta_t = t - t_ref
            phi, error = basis.evolve(delta_t, phi_0)

            if update_basis_callback is not None:
                update_basis_callback(
                    state={
                        "phi_0": phi_0,
                        "phi": phi,
                        "error": error,
                        "delta_t": delta_t,
                        "t_ref": t_ref,
                        "t": t,
                        "basis": basis,
                        "K_t": k_t,
                        "basis time cost": build_basis_time_cost,
                        "oc_factor": oc_factor,
                    }
                )
            if error > max_error_speed * delta_t:
                logging.warning(
                    "tolerance goal cannot be reached within this subspace."
                )
                return result
        k_t = basis.operator_from_coefficients(phi)
        result.append(k_t)
        last_t = t

    return result


def projected_evolution(ham, k0, t_span, order, n_body: int = -1) -> List[Operator]:
    """
    Compute the solution of the MaxEnt projected Schrödinger equation

    dk
    -- = -i [H, k]
    dt

    as a linear combination of the iterated commutators

    k = sum phi_a(t) Q_a

    Parameters
    ----------
    ham : Operator
        The Hamiltonian operator
    k0 : Operator
        The initial condition
    t_span: np.array
        the times for with the evolution is computed
    order:
        the order of the solution

    n_body: int
        if non-negative, build a solution projected on
        the subspace of n_body operators.

    Returns
    -------
    List[Operator]:
        A list with the solution at times t_span

    """
    sigma_0 = GibbsProductDensityOperator(k0)
    sp = fetch_covar_scalar_product(sigma_0)

    basis = HierarchicalOperatorBasis(
        k0,
        ham,
        order,
        sp,
        n_body_projection=lambda op_b: n_body_projection(
            op_b, nmax=n_body, sigma=sigma_0
        ),
    )

    phi_0 = basis.coefficient_expansion(k0)
    return [basis.operator_from_coefficients(basis.evolve(t, phi_0)[0]) for t in t_span]


def adaptative_projected_evolution_b(
    ham,
    k0,
    t_span,
    order,
    n_body: int = -1,
    tol=1e-3,
    update_basis_callback: Optional[Callable] = None,
    extra_observables: Tuple = tuple(),
) -> List[Operator]:
    """
    Compute the solution of the MaxEnt projected Schrödinger equation

    dk
    -- = -i [H, k]
    dt

    as a linear combination of a an operator basis

    k = sum phi_a(t) Q_a

    chosen adaptatively along the evolution.

    Parameters
    ----------
    ham : Operator
        The Hamiltonian operator
    k0 : Operator
        The initial condition
    t_span: np.array
        the times for with the evolution is computed
    order:
        the order of the solution

    n_body: int
        if non-negative, build a solution projected on
        the subspace of n_body operators.

    tol: float
        the maximum induced distance between the projected solution
        and the exact solution.

    update_basis_callback: Callable[dict], optional
        if not None, this function is called each time the basis is rebuilt.

    Returns
    -------
    List[Operator]:
        A list with the solution at times t_span

    """

    def update_basis(k, sigma):
        k_ref_new, sigma = self_consistent_project_meanfield(
            k, sigma, proj_func=n_body_projection
        )
        new_basis = HierarchicalOperatorBasis(
            k,
            ham,
            order,
            fetch_covar_scalar_product(sigma),
            n_body_projection=lambda op_b: n_body_projection(
                op_b, nmax=n_body, sigma=sigma
            ),
        )

        rest_elements = extra_observables
        if k is not k_ref_new:
            rest_elements = rest_elements + (k_ref_new,)
        if rest_elements:
            new_basis = new_basis + rest_elements
        return (
            new_basis,
            sigma,
        )

    t_max = t_span[-1]
    max_error_speed = tol / t_max
    logging.info(f"max_error_speed:{max_error_speed}")

    k_t = k0
    basis, sigma_ref = update_basis(k_t, GibbsProductDensityOperator({}, k_t.system))
    phi_0 = basis.coefficient_expansion(k_t)
    logging.info(f"phi_0={phi_0}")
    result = [k_t]
    last_t = t_ref = 0

    for t in t_span[1:]:
        delta_t = t - t_ref
        phi, error = basis.evolve(delta_t, phi_0)
        oc_factor = occupation_factor(phi)
        if error > max_error_speed * delta_t:
            logging.info(
                (
                    f"At time {t} the estimated error {error} "
                    f"is beyond the expected limit{max_error_speed * delta_t}. Updating basis."
                )
            )
            start_basis_time = datetime.now()
            basis, sigma_ref = update_basis(k_t, sigma_ref)
            build_basis_time_cost = datetime.now() - start_basis_time
            phi_0 = basis.coefficient_expansion(k_t)
            t_ref = last_t
            delta_t = t - t_ref
            phi, error = basis.evolve(delta_t, phi_0)

            if update_basis_callback is not None:
                update_basis_callback(
                    state={
                        "phi_0": phi_0,
                        "phi": phi,
                        "error": error,
                        "delta_t": delta_t,
                        "t_ref": t_ref,
                        "t": t,
                        "basis": basis,
                        "K_t": k_t,
                        "basis time cost": build_basis_time_cost,
                        "oc_factor": oc_factor,
                    }
                )
            if error > max_error_speed * delta_t:
                logging.warning(
                    "tolerance goal cannot be reached within this subspace."
                )
                return result
        k_t = basis.operator_from_coefficients(phi)
        result.append(k_t)
        last_t = t

    return result


def adaptative_projected_evolution_c(
    ham,
    k0,
    t_span,
    order,
    n_body: int = -1,
    tol=1e-3,
    update_basis_callback: Optional[Callable] = None,
) -> List[Operator]:
    """
    Compute the solution of the MaxEnt projected Schrödinger equation

    dk
    -- = -i [H, k]
    dt

    as a linear combination of a an operator basis

    k = sum phi_a(t) Q_a

    chosen adaptatively along the evolution.

    Parameters
    ----------
    ham : Operator
        The Hamiltonian operator
    k0 : Operator
        The initial condition
    t_span: np.array
        the times for with the evolution is computed
    order:
        the order of the solution

    n_body: int
        if non-negative, build a solution projected on
        the subspace of n_body operators.

    tol: float
        the maximum induced distance between the projected solution
        and the exact solution.

    update_basis_callback: Callable[dict], optional
        if not None, this function is called each time the basis is rebuilt.

    Returns
    -------
    List[Operator]:
        A list with the solution at times t_span

    """

    def update_basis(k, sigma):
        k_ref_new, sigma = self_consistent_project_meanfield(
            k, sigma, proj_func=n_body_projection
        )
        new_basis = HierarchicalOperatorBasis(
            k_ref_new,
            ham,
            order,
            fetch_covar_scalar_product(sigma),
            n_body_projection=lambda op_b: n_body_projection(
                op_b, nmax=n_body, sigma=sigma
            ),
        )
        if k is not k_ref_new:
            new_basis = new_basis + (k,)
        return (
            new_basis,
            sigma,
        )

    t_max = t_span[-1]
    max_error_speed = tol / t_max
    logging.info(f"max_error_speed:{max_error_speed}")

    k_t = k0
    basis, sigma_ref = update_basis(k_t, GibbsProductDensityOperator({}, k_t.system))
    phi_0 = basis.coefficient_expansion(k_t)
    logging.info(f"phi_0={phi_0}")
    result = [k_t]
    last_t = t_ref = 0

    for t in t_span[1:]:
        delta_t = t - t_ref
        phi, error = basis.evolve(delta_t, phi_0)
        oc_factor = occupation_factor(phi)
        if error > max_error_speed * delta_t:
            logging.info(
                (
                    f"At time {t} the estimated error {error} "
                    f"is beyond the expected limit{max_error_speed * delta_t}. Updating basis."
                )
            )
            start_basis_time = datetime.now()
            basis, sigma_ref = update_basis(k_t, sigma_ref)
            build_basis_time_cost = datetime.now() - start_basis_time
            phi_0 = basis.coefficient_expansion(k_t)
            t_ref = last_t
            delta_t = t - t_ref
            phi, error = basis.evolve(delta_t, phi_0)

            if update_basis_callback is not None:
                update_basis_callback(
                    state={
                        "phi_0": phi_0,
                        "phi": phi,
                        "error": error,
                        "delta_t": delta_t,
                        "t_ref": t_ref,
                        "t": t,
                        "basis": basis,
                        "K_t": k_t,
                        "basis time cost": build_basis_time_cost,
                        "oc_factor": oc_factor,
                    }
                )
            if error > max_error_speed * delta_t:
                logging.warning(
                    "tolerance goal cannot be reached within this subspace."
                )
                return result
        k_t = basis.operator_from_coefficients(phi)
        result.append(k_t)
        last_t = t

    return result
