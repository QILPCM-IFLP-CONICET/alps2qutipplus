"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

import logging
from typing import List

from alpsqutip.meanfield import project_meanfield
from alpsqutip.operators import Operator
from alpsqutip.operators.states import GibbsProductDensityOperator
from alpsqutip.projections import n_body_projection
from alpsqutip.scalarprod import HierarchicalOperatorBasis, fetch_covar_scalar_product

# function used to safely and robustly map K-states to states


def projected_evolution(
    ham, k0, t_span, order, sigma_0, n_body: int = -1
) -> List[Operator]:
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


def adaptative_projected_evolution(
    ham, k0, t_span, order, sigma_0, n_body: int = -1, tol=1e-3
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

    tol: the maximum induced distance between the projected solution
         and the exact solution.

    Returns
    -------
    List[Operator]:
        A list with the solution at times t_span

    """

    def update_basis(k, sigma):
        k_ref_new, sigma = project_meanfield(k, sigma, proj_function=n_body_projection)
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

    basis = update_basis(k0, sigma_0)
    phi_0 = basis.coefficient_expansion(k0)
    result = [k0]

    k_t = k0
    sigma_ref = GibbsProductDensityOperator(k_t)
    last_t = t_ref = 0
    for t in t_span[1:]:
        delta_t = t - t_ref
        phi, error = basis.evolve(delta_t, phi_0)
        if error > max_error_speed * delta_t:
            logging.info(
                (
                    f"At time {t} the estimated error {error} "
                    f"is beyond the expected limit{max_error_speed * delta_t}. Updating basis."
                )
            )
            basis, sigma_ref = update_basis(k_t, sigma_ref)
            t_ref = last_t
            delta_t = t - t_ref
            phi, error = basis.evolve(delta_t, phi_0)
            if error > max_error_speed * delta_t:
                logging.warning(
                    "tolerance goal cannot be reached within this subspace."
                )
                return result
        k_t = basis.operator_from_coefficients(phi)
        result.append(k_t)
        last_t = t

    return [basis.operator_from_coefficients([0]) for t in t_span]
