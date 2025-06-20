"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

from typing import List

from alpsqutip.operators import Operator
from alpsqutip.operators.states.meanfield.projections import project_to_n_body_operator
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
        n_body_projection=lambda op_b: project_to_n_body_operator(
            op_b, nmax=n_body, sigma=sigma_0
        ),
    )

    phi_0 = basis.coefficient_expansion(k0)
    return [basis.operator_from_coefficients(basis.evolve(t, phi_0)[0]) for t in t_span]
