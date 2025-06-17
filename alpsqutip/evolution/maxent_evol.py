"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy import linalg

from alpsqutip.evolution.hierarchical_basis import (
    build_hierarchical_basis,
    fn_hij_tensor_with_errors,
    k_state_from_phi_basis,
)
from alpsqutip.operators import Operator
from alpsqutip.operators.states.meanfield.projections import project_to_n_body_operator
from alpsqutip.scalarprod import fetch_covar_scalar_product, orthogonalize_basis

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
    h_basis = build_hierarchical_basis(ham, k0, order)
    sp = fetch_covar_scalar_product(sigma_0)
    # Project to n_body subspace if required
    if n_body >= 0:
        h_basis = [
            project_to_n_body_operator(op_b, nmax=n_body, sigma=sigma_0)
            for op_b in h_basis
        ]
    h_basis = orthogonalize_basis(h_basis, sp)
    hij, werrs = fn_hij_tensor_with_errors(h_basis, sp, ham)
    result = []
    phi0 = np.array([sp(b_op, k0) for b_op in h_basis])
    for indx, t in enumerate(t_span):
        phi = linalg.expm(hij * t) @ phi0
        k_inst = k_state_from_phi_basis(phi, h_basis)
        result.append(k_inst)
    return result
