"""
Tests for max-ent solvers
"""

from test.helper import (
    GIBBS_GENERATOR_TESTS,
    HAMILTONIAN,
    OPERATOR_TYPE_CASES,
    SX_AB,
    TEST_CASES_STATES,
    check_operator_equality,
)

import numpy as np
import pytest
import qutip

from alpsqutip.evolution import (
    build_hierarchical_basis,
    fn_hij_tensor_with_errors,
    projected_evolution,
    series_evolution,
)
from alpsqutip.operators.functions import commutator, spectral_norm
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.scalarprod import fetch_covar_scalar_product, orthogonalize_basis

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


HAMILTONIANS = {
    name: op_ham for name, op_ham in OPERATOR_TYPE_CASES.items() if op_ham.isherm
}


def check_orthogonality(basis, sp):
    """
    check if basis is orthonormal
    """
    for pos, b_op in enumerate(basis):
        norm = sp(b_op, b_op) ** 0.5
        assert abs(norm - 1) < 1e-6, (
            f"basis element at position {pos}" f"have norm {norm}"
        )
    for pos1, b_op1 in enumerate(basis):
        for pos2, b_op2 in enumerate(basis):
            if b_op1 is b_op2:
                continue
            overlap = sp(b_op1, b_op2)
            assert abs(norm - 1) < 1e-6, (
                f"basis element at position {pos1} and {pos2} "
                f" fail to be orthogonal in {overlap}"
            )


def compare_basis(basis, basis_qutip):
    """
    Compare a basis of Operator objects
    and a basis of raw qutip operators.
    """
    assert len(basis) == len(basis_qutip)
    idx = 0
    for b_op, qutip_b_op in zip(basis, basis_qutip):
        assert b_op.isherm
        assert qutip_b_op.isherm
        assert check_operator_equality(
            b_op.to_qutip(), qutip_b_op
        ), f"Operators in position {idx} do not match."
        idx += 1


def compare_solutions(sol, sol_qutip, t_span, order, coeff_bound):
    """
    compare two solutions
    """
    diff_sols = [
        spectral_norm(k1.to_qutip() - k2_qutip) for k1, k2_qutip in zip(sol, sol_qutip)
    ]
    assert all(
        error <= coeff_bound * t**order + 1.0e-8 for t, error in zip(t_span, diff_sols)
    )


def fn_hij_tensor_with_errors_from_qutip(basis, sp, ham_j):

    comm_h_ops = [(ham_j * op2 - op2 * ham_j) for op2 in basis]
    local_h_ij = np.zeros([len(basis), len(basis)], dtype=complex)
    for i, b in enumerate(basis):
        for j, comm_op in enumerate(comm_h_ops):
            res = sp(b, comm_op)
            local_h_ij[i, j] = res

    proj_comm_norms_sq = (sum(col**2) for col in local_h_ij.transpose())
    comm_full_norms_sq = (sp(comm_op, comm_op) for comm_op in comm_h_ops)

    errors_w = [
        (max(full_sq - proj_sq, 0.0)) ** 0.5
        for full_sq, proj_sq in zip(comm_full_norms_sq, proj_comm_norms_sq)
    ]
    return local_h_ij, errors_w


@pytest.mark.parametrize(
    ["name_ham", "ham", "name_k0", "k0", "sigma_name", "sigma"],
    [
        (name_ham, ham, name_k0, k0, name_sigma, sigma)
        for name_ham, ham in HAMILTONIANS.items()
        for name_k0, k0 in GIBBS_GENERATOR_TESTS.items()
        for name_sigma, sigma in TEST_CASES_STATES.items()
        if isinstance(sigma, GibbsProductDensityOperator)
    ],
)
def test_build_hierarchical_basis(name_ham, ham, name_k0, k0, sigma_name, sigma):
    """
    Check the construction of hierarchical basis
    """
    from alpsqutip.operators.quadratic import QuadraticFormOperator

    if isinstance(ham, QuadraticFormOperator):
        return
    if isinstance(k0, QuadraticFormOperator):
        return

    deep = 3
    print("H=", name_ham, "K0=", name_k0, "sigma=", sigma_name)
    qutip_ham = ham.to_qutip() * (-1j)
    qutip_k0 = k0.to_qutip()
    qutip_sigma = sigma.to_qutip()

    print("ham:\n", ham)
    print("\n\nk0:\n", k0)

    # Construction
    print("Build basis")
    h_basis = build_hierarchical_basis(ham, k0, deep)
    assert all(
        b_op.isherm for b_op in h_basis
    ), "operators in a Hierarchical basis must be hermitician."
    h_basis_qutip = [qutip_k0]
    for i in range(deep):
        h_basis_qutip.append(
            qutip_ham * h_basis_qutip[-1] - h_basis_qutip[-1] * qutip_ham
        )

    print("Comparing basis against raw qutip")
    compare_basis(h_basis, h_basis_qutip)

    sp = fetch_covar_scalar_product(sigma)

    def sp_qutip(x, y):
        return np.real((qutip_sigma * x * y).tr())

    h_basis_orth = orthogonalize_basis(h_basis, sp)
    h_basis_orth_qutip = orthogonalize_basis(h_basis_qutip, sp_qutip)

    check_orthogonality(h_basis_orth, sp)
    check_orthogonality(h_basis_orth_qutip, sp_qutip)

    compare_basis(h_basis_orth, h_basis_orth_qutip)
    # Build H
    print("Build Hij")
    hij, werrs = fn_hij_tensor_with_errors(h_basis_orth, sp, ham)
    hij_qutip, werrs_qutip = fn_hij_tensor_with_errors_from_qutip(
        h_basis_orth_qutip, sp_qutip, qutip_ham
    )

    assert all(
        abs(x) < 1e-9 for x in (hij_qutip - hij).flatten()
    ), "hij and hij_qutip do not match."
    print("werrs qutip", werrs_qutip)
    print("werrs      ", werrs)
    assert all(
        abs(x - y) < 1e-6 for x, y in zip(werrs_qutip, werrs)
    ), f"werrs and werrs_qutip do not match {[abs(x - y)  for x, y in zip(werrs_qutip, werrs)]}."

    # check commutators

    comm1 = commutator(ham, k0).simplify() / 1j
    comm1_qutip = qutip_ham * qutip_k0 - qutip_k0 * qutip_ham

    delta_phi1 = np.array([abs(sp(b_op, comm1)) for b_op in h_basis_orth])
    delta_phi1_qutip = np.array(
        [abs(sp_qutip(b_op, comm1_qutip)) for b_op in h_basis_orth_qutip]
    )
    assert len(delta_phi1) == len(delta_phi1_qutip)
    if len(delta_phi1) > 0:
        assert all(abs(x - y) < 1e-10 for x, y in zip(delta_phi1, delta_phi1_qutip)), (
            f"Delta phi1 = {delta_phi1}\n" f"in qutip   = {delta_phi1_qutip}\n"
        )


def test_evolution():
    """
    Compare the evolution with the series expansion solution
    """
    t_span = np.linspace(0, 0.1, 200)
    k0 = SX_AB
    sigma = GibbsProductDensityOperator(k0)
    qutip_solution = qutip.mesolve(HAMILTONIAN.to_qutip(), k0.to_qutip(), t_span).states
    for order in range(1, 5):
        as_series_solution = series_evolution(HAMILTONIAN, k0, t_span, order)
        compare_solutions(as_series_solution, qutip_solution, t_span, order, 0.5)

        projected_solution = projected_evolution(
            HAMILTONIAN, k0, t_span, order, sigma_0=sigma
        )
        compare_solutions(projected_solution, qutip_solution, t_span, order, 0.5)
