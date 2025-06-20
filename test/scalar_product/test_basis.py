from test.helper import HAMILTONIAN, SX_A, SX_TOTAL, SY_TOTAL, check_operator_equality

import numpy as np

from alpsqutip.operators.functions import commutator
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.scalarprod.basis import HierarchicalOperatorBasis, OperatorBasis
from alpsqutip.scalarprod.build import fetch_covar_scalar_product
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

K0_REFERENCE = SX_A
HAMILTONIAN_REFERENCE = HAMILTONIAN + SX_TOTAL
GENERATOR_REFERENCE = HAMILTONIAN_REFERENCE * 1j
SIGMA_REFERENCE = GibbsProductDensityOperator(K0_REFERENCE)
REFERENCE_SP = fetch_covar_scalar_product(SIGMA_REFERENCE)

BASIS_REFERENCE = [K0_REFERENCE]
for i in range(5):
    BASIS_REFERENCE.append(commutator(BASIS_REFERENCE[-1], HAMILTONIAN_REFERENCE * 1j))


def compare_basis(b1, b2):
    """
    Compare the tensors of both basis
    """
    assert len(b1.operator_basis) == len(b2.operator_basis)
    idx = 0
    for op1, op2 in zip(b1.operator_basis, b2.operator_basis):
        print(f"comparing b_{idx}")
        idx += 1
        print("first basis:\n", op1)
        print("second basis:\n", op2)

        assert check_operator_equality(op1, op2)

    assert np.allclose(b1.gram, b2.gram), f"{b1.gram}!={b2.gram}"
    assert np.allclose(b1.gram_inv, b2.gram_inv), f"{b1.gram_inv}!={b2.gram_inv}"
    assert np.allclose(b1.errors, b2.errors), f"{b1.errors}!={b2.errors}"
    assert np.allclose(
        b1.gen_matrix, b2.gen_matrix
    ), f"{b1.gen_matrix} != {b2.gen_matrix}"


def test_singular_basis_operator():
    h = SY_TOTAL
    basis1 = OperatorBasis(
        (K0_REFERENCE, HAMILTONIAN_REFERENCE, K0_REFERENCE - HAMILTONIAN_REFERENCE),
        h,
        REFERENCE_SP,
    )
    basis2 = HierarchicalOperatorBasis(K0_REFERENCE, h, 4, REFERENCE_SP)
    print("basis1.gram\n", basis1.gram)
    print("basis1.gram_inv\n", basis1.gram_inv)
    assert len(basis1.operator_basis) == 2
    assert len(basis2.operator_basis) == 2


def test_basis_operator():

    k_0 = K0_REFERENCE
    h = HAMILTONIAN_REFERENCE
    sp = REFERENCE_SP
    basis = OperatorBasis(tuple(BASIS_REFERENCE[:3]), h, sp)

    # Check that the projection is consistent:
    k_p = basis.project_onto(k_0)
    assert check_operator_equality(k_0, k_p), "projection should act trivially"
    phi_0 = basis.coefficient_expansion(k_0)
    assert all(abs(c) < 1e-10 for c in phi_0[1:]), (
        "the only non-vanishing coefficient for the expansion for the "
        "first element in the base should be the first one."
        f"Got {phi_0}"
    )

    delta_t = 1
    # Solution as a truncated Dyson's series:
    k_t_series = (
        BASIS_REFERENCE[0]
        + BASIS_REFERENCE[1] * delta_t
        + BASIS_REFERENCE[2] * delta_t**2 / 2.0
        + BASIS_REFERENCE[3] * delta_t**3 / 6.0
        + BASIS_REFERENCE[4] * delta_t**4 / 24.0
    )
    phi_t_series = basis.coefficient_expansion(k_t_series)

    # Using the evolution method of *evolve*
    phi_t_evolve, error = basis.evolve(delta_t, phi_0)
    k_t_proj = basis.operator_from_coefficients(phi_t_evolve)

    # Compute the norm of the difference for both solutions:
    # from coefficients
    delta_phi = phi_t_evolve - phi_t_series
    norm_delta_phi = (delta_phi @ (basis.gram @ delta_phi)) ** 0.5
    # from operators:
    delta_k = basis.project_onto(k_t_proj - k_t_series)
    norm_delta_k = sp(delta_k, delta_k) ** 0.5
    assert (
        norm_delta_k - norm_delta_phi
    ) < ALPSQUTIP_TOLERANCE, f"must coincide {norm_delta_k-norm_delta_phi}"
    assert (
        norm_delta_phi < error
    ), f"|Delta K|={norm_delta_phi} > {error}=estimated error"


def test_hierarchical_operator_basis():

    k_0 = K0_REFERENCE
    h = HAMILTONIAN_REFERENCE
    sp = REFERENCE_SP
    generic_basis = OperatorBasis(tuple(BASIS_REFERENCE[:4]), h, sp)
    basis = HierarchicalOperatorBasis(k_0, h, 4, sp)
    compare_basis(basis, generic_basis)

    # Check that the projection is consistent:
    k_p = basis.project_onto(k_0)
    assert check_operator_equality(k_0, k_p), "projection should act trivially"
    phi_0 = basis.coefficient_expansion(k_0)
    assert all(abs(c) < 1e-10 for c in phi_0[1:]), (
        "the only non-vanishing coefficient for the expansion for the "
        "first element in the base should be the first one."
        f"Got {phi_0}"
    )

    delta_t = 0.01
    # Solution as a truncated Dyson's series:
    k_t_series = (
        BASIS_REFERENCE[0]
        + BASIS_REFERENCE[1] * delta_t
        + BASIS_REFERENCE[2] * delta_t**2 / 2.0
        + BASIS_REFERENCE[3] * delta_t**3 / 6.0
        + BASIS_REFERENCE[4] * delta_t**4 / 24.0
    )
    phi_t_series = basis.coefficient_expansion(k_t_series)

    # Using the evolution method of *evolve*
    phi_t_evolve, error = basis.evolve(delta_t, phi_0)
    k_t_proj = basis.operator_from_coefficients(phi_t_evolve)

    # Compute the norm of the difference for both solutions:
    # from coefficients
    delta_phi = phi_t_evolve - phi_t_series
    norm_delta_phi = (delta_phi @ (basis.gram @ delta_phi)) ** 0.5
    # from operators:
    delta_k = basis.project_onto(k_t_proj - k_t_series)
    norm_delta_k = sp(delta_k, delta_k) ** 0.5
    assert (
        norm_delta_k - norm_delta_phi
    ) < ALPSQUTIP_TOLERANCE, f"must coincide {norm_delta_k-norm_delta_phi}"
    assert (
        norm_delta_phi < error
    ), f"|Delta K|={norm_delta_phi} > {error}=estimated error"
