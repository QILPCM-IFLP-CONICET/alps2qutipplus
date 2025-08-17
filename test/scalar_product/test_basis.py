from test.helper import (
    HAMILTONIAN,
    SX_A,
    SX_TOTAL,
    SY_TOTAL,
    SYSTEM,
    SZ_TOTAL,
    check_operator_equality,
)

import numpy as np
from numpy.linalg import inv

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


def check_basis_equivalence(basis1, basis2):
    """
    check if both basis are equivalent
    """
    if len(basis1.operator_basis) != len(basis2.operator_basis):
        print(
            "basis have different number of generators",
            len(basis1.operator_basis),
            "!=",
            len(basis2.operator_basis),
        )
        print(f"gram basis1    {type(basis1)}\n", basis1.gram)
        print("  eigenvalues", np.linalg.eigvalsh(basis1.gram))
        print(f"gram basis2 {type(basis2)}\n", basis2.gram)
        print("  eigenvalues", np.linalg.eigvalsh(basis2.gram))
        return False
    for pos, op1 in enumerate(basis1.operator_basis):
        error = basis2.project_onto(op1) - op1
        error_norm = abs(basis1.sp(error, error)) ** 0.5
        if error_norm > ALPSQUTIP_TOLERANCE:
            print(
                f"{pos}-th can not faitfully represented in basis2 for the tolerance. Error=",
                error_norm,
            )
            return False
    return True


def check_basis_consistency(basis):
    basis_size = len(basis.operator_basis)
    assert len(basis.gram) == basis_size
    assert len(basis.gram_inv) == basis_size
    assert len(basis.errors) == basis_size
    assert len(basis.gen_matrix) == basis_size

    assert np.allclose(inv(basis.gram), basis.gram_inv)
    if basis.generator is not None:
        check_gen_matrix(basis)
    else:
        assert all(coeff == 0 for coeff in basis.errors)
        assert all(coeff == 0 for row in basis.gen_matrix for coeff in row)


def check_gen_matrix(basis):
    """
    Check that the gen_matrix attribute
    has the right values
    """
    operators = basis.operator_basis
    sp = basis.sp
    n = len(operators)
    generator = basis.generator

    hij = np.zeros(
        (
            n,
            n,
        )
    )
    comm_norms = np.empty((n,), dtype=hij.dtype)
    for j, op1 in enumerate(operators):
        comm = commutator(op1, generator)
        comm_norms[j] = sp(comm, comm)
        for i, op2 in enumerate(operators):
            hij[i, j] = sp(op2, comm)

    assert np.allclose(
        basis.gram_inv @ basis.gram, np.eye(n)
    ), "gram_inv must be the inverse of gram."

    genij = basis.gram_inv @ hij
    assert np.allclose(
        basis.gen_matrix.round(8), genij.round(8)
    ), f"\n{basis.gen_matrix.round(8)}\n!=\n{hij.round(8)}"

    basis_errors_sq = basis.errors**2
    computed_errors_sq = comm_norms - np.array(
        [genij[:, j] @ hij[:, j] for j in range(n)]
    )

    assert np.allclose(
        basis_errors_sq, computed_errors_sq
    ), f"\n{basis_errors_sq.round(8)}\n!=\n{computed_errors_sq.round(8)}"


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
    basis = HierarchicalOperatorBasis(k_0, h, 3, sp)
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


def test_add_basis():

    k_0 = K0_REFERENCE
    h = HAMILTONIAN_REFERENCE
    sp = REFERENCE_SP
    test_operator = SX_TOTAL + SY_TOTAL
    full_norm = sp(test_operator, test_operator)

    generic_basis = OperatorBasis(tuple(BASIS_REFERENCE[:3]), h, sp)
    print("generic basis has ", len(generic_basis.operator_basis), "elements.")
    print("check gen generic_basis")
    check_basis_consistency(generic_basis)
    proj_op = generic_basis.project_onto(test_operator)
    prj_1_norm = sp(proj_op, proj_op)
    assert prj_1_norm < full_norm

    hierarchical_basis = HierarchicalOperatorBasis(k_0, h, 3, sp)
    print(
        "hierarchical basis has ", len(hierarchical_basis.operator_basis), "elements."
    )
    print("check gen hierarchical_basis")
    check_basis_consistency(hierarchical_basis)

    proj_op = hierarchical_basis.project_onto(test_operator)
    prj_2_norm = sp(proj_op, proj_op)
    assert prj_2_norm == prj_1_norm

    ### Extending from right

    print("extending the basis from the right")
    basis_extended1 = hierarchical_basis + (
        SX_TOTAL,
        SY_TOTAL,
        SZ_TOTAL,
    )
    print("Right extended basis has ", len(basis_extended1.operator_basis), "elements.")
    assert not check_basis_equivalence(
        basis_extended1, hierarchical_basis
    ), "the extended basis should not be equivalent to the original"

    print("check gen basis extended1")
    check_basis_consistency(basis_extended1)

    proj_op = basis_extended1.project_onto(test_operator)

    prj_ext1_norm = sp(proj_op, proj_op)
    assert full_norm == prj_ext1_norm, f"{full_norm} != {prj_ext1_norm}"

    ## Extending from left

    print("extending the basis from the left.")
    basis_extended2 = (
        SX_TOTAL,
        SY_TOTAL,
        SZ_TOTAL,
    ) + hierarchical_basis
    print("left extended basis has ", len(basis_extended2.operator_basis), "elements.")
    assert not check_basis_equivalence(basis_extended2, hierarchical_basis)
    print("check gen basis extended2")
    check_basis_consistency(basis_extended2)
    proj_op = basis_extended2.project_onto(test_operator)
    prj_ext2_norm = sp(proj_op, proj_op)
    assert abs(full_norm - prj_ext2_norm) < 1e-10, f"{full_norm} != {prj_ext2_norm}"
    assert check_basis_equivalence(
        basis_extended1, basis_extended2
    ), "both basis must be equivalent."


def test_hierarchical_basis_closed_algebra():
    # A Hierarchical basis with a generator that commutes with the seed:
    basis = HierarchicalOperatorBasis(HAMILTONIAN, HAMILTONIAN, 3, REFERENCE_SP)
    assert (
        len(basis.operator_basis) == 1
    ), "Generator commutes with the seed. Deep 0 expected."

    # A hierarchical basis with zero deep:
    basis = HierarchicalOperatorBasis(SX_TOTAL, SZ_TOTAL, 0, REFERENCE_SP)
    assert (
        len(basis.operator_basis) == 1
    ), "If deep is zero, the basis must contain just the seed."

    # A closed algebra with different deeps
    basis_a = HierarchicalOperatorBasis(SX_TOTAL, SZ_TOTAL, 1, REFERENCE_SP)
    basis_b = HierarchicalOperatorBasis(SX_TOTAL, SZ_TOTAL, 3, REFERENCE_SP)
    assert check_basis_equivalence(basis_a, basis_b), "closed algebras saturates deep."
    assert len(basis_b.operator_basis) == 2, f"{basis_b.operator_basis} should be 2."
    check_operator_equality(SY_TOTAL, basis_a.project_onto(SY_TOTAL))


def test_extend_basis():
    print("0. create a local sx basis")
    local_sx = [SYSTEM.site_operator("Sx", site) for site in SYSTEM.sites]
    print("1. build a hierarchical basis")
    basis1 = sum(
        HierarchicalOperatorBasis(op, HAMILTONIAN, 2, REFERENCE_SP) for op in local_sx
    )
    print("2. checking consistency of basis 1")
    check_basis_consistency(basis1)

    print("4. build a test element.")
    test_elem = commutator(HAMILTONIAN, local_sx[0] + local_sx[2]) * 1j

    print("   test_elem", test_elem.isherm, "\n", test_elem)
    print("   test_elem", [t.isherm for t in test_elem.terms])

    assert test_elem.isherm
    print("5. check that testlemen is generated by the basis.")
    assert check_operator_equality(
        test_elem, basis1.project_onto(test_elem), tolerance=1e-7
    )
    print("adding basis 1 with a linar combination")
    basis2 = basis1 + (test_elem,)
    print(" the result has", len(basis1.operator_basis), "elements")
    assert check_basis_equivalence(basis1, basis2)
