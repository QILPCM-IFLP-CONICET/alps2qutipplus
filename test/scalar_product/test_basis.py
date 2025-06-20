from test.helper import HAMILTONIAN, SX_A, SX_TOTAL, check_operator_equality

from alpsqutip.operators.functions import commutator
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.scalarprod.basis import OperatorBasis
from alpsqutip.scalarprod.build import fetch_covar_scalar_product
from alpsqutip.settings import ALPSQUTIP_TOLERANCE


def test_basis_operator():

    k_0 = SX_A
    h = HAMILTONIAN + SX_TOTAL
    comm1 = commutator(h * 1j, k_0)
    comm2 = commutator(h * 1j, comm1)
    comm3 = commutator(h * 1j, comm2)
    comm4 = commutator(h * 1j, comm3)
    sigma = GibbsProductDensityOperator(k_0)
    sp = fetch_covar_scalar_product(sigma)
    basis = OperatorBasis((k_0, comm1, comm2, comm3, h), h, sp)

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
        k_0
        - comm1 * delta_t
        + comm2 * delta_t**2 / 2.0
        - comm3 * delta_t**3 / 6.0
        + comm4 * delta_t**4 / 24.0
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
