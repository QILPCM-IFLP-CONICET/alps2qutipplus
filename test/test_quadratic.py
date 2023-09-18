"""
Basic unit test.
"""


from alpsqutip.model import build_spin_chain
from alpsqutip.operators import (LocalOperator, OneBodyOperator,
                                 ProductOperator, QutipOperator, SumOperator)
from alpsqutip.quadratic import QuadraticFormOperator, simplify_quadratic_form

from .helper import check_operator_equality

CHAIN_SIZE = 6

system_descriptor = build_spin_chain(CHAIN_SIZE)
sites = tuple(s for s in system_descriptor.sites.keys())

sz_total = system_descriptor.global_operator("Sz")
hamiltonian = system_descriptor.global_operator("Hamiltonian")


def test_build_hamiltonian():
    """build ham"""
    print(system_descriptor.operators["global_operators"].keys())
    assert sz_total is not None
    assert hamiltonian is not None
    hamiltonian_with_field = hamiltonian + 1.27 * sz_total

    ham_qf = QuadraticFormOperator.build_from_operator(hamiltonian_with_field)
    assert check_operator_equality(
        ham_qf.to_qutip(), hamiltonian_with_field.to_qutip())

    ham_qf_simp = simplify_quadratic_form(ham_qf)
    assert check_operator_equality(
        ham_qf_simp.to_qutip(), hamiltonian_with_field.to_qutip()
    )

    assert len(ham_qf_simp.terms) < len(ham_qf.terms)
    ham_qf = ham_qf_simp

    ham_qf_simp = simplify_quadratic_form(ham_qf)
    assert check_operator_equality(
        ham_qf_simp.to_qutip(), hamiltonian_with_field.to_qutip()
    )
    assert len(ham_qf_simp.terms) == len(ham_qf.terms)


def notest_meanfield():
    from alpsqutip.model import build_spin_chain
    from alpsqutip.quadratic import (
        QuadraticFormOperator, selfconsistent_meanfield_from_quadratic_form)
    system = build_spin_chain(10)
    sx_1 = system.site_operator("Sx", "1[0]")
    sx_2 = system.site_operator("Sx", "1[1]")
    sy_1 = system.site_operator("Sy", "1[0]")
    sy_2 = system.site_operator("Sy", "1[1]")

    hamiltonian = system.global_operator("Hamiltonian")
    qhamiltonian = QuadraticFormOperator.build_from_operator(hamiltonian)
    result = selfconsistent_meanfield_from_quadratic_form(qhamiltonian, 40)
    result.expect(sx_1*sx_2+sy_1*sy_2)
