"""
Basic unit test.
"""

from alpsqutip.model import build_spin_chain
from alpsqutip.operator_functions import (
    hermitian_and_antihermitian_parts,
    simplify_sum_operator,
)
from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
    SumOperator,
)
from alpsqutip.operators.quadratic import (
    QuadraticFormOperator,
    build_quadratic_form_from_operator,
    simplify_quadratic_form,
)

from .helper import check_operator_equality, operator_type_cases

CHAIN_SIZE = 6

# system_descriptor = build_spin_chain(CHAIN_SIZE)
# sites = tuple(s for s in system_descriptor.sites.keys())

# sz_total = system_descriptor.global_operator("Sz")
# hamiltonian = system_descriptor.global_operator("Hamiltonian")


nonquadratic_test_cases = [
    "three body, hermitician",
    "three body, non hermitician",
    "qutip operator",
]


def test_first():
    for name, operator in operator_type_cases.items():
        print(name)
        qutip_op = operator.to_qutip()
        real_part, imag_part = hermitian_and_antihermitian_parts(operator)
        imag_part = simplify_sum_operator(imag_part)
        if qutip_op.isherm:
            if bool(imag_part):
                print("imaginary part:", type(imag_part), imag_part.simplify())
                print([type(t) for t in imag_part.terms])
            assert not bool(
                imag_part
            ), f"<<{name}>> has marked as hermitician, but has an imaginary part."

        simplified = simplify_sum_operator(operator)

        check_operator_equality(qutip_op, simplified.to_qutip())
        assert simplified.isherm == qutip_op.isherm


def test_quadratic():
    def self_adjoint_part(op_g):
        return 0.5 * (op_g + op_g.dag())

    test_cases = operator_type_cases
    skip_cases = []

    for name, operator in test_cases.items():
        print("\n *******\n\n name: ", name)
        print("quadratic form. Force hermitician, no simplify", type(operator))
        try:
            quadratic_form = build_quadratic_form_from_operator(
                operator, None, False, True
            )
        except ValueError:
            skip_cases.append(name)
            continue

        qutip_operator = operator.to_qutip()
        print(operator)

        check_operator_equality(
            quadratic_form.to_qutip(), self_adjoint_part(qutip_operator)
        )
        print("quadratic form. Force hermitician, simplify")
        quadratic_form = build_quadratic_form_from_operator(operator, None, True, True)
        check_operator_equality(
            quadratic_form.to_qutip(), self_adjoint_part(qutip_operator)
        )

        print("quadratic form for the general case. No simplify")
        quadratic_form = build_quadratic_form_from_operator(
            operator, None, False, False
        )
        check_operator_equality(quadratic_form.to_qutip(), qutip_operator)

        print("quadratic form for the general case. Simplify")
        quadratic_form = build_quadratic_form_from_operator(operator, None, True, False)
        print(
            quadratic_form.weights,
            "\n----\n".join(repr(term) for term in quadratic_form.terms),
        )

        check_operator_equality(quadratic_form.to_qutip(), qutip_operator)

        print("   weights:", quadratic_form.weights)
        print("Simplify the quadratic form")
        quadratic_form = simplify_quadratic_form(quadratic_form, False)
        check_operator_equality(
            quadratic_form.to_qutip(), self_adjoint_part(qutip_operator)
        )
        print("  weights:", quadratic_form.weights)
        print("Simplify the quadratic form again")
        quadratic_form = simplify_quadratic_form(quadratic_form, False)
        check_operator_equality(
            quadratic_form.to_qutip(), self_adjoint_part(qutip_operator)
        )
        print("check hermiticity", quadratic_form.weights)
        assert quadratic_form.isherm == qutip_operator.isherm

    assert all(name in nonquadratic_test_cases for name in skip_cases)
