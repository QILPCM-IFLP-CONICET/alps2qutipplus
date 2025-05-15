"""
Basic unit test.
"""

import pytest

from alpsqutip.operators.quadratic import build_quadratic_form_from_operator

from .helper import OPERATOR_TYPE_CASES, check_operator_equality

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


@pytest.mark.parametrize(["name"], list((name,) for name in OPERATOR_TYPE_CASES))
def test_simplify_quadratic_form(name):
    """
    Try to convert all the test cases into
    quadratic forms, and check if simplification
    works in all the cases.
    """
    operator = OPERATOR_TYPE_CASES[name]
    print("\n *******\n\n name: ", name)
    quadratic_form = build_quadratic_form_from_operator(operator, simplify=False)
    print(type(operator), " produced a ", type(quadratic_form))
    qutip_operator = operator.to_qutip().tidyup()
    simplified = quadratic_form.simplify()
    assert (
        simplified is simplified.simplify()
    ), "simplify of an already simpliifed object must be the same."
    check_operator_equality(qutip_operator, simplified.to_qutip())
    assert (
        quadratic_form.isherm == simplified.isherm
    ), "quadratic form changed its hermitician character after simplification."
    assert (
        qutip_operator.isherm == quadratic_form.isherm
    ), "qutip operator and the quadratic form have different hermitician character."


@pytest.mark.parametrize(["name"], list((name,) for name in OPERATOR_TYPE_CASES))
def test_build_quadratic(name):
    """
    Test the function build_quadratic_hermitician.
    No assumptions on the hermiticity of the operator
    are done.
    """
    operator = OPERATOR_TYPE_CASES[name]
    print("\n *******\n\n name: ", name)
    print("quadratic form from", type(operator))
    quadratic_form = build_quadratic_form_from_operator(operator, simplify=False)
    qutip_operator = operator.to_qutip()

    check_operator_equality(quadratic_form.to_qutip(), qutip_operator)
    assert quadratic_form.isherm == qutip_operator.isherm, (
        "operator and its convertion to qutip "
        "should have the same hermitician character."
    )


@pytest.mark.parametrize(["name"], list((name,) for name in OPERATOR_TYPE_CASES))
def test_build_quadratic_hermitician(name):
    """
    Test the function build_quadratic_hermitician
    if is assumed that the original operator is hermitician.
    """

    def self_adjoint_part(op_g):
        return 0.5 * (op_g + op_g.dag())

    skip_cases = []
    operator = OPERATOR_TYPE_CASES[name]
    if name in skip_cases:
        return
    print("\n *******\n\n name: ", name)
    print("quadratic form. Forcing hermitician", type(operator))

    quadratic_form = build_quadratic_form_from_operator(operator, True, True)
    qutip_operator = self_adjoint_part(operator.to_qutip())

    check_operator_equality(quadratic_form.to_qutip(), qutip_operator)
    assert quadratic_form.isherm, "quadratic form must be hermitician"
