"""
Basic unit test.
"""

from alpsqutip.operators.quadratic import build_quadratic_form_from_operator

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


def test_simplify_quadratic_form():
    """
    Try to convert all the test cases into
    quadratic forms, and check if simplification
    works in all the cases.
    """
    test_cases = operator_type_cases
    for name, operator in test_cases.items():
        print("\n *******\n\n name: ", name)
        print("quadratic form", type(operator))
        quadratic_form = build_quadratic_form_from_operator(operator, simplify=False)
        qutip_operator = operator.to_qutip()
        simplified = quadratic_form.simplify()
        check_operator_equality(qutip_operator, simplified.to_qutip())
        assert qutip_operator.isherm == quadratic_form.isherm
        assert quadratic_form.isherm == simplified.isherm
        assert simplified is simplified.simplify()


def test_build_quadratic():
    """
    Test the function build_quadratic_hermitician.
    No assumptions on the hermiticity of the operator
    are done.
    """
    test_cases = operator_type_cases

    for name, operator in test_cases.items():
        print("\n *******\n\n name: ", name)
        print("quadratic form", type(operator))
        quadratic_form = build_quadratic_form_from_operator(operator, simplify=False)
        qutip_operator = operator.to_qutip()

        check_operator_equality(quadratic_form.to_qutip(), qutip_operator)
        assert quadratic_form.isherm == qutip_operator.isherm


def test_build_quadratic_hermitician():
    """
    Test the function build_quadratic_hermitician
    if is assumed that the original operator is hermitician.
    """

    def self_adjoint_part(op_g):
        return 0.5 * (op_g + op_g.dag())

    test_cases = operator_type_cases
    skip_cases = []
    for name, operator in test_cases.items():
        if name in skip_cases:
            continue
        print("\n *******\n\n name: ", name)
        print("quadratic form. Force hermitician", type(operator))
        try:
            quadratic_form = build_quadratic_form_from_operator(operator, True, False)
        except ValueError:
            skip_cases.append(name)
            continue

        qutip_operator = self_adjoint_part(operator.to_qutip())

        check_operator_equality(quadratic_form.to_qutip(), qutip_operator)
        assert quadratic_form.isherm
