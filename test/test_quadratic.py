"""
Basic unit test.
"""

import pytest

from alpsqutip.operators.basic import ProductOperator
from alpsqutip.operators.quadratic.build import (
    build_quadratic_form_from_operator,
    classify_terms,
)

from .helper import (
    OPERATOR_TYPE_CASES,
    TEST_CASES_STATES,
    check_equality,
    check_operator_equality,
)

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
        "operator and its conversion to qutip "
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

    operator = OPERATOR_TYPE_CASES[name]
    print("\n *******\n\n name: ", name)
    print("quadratic form. Forcing hermitician", type(operator))

    quadratic_form = build_quadratic_form_from_operator(operator, True, True)
    qutip_operator = self_adjoint_part(operator.to_qutip())

    check_operator_equality(quadratic_form.to_qutip(), qutip_operator)
    assert quadratic_form.isherm, "quadratic form must be hermitician"


@pytest.mark.parametrize(
    ["operator_name", "state_name"],
    list(
        (
            name,
            state,
        )
        for name in OPERATOR_TYPE_CASES
        for state in TEST_CASES_STATES
    ),
)
def test_classify_terms(operator_name, state_name):
    print("classifying terms from ", operator_name, "relative to", state_name)

    operator = OPERATOR_TYPE_CASES[operator_name]
    state = TEST_CASES_STATES[state_name]
    if state is not None:
        if hasattr(state, "to_product_state"):
            state = state.to_product_state()
        if not hasattr(state, "sites_op"):
            return

    quadratic_dict, linear, rest = classify_terms(operator, state)
    assert isinstance(quadratic_dict, dict)
    for block, terms in quadratic_dict.items():
        assert all(
            isinstance(term, ProductOperator) for term in terms
        ), "all the terms should be product operators."
        assert len(block) == 2

    if state is None:
        expect_val, linear_expect_val = operator.tr(), sum(term.tr() for term in linear)

        assert check_equality(
            expect_val, linear_expect_val
        ), f"the trace of {operator_name} must must be equal to the sum of the traces of the linear terms"

        all(
            check_equality(term.tr(), 0.0) for term in linear
        ), f"For {state_name} state, the remainder must be zero trace."

        assert all(
            check_equality(term.tr(), 0.0)
            for terms_block in quadratic_dict.values()
            for term in terms_block
        ), f"For {state_name} state, all the quadratic terms must be zero trace."

        assert all(
            check_equality(term.tr(), 0.0) for term in rest
        ), f"For {state_name} state, the remainder must be zero trace."

    else:
        expect_val, linear_expect_val = state.expect(operator), sum(
            state.expect(linear)
        )
        assert check_equality(
            expect_val, linear_expect_val
        ), f"the expectation value of {operator_name} relative to {state_name} must must be equal to the sum of the expectation values of the linear terms."

        assert all(
            check_equality(state.expect(term), 0.0)
            for terms_block in quadratic_dict.values()
            for term in terms_block
        ), f"For {state_name} state, all the quadratic terms must be zero expectation value. "
        assert all(
            check_equality(state.expect(term), 0.0) for term in rest
        ), f"For {state_name} state, the remainder must be zero expectation value. "
