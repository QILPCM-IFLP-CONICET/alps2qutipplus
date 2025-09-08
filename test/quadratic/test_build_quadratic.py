"""
Basic unit test.
"""

from test.helper import (
    OPERATOR_TYPE_CASES,
    TEST_CASES_STATES,
    check_equality,
    check_operator_equality,
)

import pytest

from alpsqutip.operators.basic import ProductOperator
from alpsqutip.operators.quadratic.build import (
    build_quadratic_form_from_operator,
    classify_terms,
)
from alpsqutip.operators.states import ProductDensityOperator

nonquadratic_test_cases = [
    "three body, hermitician",
    "three body, non hermitician",
    "qutip operator",
]


@pytest.mark.parametrize(
    ["operator_name", "state_name"],
    list(
        (operator_name, state_name)
        for operator_name in OPERATOR_TYPE_CASES
        for state_name in TEST_CASES_STATES
    ),
)
def test_build_quadratic(operator_name, state_name):
    """
    Test the function build_quadratic_hermitician.
    No assumptions on the hermiticity of the operator
    are done.
    """
    state = TEST_CASES_STATES[state_name]
    operator = OPERATOR_TYPE_CASES[operator_name]
    if hasattr(state, "as_product_state"):
        state = state.as_product_state()
    if state is not None or not isinstance(state, ProductDensityOperator):
        return

    print(
        "\n *******\n\n convert : ",
        operator_name,
        "to quadratic form relative to",
        state_name,
    )
    print("quadratic form from", type(operator))
    qutip_operator = operator.to_qutip()
    quadratic_form = build_quadratic_form_from_operator(
        operator, simplify=False, sigma_ref=state
    )
    check_operator_equality(
        quadratic_form.to_qutip(), qutip_operator
    ), "qutip form does not match."
    assert quadratic_form.isherm == qutip_operator.isherm, (
        "operator and its conversion to qutip "
        "should have the same hermitician character."
    )

    linear_term = quadratic_form.linear_term
    offset = quadratic_form.offset
    if state is None:
        original_expectation_value = operator.tr()
        converted_expectation_value = quadratic_form.tr()
        linear_term_exp_value = linear_term.tr() if linear_term is not None else 0
        offset_exp_value = offset.tr() if offset is not None else 0
    else:
        original_expectation_value = state.expect(operator)
        converted_expectation_value = state.expect(quadratic_form)
        linear_term_exp_value = (
            state.expect(linear_term) if linear_term is not None else 0
        )
        offset_exp_value = state.expect(offset) if offset is not None else 0

    assert check_equality(
        original_expectation_value, converted_expectation_value
    ), "the expectation value should be preserved"
    assert check_equality(
        original_expectation_value, linear_term_exp_value
    ), "the expectation value of the original operator and the linear term must coincide"
    assert check_equality(
        offset_exp_value, 0
    ), "the expectation value of offset must be zero."

    for basis_elem in quadratic_form.basis:
        if state is None:
            assert check_equality(basis_elem.tr(), 0.0), "trace must be zero"
        else:
            assert check_equality(
                state.expect(basis_elem), 0.0
            ), "expectation values must be zero"
        assert basis_elem.isherm, "basis elements must be hermitician"


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
