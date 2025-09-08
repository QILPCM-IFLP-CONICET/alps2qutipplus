"""
Basic unit test.
"""

import pytest

from alpsqutip.operators.basic import ProductOperator
from alpsqutip.operators.quadratic.build import (
    build_quadratic_form_from_operator,
    classify_terms,
)

from test.helper import (
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

QUADRATIC_OPERATORS = {
    f"Quadratic form from name": build_quadratic_form_from_operator(operator, simplify=False)
    for name, operator in OPERATOR_TYPE_CASES.items()
}


@pytest.mark.parametrize(["name"], list((name,) for name in QUADRATIC_OPERATORS))
def test_simplify_quadratic_form(name):
    """
    Try to convert all the test cases into
    quadratic forms, and check if simplification
    works in all the cases.
    """
    operator = QUADRATIC_OPERATORS[name]
    print("\n *******\n\n name: ", name)
    qutip_operator = operator.to_qutip().tidyup()
    simplified = operator.simplify()
    assert (
        simplified is simplified.simplify()
    ), "simplify of an already simpliifed object must be the same."
    check_operator_equality(qutip_operator, simplified.to_qutip())
    assert (
        operator.isherm == simplified.isherm
    ), "quadratic form changed its hermitician character after simplification."
    assert (
        qutip_operator.isherm == operator.isherm
    ), "qutip operator and the quadratic form have different hermitician character."



@pytest.mark.parametrize(["name"], list((name,) for name in QUADRATIC_OPERATORS))
def test_flat_quadratic(name):
    """Test flat"""

    operator = QUADRATIC_OPERATORS[name]
    print("\n *******\n\n name: ", name)
    qutip_operator = operator.to_qutip().tidyup()
    flat_operator = operator.flat()
    # 
    if hasattr(flat_operator, "terms"):
        assert not any(hasattr(term, "terms") for term in flat_operator.terms), f"For {name}, flat() return a nested sum operator."

    check_operator_equality(qutip_operator, flat_operator.to_qutip())
    assert (
        operator.isherm == flat_operator.isherm
    ), "quadratic form changed its hermitician character after simplification."
    assert (
        qutip_operator.isherm == operator.isherm
    ), "qutip operator and the quadratic form have different hermitician character."

    
