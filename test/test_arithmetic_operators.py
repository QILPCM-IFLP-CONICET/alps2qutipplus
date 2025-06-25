"""
Basic unit test.
"""

import numpy as np
import pytest

from .helper import (
    OPERATOR_TYPE_CASES,
    TEST_CASES_STATES,
    check_operator_equality,
)

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)

OPERATORS_AND_STATE_CASES = OPERATOR_TYPE_CASES.copy()
OPERATORS_AND_STATE_CASES.update(TEST_CASES_STATES)

OPERATOR_TYPE_CASES_QUTIP = {
    key: operator.to_qutip() for key, operator in OPERATORS_AND_STATE_CASES.items()
}


@pytest.mark.parametrize(
    ["key1", "test_operator1", "key2", "test_operator2"],
    [
        (key1, test_operator1, key2, test_operator2)
        for key1, test_operator1 in OPERATORS_AND_STATE_CASES.items()
        for key2, test_operator2 in OPERATORS_AND_STATE_CASES.items()
    ],
)
def test_arithmetic_operators(key1, test_operator1, key2, test_operator2):
    """
    Test consistency of arithmetic expressions
    """
    op1_qutip = OPERATOR_TYPE_CASES_QUTIP[key1]
    if key1 == key2:
        print("comparing unary operations for ", key1)
        print("   # negate")
        result = -test_operator1
        check_operator_equality(result.to_qutip(), -op1_qutip)
        print("   # dag")
        result = test_operator1.dag()
        check_operator_equality(result.to_qutip(), op1_qutip.dag())

    print("add ", key1, " and ", key2)
    op2_qutip = OPERATOR_TYPE_CASES_QUTIP[key2]
    print(type(test_operator1), "+", type(test_operator2))
    result = test_operator1 + test_operator2

    check_operator_equality(result.to_qutip(), (op1_qutip + op2_qutip))

    print("product of ", key1, " and ", key2)
    result = test_operator1 * test_operator2

    check_operator_equality(result.to_qutip(), (op1_qutip * op2_qutip))


@pytest.mark.parametrize(
    ["key1", "test_operator1", "key2", "value"],
    [
        (key1, test_operator1, key2, value)
        for key1, test_operator1 in OPERATORS_AND_STATE_CASES.items()
        for key2, value in [
            ("int positive", 2),
            ("int negative", -3),
            ("int zero", 0),
            ("float zero", 0.0),
            ("float positive", 2.0),
            ("float negative", -2.0),
            ("imaginary", 2.0j),
            ("complex zero", 0.0j),
            ("complex negative", 2.0 + 2.0j),
            ("float64 positive", np.float64(2.0)),
            ("float64 negative", np.float64(-2.0)),
            ("imaginary 128", np.complex128(2.0j)),
            ("zero complex 128", np.complex128(0.0j)),
            ("complex128", np.complex128(2.0 + 2.0j)),
        ]
    ],
)
def test_arithmetic_operators_with_numbers(key1, test_operator1, key2, value):
    """
    Test consistency of arithmetic expressions
    """
    op1_qutip = OPERATOR_TYPE_CASES_QUTIP[key1]

    print("add ", key1, " and ", key2)
    print(type(test_operator1), "+", type(value))
    result = test_operator1 + value
    result_bw = value + test_operator1
    sum_qutip = op1_qutip + value
    check_operator_equality(result.to_qutip(), sum_qutip)
    check_operator_equality(result_bw.to_qutip(), sum_qutip)

    print("product of ", key1, " and ", key2)
    result = test_operator1 * value
    result_bw = value * test_operator1
    qutip_prod = op1_qutip * value

    check_operator_equality(result.to_qutip(), qutip_prod)
    check_operator_equality(result_bw.to_qutip(), qutip_prod)

    result = result.simplify()
    result_bw = result_bw.simplify()

    check_operator_equality(result.to_qutip(), qutip_prod)
    check_operator_equality(result_bw.to_qutip(), qutip_prod)
