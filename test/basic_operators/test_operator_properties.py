"""
Basic unit test.
"""

from test.helper import (
    FULL_TEST_CASES,
    OBSERVABLE_CASES,
    SX_A as LOCAL_SX_A,
    SY_B,
    SZ_C,
    SZ_TOTAL,
)

import numpy as np
import pytest
import qutip

from alpsqutip.operators import ProductOperator
from alpsqutip.operators.basic import empty_op, is_diagonal_op, is_scalar_op

SX_A = ProductOperator({LOCAL_SX_A.site: LOCAL_SX_A.operator}, 1.0, LOCAL_SX_A.system)
SX_A2 = SX_A * SX_A
SX_ASY_B = SX_A * SY_B
SX_ASYB_TIMES_2 = 2 * SX_ASY_B
OPGLOBAL = SZ_C + SX_ASYB_TIMES_2


@pytest.mark.parametrize(["name", "value"], list(FULL_TEST_CASES.items()))
def test_empty_op(name, value):
    """
    test for the function that checks if the operator
    is equivalent to 0.
    """
    if value is None:
        return
    value_qutip = value.to_qutip()
    print(name, "of type ", type(value), type(value_qutip))
    assert empty_op(value_qutip) == empty_op(value)


def test_is_scalar_or_diagonal_operator():
    """
    test for the function that checks if the operator
    is equivalent to 0.
    """

    test_cases = {
        "zero": (0 * qutip.qeye(4), True, True),
        "identity": (qutip.qeye(4), True, True),
        "dense zero": ((-100 * qutip.qeye(4)).expm(), True, True),
        "dense one times 0": ((0 * qutip.qeye(4)).expm() * 0, True, True),
        "dense one": ((0 * qutip.qeye(4)).expm(), True, True),
        "dense scalar": (qutip.qeye(4).expm(), True, True),
        "projection": (qutip.projection(4, 2, 2), True, False),
        "coherence": (qutip.projection(4, 1, 2), False, False),
        "sigmax": (qutip.sigmax(), False, False),
        "sigmaz": (qutip.sigmaz(), True, False),
        "dense sigmax": (qutip.sigmax().expm(), False, False),
        "dense sigmaz": (qutip.sigmaz().expm(), True, False),
        "dense z projection": ((100 * qutip.sigmaz()).expm(), True, False),
        "dense x projection": ((100 * qutip.sigmax()).expm(), False, False),
    }

    for name, test in test_cases.items():
        print(name, "of type ", type(test[0]), "(", type(test[0].data), ")")
        assert is_diagonal_op(test[0]) == test[1]
        assert is_scalar_op(test[0]) == test[2]


@pytest.mark.parametrize(["name", "value"], list(FULL_TEST_CASES.items()))
def test_trace(name, value):
    """
    test for the function that checks if the operator
    is equivalent to 0.
    """

    if value is None:
        return
    # TODO: check the trace of quadratic operators.
    if name in (
        "hermitician quadratic operator",
        "non hermitician quadratic operator",
    ):
        return
    value_tr = value.tr()
    value_qtip_tr = value.to_qutip().tr()
    print(name, "of type ", type(value), "->", value.tr(), value_qtip_tr)
    print(" trace match?", value_tr == value_qtip_tr)
    assert abs(value_tr - value_qtip_tr) < 1.0e-9


@pytest.mark.parametrize(["key", "observable"], list(OBSERVABLE_CASES.items()))
def test_isherm_operator(key, observable):
    """
    Check if hermiticity is correctly determined
    """

    def do_test_case(name, observable):
        if isinstance(observable, list):
            for op_case in observable:
                do_test_case(name, op_case)
            return

        assert observable.isherm, f"{key} is not hermitician?"

        ham = OBSERVABLE_CASES["hamiltonian"]
        print("***addition***")
        assert (ham + 1.0).isherm
        assert (ham + SZ_TOTAL).isherm
        print("***scalar multiplication***")
        assert (2.0 * ham).isherm
        print("***scalar multiplication for a OneBody Operator")
        assert (2.0 * SZ_TOTAL).isherm
        assert (ham * 2.0).isherm
        assert (SZ_TOTAL * 2.0).isherm
        assert (SZ_TOTAL.expm()).isherm
        assert (ham**3).isherm

    do_test_case(key, observable)


@pytest.mark.parametrize(["name", "operator"], list(FULL_TEST_CASES.items()))
def test_isdiagonal(name, operator):
    """test the isdiag property"""

    print("checking diagonality in ", name, type(operator))
    qobj = operator.to_qutip()
    data_qt = qobj.data
    print("data_qt type:", type(data_qt))

    if hasattr(data_qt, "to_ndarray"):
        full_array = data_qt.to_ndarray()
    if hasattr(data_qt, "toarray"):
        full_array = data_qt.toarray()
    if hasattr(data_qt, "to_array"):
        full_array = data_qt.to_array()

    print(full_array)
    qt_is_diagonal = not (full_array - np.diag(full_array.diagonal())).any()
    assert qt_is_diagonal == operator.isdiagonal


@pytest.mark.parametrize(["key", "operator"], list(FULL_TEST_CASES.items()))
def test_norm(key, operator):
    """test the isdiag property"""

    print("checking norms for", key, type(operator))
    q_op = operator.to_qutip_operator()
    for ord in ["fro", "nuc", 2]:
        print("   checking for ord:", ord)
        qutip_value = q_op.norm(ord)
        value = operator.norm(ord)
        assert (
            abs(value - qutip_value) < 1e-9
        ), f"     {value}!={qutip_value} factor ({value / qutip_value})."
