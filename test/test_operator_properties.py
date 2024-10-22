"""
Basic unit test.
"""

import numpy as np
import qutip

from alpsqutip.operators import ProductOperator
from alpsqutip.operators.basic import empty_op, is_diagonal_op, is_scalar_op

from .helper import (
    full_test_cases,
    observable_cases,
    sx_A as local_sx_A,
    sy_B,
    sz_C,
    sz_total,
)

sx_A = ProductOperator({local_sx_A.site: local_sx_A.operator}, 1.0, local_sx_A.system)
sx_A2 = sx_A * sx_A
sx_Asy_B = sx_A * sy_B
sx_AsyB_times_2 = 2 * sx_Asy_B
opglobal = sz_C + sx_AsyB_times_2


def test_empty_op():
    """
    test for the function that checks if the operator
    is equivalent to 0.
    """
    for name, value in full_test_cases.items():
        if value is None:
            continue
        print(name, "of type ", type(value))
        assert not empty_op(value.to_qutip())


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


def test_trace():
    """
    test for the function that checks if the operator
    is equivalent to 0.
    """
    for name, value in full_test_cases.items():
        if value is None:
            continue
        # TODO: check the trace of quadratic operators.
        if name in (
            "hermitician quadratic operator",
            "non hermitician quadratic operator",
        ):
            continue
        value_tr = value.tr()
        value_qtip_tr = value.to_qutip().tr()
        print(name, "of type ", type(value), "->", value.tr(), value_qtip_tr)
        print(" trace match?", value_tr == value_qtip_tr)
        assert abs(value_tr - value_qtip_tr) < 1.0e-9


def test_isherm_operator():
    """
    Check if hermiticity is correctly determined
    """

    def do_test_case(name, observable):
        if isinstance(observable, list):
            for op_case in observable:
                do_test_case(name, op_case)
            return

        assert observable.isherm, f"{key} is not hermitician?"

        ham = observable_cases["hamiltonian"]
        print("***addition***")
        assert (ham + 1.0).isherm
        assert (ham + sz_total).isherm
        print("***scalar multiplication***")
        assert (2.0 * ham).isherm
        print("***scalar multiplication for a OneBody Operator")
        assert (2.0 * sz_total).isherm
        assert (ham * 2.0).isherm
        assert (sz_total * 2.0).isherm
        assert (sz_total.expm()).isherm
        assert (ham**3).isherm

    for key, observable in observable_cases.items():
        do_test_case(key, observable)


def test_isdiagonal():
    """test the isdiag property"""
    for key, operator in full_test_cases.items():
        print("checking diagonality in ", key, type(operator))
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
