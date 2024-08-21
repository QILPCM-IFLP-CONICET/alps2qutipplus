"""
Basic unit test.
"""

import numpy as np

from alpsqutip.operators import ProductOperator

from .helper import (
    full_test_cases,
    observable_cases,
    operator_type_cases,
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
        qobj = operator.to_qutip()
        data = qobj.data
        is_diagonal = (data.toarray() == np.diag(data.diagonal())).all()
        assert is_diagonal == operator.isdiagonal
