"""
Basic unit test.
"""

import numpy as np
from qutip import jmat, qeye, tensor

from alpsqutip.operators import ProductOperator, QutipOperator
from alpsqutip.qutip_tools.tools import (
    data_get_type,
    data_is_diagonal,
    data_is_scalar,
    data_is_zero,
    factorize_qutip_operator,
)

from .helper import (
    CHAIN_SIZE,
    check_operator_equality,
    operator_type_cases,
    sites,
    sx_A as local_sx_A,
    sy_B,
    sz_C,
)

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


sx_A = ProductOperator({local_sx_A.site: local_sx_A.operator}, 1.0, local_sx_A.system)
sx_A2 = sx_A * sx_A
sx_Asy_B = sx_A * sy_B
sx_AsyB_times_2 = 2 * sx_Asy_B
opglobal = sz_C + sx_AsyB_times_2

id_2 = qeye(2)
id_3 = qeye(3)
sx, sy, sz = jmat(0.5)
lx, ly, lz = jmat(1.0)

QUTIP_TEST_CASES = {
    "product_scalar": {
        "operator": 3 * tensor(id_2, id_3),
        "diagonal": True,
        "scalar": True,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_diagonal": {
        "operator": tensor(sz, lz),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_non_diagonal": {
        "operator": tensor(sx, lx),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_zero": {
        "operator": 0 * tensor(id_2, id_3),
        "diagonal": True,
        "scalar": True,
        "zero": True,
        "type": np.dtype("complex128"),
    },
    "scalar": {
        "operator": 3 * id_2,
        "diagonal": True,
        "scalar": True,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal": {
        "operator": sz + 0.5 * id_2,
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "non_diagonal": {
        "operator": lx,
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "zero": {
        "operator": 0 * id_2,
        "diagonal": True,
        "scalar": True,
        "zero": True,
        "type": np.dtype("complex128"),
    },
    "complex": {
        "operator": sy,
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "dense": {
        "operator": sx.expm(),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal dense": {
        "operator": sz.expm(),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "tensor dense": {
        "operator": tensor(sx, lx).expm(),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "tensor diagonal dense": {
        "operator": tensor(sz, sz).expm(),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal dense zero": {
        "operator": (-10000 * (id_2 + sz)).expm(),
        "diagonal": True,
        "scalar": True,
        "zero": True,
        "type": np.dtype("complex128"),
    },
}


def test_qutip_properties():

    for case, data in QUTIP_TEST_CASES.items():
        print("testing ", case)
        operator_data = data["operator"].data
        assert data["diagonal"] == data_is_diagonal(operator_data)
        assert data["scalar"] == data_is_scalar(operator_data)
        assert data["zero"] == data_is_zero(operator_data)
        assert data["type"] is data_get_type(operator_data)


def test_factorize_qutip_operators():
    """
    test decomposition of qutip operators
    as sums of product operators
    """
    for name, operator_case in operator_type_cases.items():
        print("decomposing ", name)
        acts_over = operator_case.acts_over()
        if acts_over:
            operator_case = operator_case.partial_trace(acts_over)
        qutip_operator = operator_case.to_qutip()
        terms = factorize_qutip_operator(qutip_operator)
        reconstructed = sum(tensor(*t) for t in terms)
        assert check_operator_equality(
            qutip_operator, reconstructed
        ), "reconstruction does not match with the original."


def test_qutip_operators():
    """Test for the qutip representation"""

    sx_A_qt = sx_A.to_qutip_operator()
    sx_A2_qt = sx_A_qt * sx_A_qt

    syB_qt = sy_B.to_qutip_operator()
    szC_qt = sz_C.to_qutip_operator()
    sx_AsyB_qt = sx_A_qt * syB_qt
    sx_AsyB_times_2_qt = 2 * sx_AsyB_qt
    opglobal_qt = szC_qt + sx_AsyB_times_2_qt

    subsystems = [
        [sites[0]],
        [sites[1]],
        [sites[0], [sites[1]]],
        [sites[1], [sites[2]]],
    ]

    for subsystem in subsystems:
        assert (sx_A_qt).partial_trace(subsystem).tr() == 0.0
        expect_value = 0.5 * 2 ** (CHAIN_SIZE - 1)
        assert (sx_A2_qt).partial_trace(subsystem).tr() == expect_value
        assert (sx_AsyB_qt * sx_A_qt * syB_qt).partial_trace(
            subsystem
        ).tr() == 0.25 * 2 ** (CHAIN_SIZE - 2)
        assert (opglobal_qt * sx_A_qt * syB_qt).partial_trace(
            subsystem
        ).tr() == 0.5 * 2 ** (CHAIN_SIZE - 2)
        assert (szC_qt * szC_qt).partial_trace(subsystem).tr() == 0.5 * 2 ** (
            CHAIN_SIZE - 1
        )
        assert (sx_AsyB_times_2_qt * sx_AsyB_times_2_qt).partial_trace(
            subsystem
        ).tr() == 2 ** (CHAIN_SIZE - 2)
        expected_value = 2 ** (CHAIN_SIZE - 2) * 2
        assert (opglobal_qt * opglobal_qt).partial_trace(
            subsystem
        ).tr() == expected_value
        assert (opglobal_qt * sx_A_qt).partial_trace(subsystem).tr() == 0.0
        assert (opglobal_qt * opglobal).partial_trace(subsystem).tr() == 2 ** (
            CHAIN_SIZE - 2
        ) * 2
        assert (opglobal * opglobal_qt).partial_trace(subsystem).tr() == 2 ** (
            CHAIN_SIZE - 2
        ) * 2

    # Tests for QutipOperators defined without a system
    detached_qutip_operator = QutipOperator(sx_AsyB_times_2_qt.operator)
    expected_value = 0.5**2 * 2 ** (CHAIN_SIZE - 2)
    assert ((sx_AsyB_times_2_qt.operator) ** 2).tr() == expected_value
    expected_value = 0.5**2 * 2 ** (CHAIN_SIZE - 2)
    assert (detached_qutip_operator * detached_qutip_operator).tr() == expected_value

    detached_qutip_operator = QutipOperator(
        sx_AsyB_times_2_qt.operator, names={s: i for i, s in enumerate(sites)}
    )
    assert (detached_qutip_operator * detached_qutip_operator).partial_trace(
        sites[0]
    ).tr() == 0.5**2 * 2 ** (CHAIN_SIZE - 2)
