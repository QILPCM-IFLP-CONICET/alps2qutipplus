"""
Basic unit test.
"""

import numpy as np
import pytest
from qutip import jmat, qeye, tensor

from alpsqutip.operators import Operator, ProductOperator, QutipOperator, ScalarOperator
from alpsqutip.qutip_tools.tools import (
    data_get_type,
    data_is_diagonal,
    data_is_scalar,
    data_is_zero,
    decompose_qutip_operator,
)

from .helper import (
    CHAIN_SIZE,
    OPERATOR_TYPE_CASES,
    check_operator_equality,
    SITES,
    sx_A as LOCAL_SX_A,
    sy_B as SY_B,
    sz_C as SZ_C,
)

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


SX_A = ProductOperator({LOCAL_SX_A.site: LOCAL_SX_A.operator}, 1.0, LOCAL_SX_A.system)
SX_A2 = SX_A * SX_A
SX_A_SY_B = SX_A * SY_B
SX_A_SY_B_TIMES_2 = 2 * SX_A_SY_B
OP_GLOBAL = SZ_C + SX_A_SY_B_TIMES_2


SX_A_QT = SX_A.to_qutip_operator()
SX_A2_QT = SX_A_QT * SX_A_QT

SY_B_QT = SY_B.to_qutip_operator()
SZ_C_QT = SZ_C.to_qutip_operator()
SX_A_SY_B_QT = SX_A_QT * SY_B_QT
SX_A_SY_B_TIMES_2_QT = 2 * SX_A_SY_B_QT
OP_GLOBAL_QT = SZ_C_QT + SX_A_SY_B_TIMES_2_QT

ID_2_QUTIP = qeye(2)
ID_3_QUTIP = qeye(3)
SX_QUTIP, SY_QUTIP, SZ_QUTIP = jmat(0.5)
LX_QUTIP, LY_QUTIP, LZ_QUTIP = jmat(1.0)


SUBSYSTEMS = [
    frozenset((SITES[0],)),
    frozenset((SITES[1],)),
    frozenset((SITES[3],)),
    frozenset(
        (
            SITES[0],
            SITES[1],
        )
    ),
    frozenset(
        (
            SITES[1],
            SITES[2],
        )
    ),
]


QUTIP_TEST_CASES = {
    "product_scalar": {
        "operator": 3 * tensor(ID_2_QUTIP, ID_3_QUTIP),
        "diagonal": True,
        "scalar": True,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_diagonal": {
        "operator": tensor(SZ_QUTIP, LZ_QUTIP),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_non_diagonal": {
        "operator": tensor(SX_QUTIP, LX_QUTIP),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_zero": {
        "operator": 0 * tensor(ID_2_QUTIP, ID_3_QUTIP),
        "diagonal": True,
        "scalar": True,
        "zero": True,
        "type": np.dtype("complex128"),
    },
    "scalar": {
        "operator": 3 * ID_2_QUTIP,
        "diagonal": True,
        "scalar": True,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal": {
        "operator": SZ_QUTIP + 0.5 * ID_2_QUTIP,
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "non_diagonal": {
        "operator": LX_QUTIP,
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "zero": {
        "operator": 0 * ID_2_QUTIP,
        "diagonal": True,
        "scalar": True,
        "zero": True,
        "type": np.dtype("complex128"),
    },
    "complex": {
        "operator": SY_QUTIP,
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "dense": {
        "operator": SX_QUTIP.expm(),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal dense": {
        "operator": SZ_QUTIP.expm(),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "tensor dense": {
        "operator": tensor(SX_QUTIP, LX_QUTIP).expm(),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "tensor diagonal dense": {
        "operator": tensor(SZ_QUTIP, SZ_QUTIP).expm(),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal dense zero": {
        "operator": (-10000 * (ID_2_QUTIP + SZ_QUTIP)).expm(),
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


def test_decompose_qutip_operators():
    """
    test decomposition of qutip operators
    as sums of product operators
    """
    for name, operator_case in OPERATOR_TYPE_CASES.items():
        print("decomposing ", name)
        acts_over = tuple(sorted(operator_case.acts_over()))
        if acts_over:
            qutip_operator = operator_case.to_qutip(acts_over)
            terms = decompose_qutip_operator(qutip_operator)
            reconstructed = sum(tensor(*t) for t in terms)
            assert check_operator_equality(
                qutip_operator, reconstructed
            ), "reconstruction does not match with the original."


@pytest.mark.parametrize(
    ("case", "op_case", "expected_value"),
    [
        ("sx_A", SX_A_QT, 0.0),
        ("sy_B", SY_B_QT, 0.0),
        ("sx_A^2", SX_A2_QT, 0.25 * 2 ** (CHAIN_SIZE)),
        (
            "overlap (sxsy, sx*sy)",
            SX_A_SY_B_QT * SX_A_QT * SY_B_QT,
            0.25**2 * 2 ** (CHAIN_SIZE),
        ),
        (
            "overlap (global, sx*sy)",
            OP_GLOBAL_QT * SX_A_QT * SY_B_QT,
            2 * (0.25**2) * 2 ** (CHAIN_SIZE),
        ),
        ("Sz_C^2", SZ_C_QT * SZ_C_QT, 0.25 * 2 ** (CHAIN_SIZE)),
        (
            "sxsy^2",
            SX_A_SY_B_TIMES_2_QT * SX_A_SY_B_TIMES_2_QT,
            4 * (0.25**2) * (2**CHAIN_SIZE),
        ),
        (
            "global^2",
            OP_GLOBAL_QT * OP_GLOBAL_QT,
            (0.25 + 4 * 0.25**2) * (2**CHAIN_SIZE),
        ),
        ("global * sx_A", OP_GLOBAL_QT * SX_A_QT, 0.0),
        ("sx_A * global", SX_A_QT * OP_GLOBAL_QT, 0.0),
        (
            "global * global",
            OP_GLOBAL * OP_GLOBAL,
            (0.25 + 4 * 0.25**2) * (2**CHAIN_SIZE),
        ),
        (
            "global_QT * global_QT",
            OP_GLOBAL_QT * OP_GLOBAL_QT,
            (0.25 + 4 * 0.25**2) * (2**CHAIN_SIZE),
        ),
        (
            "global * global_qt",
            OP_GLOBAL * OP_GLOBAL_QT,
            (0.25 + 4 * 0.25**2) * (2**CHAIN_SIZE),
        ),
        (
            ">> global_qt * global",
            OP_GLOBAL_QT * OP_GLOBAL,
            (0.25 + 4 * 0.25**2) * (2**CHAIN_SIZE),
        ),
    ],
)
def test_qutip_operators(case: str, op_case: Operator, expected_value: complex):
    """Test for the qutip representation"""
    for subsystem in SUBSYSTEMS:
        ptoperator = (op_case).partial_trace(subsystem)
        assert ptoperator.tr() == expected_value


def test_as_sum_of_products():
    """
    Convert qutip operators into product
    operators back and forward
    """
    print("testing QutipOperator.as_sum_of_products")
    for name, operator_case in OPERATOR_TYPE_CASES.items():
        print("   operator", name, "of type", type(operator_case))
        qutip_op = operator_case.to_qutip_operator()
        # TODO: support handling hermitician operators
        if not qutip_op.isherm:
            continue
        reconstructed = qutip_op.as_sum_of_products()
        qutip_op2 = reconstructed.to_qutip_operator()
        assert qutip_op.system == qutip_op2.system
        print(operator_case)
        print(qutip_op.to_qutip())
        print(qutip_op2.to_qutip())
        assert qutip_op.to_qutip() == qutip_op2.to_qutip()


def test_detached_operators():
    """Check operators not coming from a system"""
    # Tests for QutipOperators defined without a system
    test_op = SX_A_SY_B_TIMES_2
    system = test_op.system
    test_op_tr = test_op.tr()
    test_op_sq_tr = (test_op * test_op).tr()
    qutip_repr = test_op.to_qutip(tuple(system.sites))
    assert test_op_tr == qutip_repr.tr()
    assert test_op_sq_tr == (qutip_repr * qutip_repr).tr()

    # Now, build a detached operator
    detached_qutip_operator = QutipOperator(qutip_repr)
    assert test_op_tr == detached_qutip_operator.tr()
    assert test_op_sq_tr == (detached_qutip_operator * detached_qutip_operator).tr()

    # sites with names
    detached_qutip_operator = QutipOperator(
        qutip_repr, names={s: i for i, s in enumerate(SITES)}
    )
    assert test_op_tr == detached_qutip_operator.tr()
    assert test_op_sq_tr == (detached_qutip_operator * detached_qutip_operator).tr()
    assert (
        test_op_tr == detached_qutip_operator.partial_trace(frozenset(SITES[0:1])).tr()
    )
    assert (
        test_op_tr == detached_qutip_operator.partial_trace(frozenset(SITES[0:2])).tr()
    )


def test_to_qutip_operator():
    # special cases
    expected_types_to_qutip = {
        "scalar, zero": ScalarOperator,
        "scalar, real": ScalarOperator,
        "scalar, complex": ScalarOperator,
        "product, zero": ScalarOperator,
        "product, 1": ScalarOperator,
    }
    for name, op_case in OPERATOR_TYPE_CASES.items():
        expected_type = expected_types_to_qutip.get(name, QutipOperator)
        op_tqo = op_case.to_qutip_operator()
        assert isinstance(
            op_tqo, expected_type
        ), f"<<{name}>> to qutip operator results in {type(op_tqo)} instead of {expected_type}"
