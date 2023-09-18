"""
Basic unit test.
"""

from alpsqutip.model import Operator
from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
    SumOperator,
)

from .helper import (
    CHAIN_SIZE,
    check_operator_equality,
    hamiltonian,
    observable_cases,
    sites,
    sx_A as local_sx_A,
    sy_A,
    sy_B,
    sz_A,
    sz_C,
    sz_total,
)

sx_A = ProductOperator({local_sx_A.site: local_sx_A.operator}, 1.0, local_sx_A.system)
sx_A2 = sx_A * sx_A
sx_Asy_B = sx_A * sy_B
sx_AsyB_times_2 = 2 * sx_Asy_B
opglobal = sz_C + sx_AsyB_times_2


def test_build_hamiltonian():
    """build ham"""
    assert sz_total is not None
    assert hamiltonian is not None
    hamiltonian_with_field = hamiltonian + sz_total
    assert check_operator_equality(
        (hamiltonian_with_field).to_qutip(),
        (hamiltonian.to_qutip() + sz_total.to_qutip()),
    )


def test_isherm_operator():
    for key, observable in observable_cases.items():
        if not isinstance(observable, Operator):
            continue
        assert observable.isherm, f"{key} is not hermitician?"

        ham = observable_cases["hamiltonian"]
        sz = observable_cases["sz_total"]
        print("***addition***")
        assert (ham + 1.0).isherm
        assert (ham + sz).isherm
        print("***scalar multiplication***")
        assert (2.0 * ham).isherm
        print("***scalar multiplication for a OneBody Operator")
        assert (2.0 * sz).isherm
        assert (ham * 2.0).isherm
        assert (sz * 2.0).isherm
        assert (sz.expm()).isherm
        assert (ham**3).isherm


def test_type_operator():
    """Tests for operator types"""
    assert isinstance(sx_A, ProductOperator)
    assert isinstance(sy_B, LocalOperator)
    assert isinstance(sz_C, LocalOperator)
    assert isinstance(2 * sy_B, LocalOperator)
    assert isinstance(sy_B * 2, LocalOperator)
    assert isinstance(sx_A + sy_B, OneBodyOperator)
    assert isinstance(sx_A + sy_B + sz_C, OneBodyOperator)
    assert isinstance(sx_A + sy_B + sx_A * sz_C, SumOperator)
    assert isinstance(3.0 * (sx_A + sy_B), OneBodyOperator)
    assert isinstance((sx_A + sy_B) * 2.0, OneBodyOperator)
    assert isinstance(sy_B + sx_A, OneBodyOperator)
    assert isinstance(sx_Asy_B, ProductOperator)
    assert len(sx_Asy_B.sites_op) == 2
    assert isinstance(sx_AsyB_times_2, ProductOperator)
    assert isinstance(opglobal, SumOperator)
    assert isinstance(sx_A + sy_B, SumOperator)
    assert len(sx_AsyB_times_2.sites_op) == 2

    opglobal.prefactor = 2
    assert sx_A2.prefactor == 1
    assert opglobal.prefactor == 2

    assert check_operator_equality(sx_A, sx_A.to_qutip())
    terms = [sx_A, sy_A, sz_A]
    assert check_operator_equality(sum(terms), sum(t.to_qutip() for t in terms))
    assert check_operator_equality(sx_A.inv(), sx_A.to_qutip().inv())
    opglobal_offset = opglobal + 1.3821
    assert check_operator_equality(
        opglobal_offset.inv(), opglobal_offset.to_qutip().inv()
    )


def test_inv_operator():
    """test the exponentiation of different kind of operators"""
    sx_A_inv = sx_A.inv()
    assert isinstance(sx_A_inv, LocalOperator)
    assert check_operator_equality(sx_A_inv.to_qutip(), sx_A.to_qutip().inv())

    sx_obl = sx_A + sy_B + sz_C
    sx_obl_inv = sx_obl.inv()
    assert isinstance(sx_obl_inv, QutipOperator)
    assert check_operator_equality(sx_obl_inv.to_qutip(), sx_obl.to_qutip().inv())

    s_prod = sx_A * sy_B * sz_C
    s_prod_inv = s_prod.inv()
    assert isinstance(s_prod, ProductOperator)
    assert check_operator_equality(s_prod_inv.to_qutip(), s_prod.to_qutip().inv())

    opglobal_offset = opglobal + 1.3821
    opglobal_offset_inv = opglobal_offset.inv()
    assert isinstance(opglobal_offset_inv, QutipOperator)
    assert check_operator_equality(
        opglobal_offset_inv.to_qutip(), opglobal_offset.to_qutip().inv()
    )


def test_exp_operator():
    """test the exponentiation of different kind of operators"""
    sx_A_exp = sx_A.expm()
    assert isinstance(sx_A_exp, LocalOperator)
    assert check_operator_equality(sx_A_exp.to_qutip(), sx_A.to_qutip().expm())

    sx_obl = sx_A + sy_B + sz_C
    sx_obl_exp = sx_obl.expm()
    assert isinstance(sx_obl_exp, ProductOperator)
    assert check_operator_equality(sx_obl_exp.to_qutip(), sx_obl.to_qutip().expm())

    opglobal_exp = opglobal.expm()
    assert isinstance(opglobal_exp, QutipOperator)
    assert check_operator_equality(opglobal_exp.to_qutip(), opglobal.to_qutip().expm())


def test_local_operator():
    """Tests for local operators"""
    assert (sx_A * sx_A).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
    assert (sz_A * sz_A).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)

    print("product * local", type(sx_A * sy_A))
    print("local * product", type(sy_A * sx_A))
    print("commutator:", type(sx_A * sy_A - sy_A * sx_A))
    print(((sx_A * sy_A - sy_A * sx_A) * sz_A).tr(), -1j * 0.5 * 2 ** (CHAIN_SIZE - 1))
    assert ((sx_A * sy_A - sy_A * sx_A) * sz_A).tr() == (
        -1j * 0.5 * 2 ** (CHAIN_SIZE - 1)
    )
    assert (sz_A * (sx_A * sy_A - sy_A * sx_A)).tr() == (
        -1j * 0.5 * 2 ** (CHAIN_SIZE - 1)
    )

    assert (sz_A * sy_B * sz_A * sy_B).tr() == 0.25 * 2 ** (CHAIN_SIZE - 2)
    assert (
        (sx_A * sy_A * sy_B - sy_A * sx_A * sy_B) * (sz_A * sy_B)
    ).tr() == -1j * 0.25 * 2 ** (CHAIN_SIZE - 2)
    assert (
        (sz_A * sy_B) * (sx_A * sy_A * sy_B - sy_A * sx_A * sy_B)
    ).tr() == -1j * 0.25 * 2 ** (CHAIN_SIZE - 2)

    assert sx_A.tr() == 0.0
    assert (sx_A2).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
    assert (sz_C * sz_C).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)

    sx_A_qt = sx_A.to_qutip()
    sx_A2_qt = sx_A2.to_qutip()

    assert sx_A_qt.tr() == 0
    assert (sx_A2_qt).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
    assert sx_A.partial_trace((sites[0],)).tr() == 0.0
    assert sx_A.partial_trace((sites[1],)).tr() == 0.0
    assert sx_A.partial_trace((sites[0], sites[1])).tr() == 0.0
    assert sx_A.partial_trace((sites[1], sites[2])).tr() == 0.0


def test_product_operator():
    """Tests for product operators"""

    assert (sx_Asy_B * sx_A * sy_B).tr() == 0.25 * 2 ** (CHAIN_SIZE - 2)
    assert (opglobal * sx_A * sy_B).tr() == 0.5 * 2 ** (CHAIN_SIZE - 2)
    assert (sx_AsyB_times_2 * sx_AsyB_times_2).tr() == 2 ** (CHAIN_SIZE - 2)
    assert (opglobal * opglobal).tr() == 2 ** (CHAIN_SIZE - 2) * 2

    sx_A_qt = sx_A.to_qutip()
    syB_qt = sy_B.to_qutip()
    szC_qt = sz_C.to_qutip()

    sx_AsyB_qt = sx_Asy_B.to_qutip()
    sx_AsyB_times_2_qt = sx_AsyB_times_2.to_qutip()
    opglobal_qt = opglobal.to_qutip()

    assert (sx_AsyB_qt * sx_A_qt * syB_qt).tr() == 0.25 * 2 ** (CHAIN_SIZE - 2)
    assert (opglobal_qt * sx_A_qt * syB_qt).tr() == 0.5 * 2 ** (CHAIN_SIZE - 2)
    assert (szC_qt * szC_qt).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
    assert (sx_AsyB_times_2_qt * sx_AsyB_times_2_qt).tr() == 2 ** (CHAIN_SIZE - 2)
    assert (opglobal_qt * opglobal_qt).tr() == 2 ** (CHAIN_SIZE - 2) * 2


def test_qutip_operators():
    """Test for the quip representation"""

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
        assert (sx_A2_qt).partial_trace(subsystem).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
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
        assert (opglobal_qt * opglobal_qt).partial_trace(subsystem).tr() == 2 ** (
            CHAIN_SIZE - 2
        ) * 2
        assert (opglobal_qt * sx_A_qt).partial_trace(subsystem).tr() == 0.0
        assert (opglobal_qt * opglobal).partial_trace(subsystem).tr() == 2 ** (
            CHAIN_SIZE - 2
        ) * 2
        assert (opglobal * opglobal_qt).partial_trace(subsystem).tr() == 2 ** (
            CHAIN_SIZE - 2
        ) * 2

    # Tests for QutipOperators defined without a system
    detached_qutip_operator = QutipOperator(sx_AsyB_times_2_qt.operator)
    assert ((sx_AsyB_times_2_qt.operator) ** 2).tr() == 2 ** (CHAIN_SIZE - 2)
    assert (detached_qutip_operator * detached_qutip_operator).tr() == 2 ** (
        CHAIN_SIZE - 2
    )

    detached_qutip_operator = QutipOperator(
        sx_AsyB_times_2_qt.operator, names={s: i for i, s in enumerate(sites)}
    )
    assert (detached_qutip_operator * detached_qutip_operator).partial_trace(
        sites[0]
    ).tr() == 2 ** (CHAIN_SIZE - 2)
