"""
Basic unit test.
"""

import numpy as np

from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
    SumOperator,
)
from alpsqutip.operators.states import DensityOperatorMixin

from .helper import (
    CHAIN_SIZE,
    check_operator_equality,
    full_test_cases,
    hamiltonian,
    operator_type_cases,
    sites,
    sx_A as local_sx_A,
    sy_A,
    sy_B,
    sz_A,
    sz_C,
    sz_total,
)

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


sx_A = ProductOperator({local_sx_A.site: local_sx_A.operator}, 1.0, local_sx_A.system)
sx_A2 = sx_A * sx_A
sx_Asy_B = sx_A * sy_B
sx_AsyB_times_2 = 2 * sx_Asy_B
opglobal = sz_C + sx_AsyB_times_2


def test_acts_over():
    """Check acts_over method"""
    full_chain = {f"1[{s}]" for s in range(CHAIN_SIZE)}

    results = {
        "scalar, real": set(),
        "scalar, complex": set(),
        "local operator, hermitician": {"1[0]"},
        "local operator, non hermitician": {"1[0]"},
        "One body, hermitician": full_chain,
        "One body, non hermitician": full_chain,
        "three body, hermitician": {"1[0]", "1[1]", "1[2]"},
        "three body, non hermitician": full_chain,
        "product operator, hermitician": {"1[0]", "1[1]"},
        "product operator, non hermitician": {"1[0]", "1[1]"},
        "sum operator, hermitician": {"1[0]", "1[1]"},
        "sum operator, hermitician from non hermitician": {"1[0]", "1[1]"},
        "sum operator, anti-hermitician": {"1[0]", "1[1]"},
        "hermitician quadratic operator": full_chain,
        "non hermitician quadratic operator": full_chain,
        "fully mixed": full_chain,
        "z semipolarized": full_chain,
        "x semipolarized": full_chain,
        "first full polarized": full_chain,
        "gibbs_sz_as_product": full_chain,
        "qutip operator": None,
        "gibbs_sz": full_chain,
        "gibbs_sz_bar": full_chain,
        "gibbs_H": full_chain,
        "mixture": full_chain,
        "mixture of first and second partially polarized": {
            "1[3]",
            "1[0]",
            "1[1]",
            "1[2]",
        },
        "log unitary": {"1[0]", "1[3]", "1[2]", "1[1]"},
        "single interaction term": {"1[1]", "1[0]"},
    }

    for name, operator in full_test_cases.items():
        print(name)
        acts_over = operator.acts_over()
        print("    acts over ", acts_over)
        assert acts_over == results[name]


def test_product_and_trace():
    """Check acts_over method"""
    qutip_ops = {}
    skip_cases = {
        "hermitician quadratic operator",
        "non hermitician quadratic operator",
    }

    passed = True

    for name1, operator1 in full_test_cases.items():
        if name1 in skip_cases:
            continue

        print("checking the trace of ", name1)
        op_qutip_1 = qutip_ops.get(name1, None)
        if op_qutip_1 is None:
            op_qutip_1 = operator1.to_qutip()
            qutip_ops[name1] = op_qutip_1

        if not abs(operator1.tr() - op_qutip_1.tr()) < 1.0e-10:
            print("   failed:", "\033[91mtraces should match.\033[0m")
            passed = False
            continue

        for name2, operator2 in full_test_cases.items():
            if name2 in skip_cases:
                continue

            print("checking the trace of the product of ", name1, "and", name2)
            op_qutip_2 = qutip_ops.get(name2, None)
            if op_qutip_2 is None:
                op_qutip_2 = operator2.to_qutip()
                qutip_ops[name2] = op_qutip_2
            # The trace of the products should match
            prod = operator1 * operator2
            if isinstance(prod, DensityOperatorMixin):
                passed = False
                print(
                    "\033[91m  failed: \033[0m",
                    f"Product of {type(operator1)}*{type(operator2)} is a density matrix.",
                )
                continue
            alps_trace = (prod).tr()
            qutip_trace = (op_qutip_1 * op_qutip_2).tr()
            if not abs(alps_trace - qutip_trace) < 1.0e-10:
                passed = False
                print(
                    "\033[91m  failed:\033[0m",
                    f"the traces of the products should match. {type(operator1)}*{type(operator2)}->{type(prod)}, {alps_trace}!={qutip_trace}",
                )
                continue
            print("\033[92m  passed.\033[0m")
        assert passed, "there are inconsistencies"


def test_build_hamiltonian():
    """build ham"""
    assert sz_total is not None
    assert hamiltonian is not None
    hamiltonian_with_field = hamiltonian + sz_total
    assert check_operator_equality(
        (hamiltonian_with_field).to_qutip(),
        (hamiltonian.to_qutip() + sz_total.to_qutip()),
    )


def test_type_operator():
    """Tests for operator types"""
    assert isinstance(sx_A, ProductOperator)
    assert isinstance(sy_B, LocalOperator)
    assert isinstance(sz_C, LocalOperator)
    assert isinstance(2 * sy_B, LocalOperator)
    assert isinstance(sy_B * 2, LocalOperator)
    assert isinstance(sx_A + sy_B, OneBodyOperator)
    assert isinstance(
        sx_A + sy_B + sz_C, OneBodyOperator
    ), f"{type(sx_A + sy_B + sz_C)} is not OneBodyOperator"
    assert isinstance(sx_A + sy_B + sx_A * sz_C, SumOperator)
    assert isinstance((sx_A + sy_B), OneBodyOperator)
    assert isinstance(
        (sx_A + sy_B) * 3, OneBodyOperator
    ), f"{type((sx_A + sy_B)*3.)} is not OneBodyOperator"
    assert isinstance(
        3.0 * (sx_A + sy_B), OneBodyOperator
    ), f"{type(3.0*(sx_A + sy_B))} is not OneBodyOperator"

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
    sx_a_inv = sx_A.inv()
    assert isinstance(sx_a_inv, LocalOperator)
    assert check_operator_equality(sx_a_inv.to_qutip(), sx_A.to_qutip().inv())

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
    print("type sx_obl", type(sx_obl))
    print("sx_obl:\n", sx_obl)
    print("sx_obl->qutip:", sx_obl.to_qutip())

    sx_obl_exp = sx_obl.expm()
    assert isinstance(sx_obl_exp, ProductOperator)
    print("exponential and conversion:")
    print(sx_obl_exp.to_qutip())
    print("conversion and then exponential:")
    print(sx_obl.to_qutip().expm())
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
    print(
        ((sx_A * sy_A - sy_A * sx_A) * sz_A).tr(),
        -1j * 0.5 * 2 ** (CHAIN_SIZE - 1),
    )
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


def test_arithmetic_operators():
    """
    Test consistency of arithmetic expressions
    """
    operator_type_cases_qutip = {
        key: operator.to_qutip() for key, operator in operator_type_cases.items()
    }

    for key1, test_operator1 in operator_type_cases.items():
        op1_qutip = operator_type_cases_qutip[key1]
        result = -test_operator1
        check_operator_equality(result.to_qutip(), -op1_qutip)

        result = test_operator1.dag()
        check_operator_equality(result.to_qutip(), op1_qutip.dag())

        for key2, test_operator2 in operator_type_cases.items():
            print("add ", key1, " and ", key2)
            op2_qutip = operator_type_cases_qutip[key2]
            print(type(test_operator1), "+", type(test_operator2))
            result = test_operator1 + test_operator2

            check_operator_equality(result.to_qutip(), (op1_qutip + op2_qutip))

            print("product of ", key1, " and ", key2)
            result = test_operator1 * test_operator2

            check_operator_equality(result.to_qutip(), (op1_qutip * op2_qutip))
