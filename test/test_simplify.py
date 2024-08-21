"""
Basic unit test.
"""

import numpy as np

from alpsqutip.operators import (
    LocalOperator,
    Operator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.quadratic import QuadraticFormOperator

from .helper import check_operator_equality, full_test_cases


def compute_size(operator: Operator):
    """
    compute the initial number of
    qutip operators needed to store
    operator
    """
    if isinstance(operator, ScalarOperator):
        return 0
    if isinstance(operator, LocalOperator):
        return 1
    if isinstance(operator, ProductOperator):
        return len(operator.sites_op)
    if isinstance(operator, SumOperator):
        return sum(compute_size(term) for term in operator.terms)
    if isinstance(operator, QutipOperator):
        return 1
    if isinstance(operator, QuadraticFormOperator):
        return sum(compute_size(term) for term in operator.terms)
    raise ValueError(f"Unknown kind of operator {type(operator)}")


def test_simplify():
    """test simplify operators"""

    for key, operator in full_test_cases.items():
        print("testing", key, "of type", type(operator))
        try:
            cases_dict = {"square": operator * operator, "sum": operator + operator}
        except ValueError:
            continue

        for arith_op, op_test in cases_dict.items():
            initial_size = compute_size(op_test)
            print("                - initial size of the operator:", initial_size)
            print("    checking with ", arith_op, " which produced", type(op_test))
            type_operand = type(op_test)
            print("        simplifying")
            simplify1 = op_test.simplify()
            print("        simplifying again")
            simplify2 = simplify1.simplify()
            assert type(simplify1) is type(simplify2), "types do not match"
            if isinstance(simplify1, SumOperator):
                print("        checking the consistency of sum operators")
                assert len(simplify1.terms) == len(simplify2.terms)
                assert all(
                    check_operator_equality(t1, t2)
                    for t1, t2 in zip(simplify1.terms, simplify2.terms)
                )
                for t1, t2 in zip(simplify1.terms, simplify2.terms):
                    assert t1 is t2, f"{t1} is not {t2}"
            print("        checking fixed point")
            assert (
                simplify1 is simplify2
            ), f"simplify should reach a fix point. {simplify1}->{simplify2}"
            print("        checking properties")
            # assert op_test.isherm == simplify1.isherm, "hermiticity should be preserved"
            assert (
                simplify1.isdiagonal or not op_test.isdiagonal
            ), "diagonality should be preserved"

            print("        checking that indeed the expression was simplified")
            if isinstance(op_test, SumOperator):
                if isinstance(simplify1, SumOperator):
                    final_size = compute_size(simplify1)
                    print("                - final size of the operator:", final_size)
                    assert (
                        initial_size >= final_size
                    ), f"we should get less terms, not more ({initial_size} < {final_size})."
                else:
                    assert isinstance(
                        simplify1,
                        (
                            type_operand,
                            ScalarOperator,
                            LocalOperator,
                            ProductOperator,
                        ),
                    )
