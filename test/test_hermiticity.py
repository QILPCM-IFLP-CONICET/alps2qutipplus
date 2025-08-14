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
from alpsqutip.operators.quadratic import QuadraticFormOperator
from alpsqutip.operators.states.gibbs import GibbsDensityOperator, GibbsProductDensityOperator

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)

OPERATORS_AND_STATE_CASES = OPERATOR_TYPE_CASES.copy()
OPERATORS_AND_STATE_CASES.update(TEST_CASES_STATES)

OPERATOR_TYPE_CASES_QUTIP = {
    key: operator.to_qutip() for key, operator in OPERATORS_AND_STATE_CASES.items()
}




@pytest.mark.parametrize(["operator_name"],
                        [(x,) for x in OPERATORS_AND_STATE_CASES]
                        )
def test_consistency_as_sum_of_products1(operator_name):
    op_case = OPERATORS_AND_STATE_CASES[operator_name]
    if not op_case.isherm:
        return
    as_sum_of_products = op_case.as_sum_of_products()
    assert as_sum_of_products.isherm
    as_sum_of_products2 = (op_case*op_case).as_sum_of_products()
    assert as_sum_of_products2.isherm



@pytest.mark.parametrize(["operator_name"],
                        list((x,) for x in  OPERATORS_AND_STATE_CASES)
                        )
def test_hermiticity_hermitician_part(operator_name):
    op_case = OPERATORS_AND_STATE_CASES[operator_name]
    herm_part = (op_case+ op_case.dag())*.5
    assert herm_part.isherm
    # TODO: Fix these cases
    if type(op_case) in (QuadraticFormOperator,
                         GibbsDensityOperator,
                         GibbsProductDensityOperator,):
        return
    op_case = OPERATORS_AND_STATE_CASES[operator_name]
    herm_part = (op_case- op_case.dag())*.5j
    
    assert herm_part.isherm




@pytest.mark.parametrize(["name_op1", "name_op2"],
                        list((x,y) for x in OPERATORS_AND_STATE_CASES
                              for y in OPERATORS_AND_STATE_CASES)
                        )
def test_consistency_sum_operators_hermiticity(name_op1, name_op2):
    test_op1 = OPERATORS_AND_STATE_CASES[name_op1]
    test_op2 = OPERATORS_AND_STATE_CASES[name_op2]
    if not(test_op1.isherm or test_op2.isherm):
        # The result is not defined
        return
    result = test_op1 + test_op2
    if test_op1.isherm and test_op2.isherm:
        assert result.isherm
    if test_op1.isherm is False and test_op2.isherm is False:
        assert not result.isherm

    

    
