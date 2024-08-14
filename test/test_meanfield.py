"""
Basic unit test for states.
"""

from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.states.meanfield import (  # self_consistent_meanfield,
    one_body_from_qutip_operator,
    project_meanfield,
)

from .helper import (  # alert,; check_equality,; expect_from_qutip,; hamiltonian,; observable_cases,; subsystems,; sz_total,
    CHAIN_SIZE,
    check_operator_equality,
    sx_A,
    sx_B,
    sx_total,
    system,
    test_cases_states,
)

TEST_STATES = {"None": None}
TEST_STATES.update(
    {
        name: test_cases_states[name]
        for name in (
            "fully mixed",
            "z semipolarized",
            "x semipolarized",
            "first full polarized",
            "gibbs_sz",
            "gibbs_sz_as_product",
            "gibbs_sz_bar",
        )
    }
)

TEST_OPERATORS = {
    "sx_total": sx_total,
    "sx_total + sx_total^2": (sx_total + sx_total * sx_total),
    "sx_A*sx_B": sx_A * sx_B,
}


EXPECTED_PROJECTIONS = {}
# sx_total is not modified
EXPECTED_PROJECTIONS["sx_total"] = {name: sx_total for name in TEST_STATES}

# sx_total^2-> sx_total * <sx>*2*(CHAIN_SIZE-1) +
#               CHAIN_SIZE/4- <sx>^2(CHAIN_SIZE-1)*CHAIN_SIZE

EXPECTED_PROJECTIONS["sx_total + sx_total^2"] = {
    name: (sx_total + CHAIN_SIZE * 0.25)
    for name in TEST_STATES
    if name != "x semipolarized"
}
EXPECTED_PROJECTIONS["sx_total + sx_total^2"]["x semipolarized"] = (
    sx_total * (1 + 2 * (CHAIN_SIZE - 1) * 0.25)
    + CHAIN_SIZE * 0.25
    - CHAIN_SIZE * (CHAIN_SIZE - 1) * 0.25**2
)
EXPECTED_PROJECTIONS["sx_A*sx_B"] = {
    name: ScalarOperator(0, system) for name in TEST_STATES if name != "x semipolarized"
}
EXPECTED_PROJECTIONS["sx_A*sx_B"]["x semipolarized"] = -0.0625 + 0.25 * (sx_A + sx_B)


def test_meanfield_projection():
    """Test the mean field projection over different states"""

    for op_name, op_test in TEST_OPERATORS.items():
        expected = EXPECTED_PROJECTIONS[op_name]
        for state_name, sigma0 in TEST_STATES.items():
            result = project_meanfield(op_test, sigma0)
            assert check_operator_equality(
                expected[state_name], result
            ), f"failed projection {state_name} for {op_name}"


def test_one_body_from_qutip_operator():
    """
    one_body_from_qutip_operator tries to decompose
    an operator K in the sparse Qutip form into
    a sum of two operators
    K = K_0 + Delta K
    with K_0 a OneBodyOperator and
    DeltaK s.t.
    Tr[DeltaK sigma]=0
    """

    for operator_name, test_operator in TEST_OPERATORS.items():
        qutip_op = test_operator.to_qutip_operator()
        for state_name, state in TEST_STATES.items():
            result = one_body_from_qutip_operator(qutip_op, state)

            # Check the structure of the result:
            average, one_body, remainder = result.terms
            assert isinstance(
                average,
                (
                    float,
                    complex,
                    ScalarOperator,
                ),
            )
            assert isinstance(
                one_body,
                (
                    LocalOperator,
                    ScalarOperator,
                    ProductOperator,
                    OneBodyOperator,
                ),
            )
            assert isinstance(remainder, QutipOperator)

            # Check the consistency
            assert check_operator_equality(
                qutip_op, result
            ), f"decomposition failed {state_name} for {operator_name}"
