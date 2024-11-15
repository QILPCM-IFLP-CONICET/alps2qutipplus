"""
Basic unit test for states.
"""

from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
)
from alpsqutip.operators.states.meanfield import (  # self_consistent_meanfield,
    one_body_from_qutip_operator,
    project_meanfield,
    project_to_n_body_operator,
)

from .helper import (  # alert,; check_equality,; expect_from_qutip,;; hamiltonian,; observable_cases,; subsystems,; sz_total,
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


def no_test_nbody_projection():
    """Test the mean field projection over different states"""
    failed = False
    for op_name, op_test in TEST_OPERATORS.items():
        print("testing the consistency of projection in", op_name)
        op_sq = op_test * op_test
        proj_sq_3 = project_to_n_body_operator(op_sq, 3)
        proj_sq_2 = project_to_n_body_operator(op_sq, 2)
        proj_sq_3_2 = project_to_n_body_operator(proj_sq_3, 2)
        if not check_operator_equality(proj_sq_2, proj_sq_3_2):
            print(" Projections do not match.")
            failed = True
    assert not failed


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
    failed = {}

    def check_result(qutip_op, result):
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
        # Check that the remainder and the one body terms have
        # zero mean:
        if state is None:
            assert abs(one_body.to_qutip().tr()) < 1.0e-9
            assert abs((remainder.to_qutip()).tr()) < 1.0e-9
        else:
            error_one_body_tr = abs((one_body.to_qutip() * state.to_qutip()).tr())
            if error_one_body_tr > 1.0e-9:
                failed.setdefault((operator_name, state_name), {})[
                    "one body tr"
                ] = error_one_body_tr
            remainder_tr = abs((remainder.to_qutip() * state.to_qutip()).tr())
            if remainder_tr > 1.0e-9:
                failed.setdefault((operator_name, state_name), {})[
                    "remainder tr"
                ] = remainder_tr
        # Check the consistency
        if not check_operator_equality(qutip_op.to_qutip(), result.to_qutip()):
            print(f"decomposition failed {state_name} for {operator_name}")
            failed.setdefault((state_name, operator_name), {})[
                "operator equality"
            ] = False

    for operator_name, test_operator in TEST_OPERATORS.items():
        full_sites = tuple(test_operator.system.sites)
        print(
            "\n",
            60 * "-",
            "\n# operator name",
            operator_name,
            "of type",
            type(test_operator),
        )
        qutip_op = test_operator.to_qutip_operator()
        print("      ->qutip_op", type(qutip_op), qutip_op.acts_over())
        for state_name, state in TEST_STATES.items():
            print(f"  - on state {state_name}")
            print("      * as QutipOperator")
            result = one_body_from_qutip_operator(qutip_op, state)
            check_result(qutip_op, result)
            print("      * as Qobj")
            result = one_body_from_qutip_operator(qutip_op.to_qutip(full_sites), state)
            check_result(qutip_op, result)

    if failed:
        print("Discrepances:")
        print("~~~~~~~~~~~~~")
        for key, errors in failed.items():
            print("   in ", key)
            for err_key, error in errors.items():
                print("    ", err_key, error)

        assert False, "discrepances"
