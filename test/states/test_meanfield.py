"""
Test functions that implement the mean field approximation.
"""

from test.helper import (
    CHAIN_SIZE,
    OPERATOR_TYPE_CASES,
    PRODUCT_GIBBS_GENERATOR_TESTS,
    SYSTEM,
    TEST_CASES_STATES,
    check_operator_equality,
    sx_A,
    sx_B,
    sx_total,
)

import pytest

from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.states import (
    GibbsProductDensityOperator,
    ProductDensityOperator,
)
from alpsqutip.operators.states.meanfield import (
    one_body_from_qutip_operator,
    project_meanfield,
    project_operator_to_m_body,
    project_to_n_body_operator,
)

TEST_STATES = {"None": None}
TEST_STATES.update(
    {
        name: TEST_CASES_STATES[name]
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
    name: (sx_total + CHAIN_SIZE * 0.25) for name in TEST_STATES
}
EXPECTED_PROJECTIONS["sx_A*sx_B"] = {
    name: ScalarOperator(0, SYSTEM) for name in TEST_STATES
}


@pytest.mark.parametrize(
    ["state_name", "state", "projection_name", "projection_function"],
    [
        (state_name, state, proj_name, proj_func)
        for state_name, state in TEST_CASES_STATES.items()
        for proj_name, proj_func in (
            ("project_to_n_body_operator", project_to_n_body_operator),
            ("project_operator_to_m_body", project_operator_to_m_body),
        )
        if isinstance(state, (GibbsProductDensityOperator, ProductDensityOperator))
    ],
)
def test_2body_to_1body_projection(
    state_name, state, projection_name, projection_function
):
    print(
        "Check that two-body operators project correctly to one body operators for the state",
        state_name,
    )
    for op_name, op_prod in TEST_OPERATORS.items():
        if not isinstance(op_prod, ProductOperator):
            continue
        if len(op_prod.acts_over()) != 2:
            continue
        print("* testing against", op_name)
        site1, site2 = op_prod.sites_op
        op1, op2 = op_prod.sites_op[site1], op_prod.sites_op[site2]
        rho_1 = state.partial_trace(frozenset([site1]))
        rho_2 = state.partial_trace(frozenset([site2]))
        op1_expect = (rho_1.to_qutip() * op1).tr()
        op2_expect = (rho_2.to_qutip() * op2).tr()
        projected_operator_analytical = (
            ScalarOperator(op1_expect * op2_expect, SYSTEM)
            + LocalOperator(site1, (op1 - op1_expect) * op2_expect, SYSTEM)
            + LocalOperator(site2, (op2 - op2_expect) * op1_expect, SYSTEM)
        )
        projected_operator = projection_function(op_prod, 1, state)
        if not check_operator_equality(
            projected_operator, projected_operator_analytical
        ):
            print("projections are different:\n")
            print("function:\n", projected_operator)
            print("analytical:\n", projected_operator_analytical)
            print(
                "difference:\n",
                (projected_operator - projected_operator_analytical).to_qutip(
                    tuple([site1, site2])
                ),
            )
            assert False, "Projection mismatches"


@pytest.mark.parametrize(
    ["op_name", "projection_name", "projection_function"],
    [
        (name, proj_name, proj_func)
        for name in TEST_OPERATORS
        for proj_name, proj_func in (
            ("project_to_n_body_operator", project_to_n_body_operator),
            ("project_operator_to_m_body", project_operator_to_m_body),
        )
    ],
)
def test_nbody_projection(op_name, projection_name, projection_function):
    """Test the mean field projection over different states"""
    op_test = TEST_OPERATORS[op_name]
    print("testing the consistency of projection in", op_name)
    op_sq = op_test * op_test
    proj_sq_3 = projection_function(op_sq, 3)
    proj_sq_2 = projection_function(op_sq, 2)
    proj_sq_3_2 = projection_function(proj_sq_3, 2)
    assert check_operator_equality(proj_sq_2, proj_sq_3_2), (
        f"Projections on two-body manifold using {projection_name} does not match for "
        f"{op_name} and {op_name} projected on the three body manyfold"
    )


@pytest.mark.parametrize(["op_name", "op_test"], list(TEST_OPERATORS.items()))
def test_meanfield_projection(op_name, op_test):
    """Test the mean field projection over different states"""
    expected = EXPECTED_PROJECTIONS[op_name]
    for state_name, sigma0 in TEST_STATES.items():
        print(
            "projecting <<",
            op_name,
            ">> in mean field relative to <<" + state_name + ">>",
        )
        result = project_meanfield(op_test, sigma0)
        print("sigma0", sigma0)
        print("expected:\n", expected[state_name])
        print("result:\n", result)

        assert check_operator_equality(
            expected[state_name], result
        ), f"failed projection {state_name} for {op_name}"


@pytest.mark.parametrize(
    ["operator_case", "operator"], list(OPERATOR_TYPE_CASES.items())
)
def test_one_body_from_qutip_operator_1(operator_case, operator):
    print(operator_case, "as scalar + one body + rest")
    result = one_body_from_qutip_operator(operator.to_qutip_operator())

    assert check_operator_equality(result, operator), "operators are not equivalent."
    if isinstance(result, (ScalarOperator, OneBodyOperator, LocalOperator)):
        return
    assert isinstance(
        result, SumOperator
    ), "the result must be a one-body operator or a sum"
    terms = result.terms
    assert len(terms) <= 3, "the result should have at most three terms."
    if not isinstance(terms[-1], (ScalarOperator, OneBodyOperator, LocalOperator)):
        last = terms[-1]
        terms = terms[:-1]
        assert abs(last.tr()) < 1e-10, "Reminder term should have zero trace."
        assert abs(terms[-1].tr()) < 1e-10, "One-body term should have zero trace."

    assert (
        isinstance(result, (ScalarOperator, OneBodyOperator, LocalOperator))
        for term in terms
    ), "first two terms should be one-body operators"


@pytest.mark.parametrize(
    ["operator_case", "operator"], list(OPERATOR_TYPE_CASES.items())
)
def test_one_body_from_qutip_operator_with_reference(operator_case, operator):

    for name_ref, gen in PRODUCT_GIBBS_GENERATOR_TESTS.items():
        sigma = GibbsProductDensityOperator(gen)

        print(operator_case, "as scalar + one body + rest w.r.t. " + name_ref)
        result = one_body_from_qutip_operator(operator.to_qutip_operator(), sigma)

        assert check_operator_equality(
            result, operator
        ), "operators are not equivalent."
        if isinstance(result, (ScalarOperator, OneBodyOperator, LocalOperator)):
            return
        assert isinstance(
            result, SumOperator
        ), "the result must be a one-body operator or a sum"
        terms = result.terms
        assert len(terms) <= 3, "the result should have at most three terms."
        print("    types:", [type(term) for term in terms])
        print("    expectation values:", [sigma.expect(term) for term in terms])
        print(
            "    expectation value:",
            sigma.expect(operator),
            sigma.expect(result),
            sigma.expect(operator.to_qutip_operator()),
        )

        if not isinstance(terms[-1], (ScalarOperator, OneBodyOperator, LocalOperator)):
            last = terms[-1]
            terms = terms[:-1]
            assert (
                abs(sigma.expect(last)) < 1e-10
            ), "Reminder term should have zero mean."
            assert (
                abs(sigma.expect(terms[-1])) < 1e-10
            ), "One-body term should have zero mean."
            # TODO: check also the orthogonality between last and one-body terms.

        assert (
            isinstance(result, (ScalarOperator, OneBodyOperator, LocalOperator))
            for term in terms
        ), "first two terms should be one-body operators"


def test_one_body_from_qutip_operator_2():
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
