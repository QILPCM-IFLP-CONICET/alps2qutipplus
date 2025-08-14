"""
Test functions that implement n-body projections
"""

import os
from test.helper import (
    CHAIN_SIZE,
    HAMILTONIAN,
    OPERATOR_TYPE_CASES,
    PRODUCT_GIBBS_GENERATOR_TESTS,
    SX_A,
    SX_B,
    SX_TOTAL,
    SYSTEM,
    TEST_CASES_STATES,
    check_operator_equality,
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
)
from alpsqutip.operators.states.meanfield.projections import (
    _project_qutip_operator_to_m_body_recursive,
    project_product_operator_as_n_body_operator,
    project_qutip_operator_as_n_body_operator,
    project_to_n_body_operator,
)
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

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
    "sx_total": SX_TOTAL,
    "-sx_total - sx_total^2/(N-1)": (
        -SX_TOTAL - SX_TOTAL * SX_TOTAL / (CHAIN_SIZE - 1)
    ),
    "sx_A*sx_B": SX_A * SX_B,
}
EXPECTED_PROJECTIONS = {}


# sx_total is not modified
EXPECTED_PROJECTIONS["sx_total"] = {name: SX_TOTAL for name in TEST_STATES}


# TODO: build this analytically
EXPECTED_PROJECTIONS["-sx_total - sx_total^2/(N-1)"] = {
    name: (
        SX_TOTAL * (-1 - 2 * 0.343947)
        + CHAIN_SIZE * (0.25 / (CHAIN_SIZE - 1) - 0.0483673)
    )
    for name in TEST_STATES
}
EXPECTED_PROJECTIONS["-sx_total - sx_total^2/(N-1)"]["x semipolarized"] = (
    -1 - 2 * 0.343947
) * SX_TOTAL + 0.1399

EXPECTED_PROJECTIONS["sx_A*sx_B"] = {
    name: ScalarOperator(0, SYSTEM) for name in TEST_STATES
}
EXPECTED_PROJECTIONS["sx_A*sx_B"]["x semipolarized"] = -0.00386592 + -0.0621765 * (
    SX_A + SX_B
)  # aqui


##################################### Heavier tests #####################################################


if os.environ.get("ALPSQUTIP_ALLTESTS"):
    ## Full Identity:
    TEST_OPERATORS["2^CHAIN_SIZE*Identity"] = ProductOperator(
        {name: 2 * site_descr["identity"] for name, site_descr in SYSTEM.sites.items()},
        1,
        SYSTEM,
    )
    EXPECTED_PROJECTIONS["2^CHAIN_SIZE*Identity"] = {
        name: TEST_OPERATORS["2^CHAIN_SIZE*Identity"] for name in TEST_STATES
    }

    # Square of the Hamiltonian (many 4-body operators)
    TEST_OPERATORS["Hamiltonian^2"] = HAMILTONIAN * HAMILTONIAN

    # Hard to converge: frustrated Hamiltonian
    TEST_OPERATORS["sx_total + sx_total^2/(N-1)"] = SX_TOTAL + SX_TOTAL * SX_TOTAL / (
        CHAIN_SIZE - 1
    )
    # TODO: build this analytically
    EXPECTED_PROJECTIONS["sx_total + sx_total^2/(N-1)"] = {
        name: (
            SX_TOTAL * (1 - 2 * 0.23105)
            + CHAIN_SIZE * (0.25 / (CHAIN_SIZE - 1) - 0.23105**2)
        )
        for name in TEST_STATES
    }
    EXPECTED_PROJECTIONS["sx_total + sx_total^2/(N-1)"]["x semipolarized"] = (
        1 - 2 * 0.3175
    ) * SX_TOTAL - 0.0700809


######################################################


@pytest.mark.parametrize(["op_name", "op_test"], list(TEST_OPERATORS.items()))
def test_compare_recursive_and_iterative_n_body_projections(op_name, op_test):
    """
    Compare the results of the recursive and the iterative implementations
    of the n-body projections.
    """
    failed = {}
    print(f"projecting <<{op_name}>> in mean field")
    isherm = op_test.isherm

    for state_name, sigma0 in TEST_STATES.items():
        print(f"  = sigma0{state_name}:\n", sigma0)
        if sigma0 is not None:
            print(" <sx>=", sigma0.expect(SX_TOTAL) / CHAIN_SIZE)
        else:
            print(" <sx>=", 0)
        for n_body in [1]:
            print("   n=", n_body)
            result_m = project_operator_to_m_body(op_test, n_body, sigma0)
            result_n = project_to_n_body_operator(op_test, n_body, sigma0)
            if isherm:
                assert (
                    result_m.isherm
                ), "project_operator_to_m_body should preserve hermiticity"
                assert (
                    result_n.isherm
                ), "project_to_n_body_operator should preserve hermiticity"

            if not check_operator_equality(result_m, result_n, 1.0e-6):
                failed[
                    (
                        state_name,
                        n_body,
                    )
                ] = f"Result m:\n{result_m}\n\n Result n:\n{result_n}\n\nDelta = \n{result_m.to_qutip()-result_n.to_qutip()}"
            else:
                print("    ...OK")
    if failed:
        for fail in failed:
            print(f" failed with <<{fail}>> as state seed. ")
            print(failed[fail])
            print(60 * "=")
        assert False, "Projections do not match."


@pytest.mark.parametrize(["op_name", "op_test"], list(TEST_OPERATORS.items()))
def test_compare_iterative_and_recursive_n_body_qutip_projections(op_name, op_test):
    """
    This test compares the results of using the recursive
    `_project_qutip_operator_to_m_body_recursive` and the iterative
    `project_qutip_operator_as_n_body_operator` n-body projections.
    """
    failed = {}
    print(f"projecting <<{op_name}>> in mean field")
    print("op_test:", op_test)
    print("op_test:", op_test.to_qutip_operator())
    op_test = op_test.to_qutip_operator()
    assert isinstance(op_test, QutipOperator)

    for state_name, sigma0 in TEST_STATES.items():
        print(f"  = sigma0{state_name}")
        for n_body in range(0, 4):
            print("   n=", n_body)
            result_m = _project_qutip_operator_to_m_body_recursive(
                op_test, n_body, sigma0
            )
            result_n = project_qutip_operator_as_n_body_operator(
                op_test, n_body, sigma0
            )
            if not check_operator_equality(result_m, result_n, 5e-6):
                failed[
                    (
                        state_name,
                        n_body,
                    )
                ] = f"Result m:\n{result_m}\n\n Result n:\n{result_n}\n\nDelta = \n{result_m.to_qutip()-result_n.to_qutip()}"
            else:
                print("    ...OK")
    if failed:
        for fail in failed:
            print(f" failed with <<{fail}>> as state seed. ")
            print(failed[fail])
            print(60 * "=")
        assert False, "Self-consistency failed for some seeds."


@pytest.mark.parametrize(["op_name", "op_test"], list(TEST_OPERATORS.items()))
def test_compare_iterative_and_recursive_n_body_product_projections(op_name, op_test):
    """
    This test compares the results of using the recursive
    `project_operator_to_m_body` and the iterative specific
    `project_product_operator_as_n_body_operator` product n-body projections.
    """
    failed = {}
    print(f"projecting <<{op_name}>> in mean field")
    print("op_test:", op_test)
    if not isinstance(op_test, ProductOperator):
        op_test = op_test.to_qutip_operator().as_sum_of_products()
    if not isinstance(op_test, ProductOperator):
        if isinstance(op_test, SumOperator):
            op_test = sorted(op_test.terms, key=lambda x: len(x.acts_over()))[-1]
        else:
            return

    for state_name, sigma0 in TEST_STATES.items():
        print(f"  = sigma0{state_name}")
        for n_body in range(0, 4):
            print("   n=", n_body)
            result_m = project_operator_to_m_body(op_test, n_body, sigma0)
            result_n = project_product_operator_as_n_body_operator(
                op_test, n_body, sigma0
            )
            if not check_operator_equality(result_m, result_n, 1e-7):
                failed[
                    (
                        state_name,
                        n_body,
                    )
                ] = f"Result m:\n{result_m}\n\n Result n:\n{result_n}\n\nDelta = \n{result_m.to_qutip()-result_n.to_qutip()}"
            else:
                print("    ...OK")
    if failed:
        for fail in failed:
            print(f" failed with <<{fail}>> as state seed. ")
            print(failed[fail])
            print(60 * "=")
        assert False, "Self-consistency failed for some seeds."


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
def test_idempotency_nbody_projection(op_name, projection_name, projection_function):
    """
    Test the mean field projection over different states,
    and using both implementations.
    Also check if hermiticity is preserved.

    """
    op_test = TEST_OPERATORS[op_name]
    print("testing the consistency of projection in", op_name)
    isherm = op_test.isherm
    op_sq = op_test * op_test
    if isherm:
        assert op_sq.isherm

    proj_sq_3 = projection_function(op_sq, 3)
    if isherm:
        assert op_sq.isherm
    proj_sq_2 = projection_function(op_sq, 2)
    if isherm:
        assert op_sq.isherm
    proj_sq_3_2 = projection_function(proj_sq_3, 2)
    if isherm:
        assert op_sq.isherm

    assert check_operator_equality(proj_sq_2, proj_sq_3_2, 5e-7), (
        f"Projections on two-body manifold using {projection_name} does not match for "
        f"{op_name} and {op_name} projected on the three body manyfold"
    )


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
def test_2body_to_1body_product_projection(
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
            projected_operator, projected_operator_analytical, tolerance=5e-9
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


@pytest.mark.parametrize(["op_name", "op_test"], list(TEST_OPERATORS.items()))
def test_self_consistent_meanfield_projection(op_name, op_test):
    """Test the mean field projection over different states"""
    if op_name not in EXPECTED_PROJECTIONS:
        return
    expected = EXPECTED_PROJECTIONS[op_name]
    failed = {}
    print(f"projecting <<{op_name}>> in mean field")

    for state_name, sigma0 in TEST_STATES.items():
        print("sigma state:", state_name)
        result = project_meanfield(op_test, sigma0, max_it=30)
        sigma_MF = GibbsProductDensityOperator(result)
        print("  <sx>_MF=", sigma_MF.expect(SX_TOTAL) / CHAIN_SIZE)
        sigma_MF_expected = GibbsProductDensityOperator(expected[state_name])
        print("  <sx>_MF_expected=", sigma_MF_expected.expect(SX_TOTAL) / CHAIN_SIZE)

        if not check_operator_equality(
            expected[state_name].to_qutip(), result.to_qutip(), 1e-3
        ):
            print("   ->failed")
            failed[state_name] = (
                f"\n\n result:\n{result}\n\n expected:\n{expected[state_name]}\n\nDelta:{result.to_qutip()-expected[state_name].to_qutip()}"
            )
    if failed:
        for fail in failed:
            print(f" failed with <<{fail}>> as state seed. ")
            print(failed[fail])
        fail_msg = (
            "Self-consistency failed for some seeds:"
            + "".join(key for key in failed)
            + "."
        )
        assert False, fail_msg


@pytest.mark.parametrize(["op_name", "op_test"], list(TEST_OPERATORS.items()))
def test_compare_meanfield_projection_using_iterative_and_recursive_projections(
    op_name, op_test
):
    """
    Compare the results of the self-consistent mean field projection from
    both iterative and recursive projection routines.
    """
    failed = {}
    print(f"projecting <<{op_name}>> in mean field")
    for state_name, sigma0 in TEST_STATES.items():
        result_m = project_meanfield(
            op_test, sigma0, proj_func=project_operator_to_m_body
        )
        result_n = project_meanfield(
            op_test, sigma0, proj_func=project_to_n_body_operator
        )
        if not check_operator_equality(result_m.to_qutip(), result_n.to_qutip()):
            failed[state_name] = (
                f"Result:\n{result_m}\n\n Result n:\n{result_n}\n\nDelta = \n{result_m.to_qutip()-result_n.to_qutip()}"
            )
            print("*****************", state_name)
            print("m rho_0\n", result_m.to_qutip().ptrace([0]))
            print("m rho_1\n", result_m.to_qutip().ptrace([1]))
            print("n rho_0\n", result_n.to_qutip().ptrace([0]))
            print("n rho_1\n", result_n.to_qutip().ptrace([1]))
            print("*****************")

    if failed:
        for fail in failed:
            print(f" failed with <<{fail}>> as state seed. ")
            print(failed[fail])

        fail_msg = (
            "Meanfield projection 2: Self-consistency failed for some seeds:\n  *"
            + "\n  *".join(key for key in failed)
            + "."
        )
        assert False, fail_msg


@pytest.mark.parametrize(
    ["operator_case", "operator"], list(OPERATOR_TYPE_CASES.items())
)
def test_one_body_from_qutip_operator_1(operator_case, operator):
    """
    Test if the `one_body_from_qutip_operator` function returns
    the right type of operator.
    """
    print(operator_case, "as scalar + one body + rest")
    result = one_body_from_qutip_operator(operator.to_qutip_operator())

    assert check_operator_equality(
        result.to_qutip(), operator.to_qutip()
    ), "operators are not equivalent."
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
        assert (
            abs(last.tr()) < ALPSQUTIP_TOLERANCE
        ), "Reminder term should have zero trace."
        assert (
            abs(terms[-1].tr()) < ALPSQUTIP_TOLERANCE
        ), "One-body term should have zero trace."

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
        ), f"Type of the One-body term {type(one_body)} was not the expected.\n{one_body}"
        assert isinstance(remainder, QutipOperator)
        # Check that the remainder and the one body terms have
        # zero mean:
        if state is None:
            assert abs(one_body.to_qutip().tr()) < ALPSQUTIP_TOLERANCE
            assert abs((remainder.to_qutip()).tr()) < ALPSQUTIP_TOLERANCE
        else:
            error_one_body_tr = abs((one_body.to_qutip() * state.to_qutip()).tr())
            if error_one_body_tr > ALPSQUTIP_TOLERANCE:
                failed.setdefault((operator_name, state_name), {})[
                    "one body tr"
                ] = error_one_body_tr
            remainder_tr = abs((remainder.to_qutip() * state.to_qutip()).tr())
            if remainder_tr > ALPSQUTIP_TOLERANCE:
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


@pytest.mark.parametrize(
    ["operator_case", "operator", "name_ref", "gen"],
    (
        (
            operator_case,
            operator,
            name_ref,
            gen,
        )
        for operator_case, operator in OPERATOR_TYPE_CASES.items()
        for name_ref, gen in PRODUCT_GIBBS_GENERATOR_TESTS.items()
    ),
)
def test_one_body_from_qutip_operator_with_reference_state(
    operator_case, operator, name_ref, gen
):
    sigma = GibbsProductDensityOperator(gen)

    print(operator_case, "as (scalar + one body + rest) w.r.t. " + name_ref)
    result = one_body_from_qutip_operator(operator.to_qutip_operator(), sigma)

    print("operator\n", operator)
    print("operator ->qutip operator\n", operator.to_qutip())
    print("result:\n", result)
    if isinstance(result, SumOperator):
        print("result:\n", result.terms)

    assert check_operator_equality(
        result, operator, 1e-6
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
            abs(sigma.expect(last)) < ALPSQUTIP_TOLERANCE
        ), "Reminder term should have zero mean."
        assert (
            abs(sigma.expect(terms[-1])) < ALPSQUTIP_TOLERANCE
        ), "One-body term should have zero mean."
        # TODO: check also the orthogonality between last and one-body terms.

    assert (
        isinstance(result, (ScalarOperator, OneBodyOperator, LocalOperator))
        for term in terms
    ), "first two terms should be one-body operators"
