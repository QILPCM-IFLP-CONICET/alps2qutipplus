import os
from test.helper import (
    CHAIN_SIZE,
    HAMILTONIAN,
    SITES,
    SX_A,
    SX_B,
    SX_TOTAL,
    SYSTEM,
    SZ_C,
    TEST_CASES_STATES,
)

import pytest

from alpsqutip.operators import ProductOperator
from alpsqutip.operators.states.meanfield import (
    project_operator_to_m_body,
)
from alpsqutip.operators.states.meanfield.projections import (  # Product; Qutip; One body
    _project_product_operator_to_m_body_recursive,
    _project_product_operator_to_one_body,
    _project_qutip_operator_to_m_body_recursive,
    project_product_operator_as_n_body_operator,
    project_qutip_operator_as_n_body_operator,
    project_qutip_to_one_body,
    project_to_n_body_operator,
)

TEST_STATES = {"None": None}
TEST_OPERATORS = {}
TEST_OPERATORS_SQ = {}
TEST_SINGLE_TERMS = {}

if os.environ.get("BENCHMARKS", 0):
    TEST_SINGLE_TERMS.update(
        {
            f"product_{i}": ProductOperator(
                {SITES[k]: SX_A.operator for k in range(i + 1)}, 1, SYSTEM
            )
            for i in range(8)
        }
    )
    TEST_SINGLE_TERMS.update(
        {f"{key}_qutip": op.to_qutip_operator for key, op in TEST_SINGLE_TERMS.items()}
    )
    for i in range(1, 7):
        TEST_SINGLE_TERMS[f"complex_{i+1}"] = sum(
            sum(
                SYSTEM.site_operator(op_name, SITES[k])
                * SYSTEM.site_operator(op_name, SITES[k + 1])
                for op_name in ["Sx", "Sy", "Sz"]
            )
            for k in range(i)
        ).to_qutip_operator()

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

    TEST_OPERATORS.update(
        {
            "sx_total": SX_TOTAL,
            "-sx_total - sx_total^2/(N-1)": (
                -SX_TOTAL - SX_TOTAL * SX_TOTAL / (CHAIN_SIZE - 1)
            ),
            "sx_A*sx_B": SX_A * SX_B,
            "Hamiltonian": HAMILTONIAN,
            "sx_A*sx_B*sz_C+ sx_A * sx_B": SX_A * SX_B * SZ_C + SX_A * SX_B,
        }
    )
    TEST_OPERATORS_SQ = {
        key: (op * op).simplify() for key, op in TEST_OPERATORS.items()
    }


@pytest.mark.parametrize(
    [
        "name",
        "state_name",
    ],
    [
        (name, state_name)
        for name in TEST_SINGLE_TERMS
        for state_name, sigma0 in TEST_STATES.items()
        if hasattr(TEST_SINGLE_TERMS[name], "sites_op")
    ],
)
def test_one_body_product_projector(benchmark, name, state_name):
    operator = TEST_SINGLE_TERMS[name]
    state = TEST_STATES[state_name]

    def impl():
        return _project_product_operator_to_one_body(operator, state)

    benchmark.pedantic(impl, rounds=3, iterations=1)


@pytest.mark.parametrize(
    [
        "name",
        "state_name",
    ],
    [
        (name, state_name)
        for name in TEST_SINGLE_TERMS
        for state_name, sigma0 in TEST_STATES.items()
        if hasattr(TEST_SINGLE_TERMS[name], "site_names")
    ],
)
def test_one_body_qutip_projector(benchmark, name, state_name):
    operator = TEST_SINGLE_TERMS[name]
    state = TEST_STATES[state_name]

    def impl():
        return project_qutip_to_one_body(operator, state)

    benchmark.pedantic(impl, rounds=3, iterations=1)


@pytest.mark.parametrize(
    [
        "name",
        "state_name",
        "projector",
        "nbody",
    ],
    [
        (name, state_name, projector, nbody)
        for name in TEST_SINGLE_TERMS
        for state_name, sigma0 in TEST_STATES.items()
        for projector in (
            "_project_product_operator_to_m_body_recursive",
            "project_product_operator_as_n_body_operator",
        )
        for nbody in range(8)
        if hasattr(TEST_SINGLE_TERMS[name], "sites_op")
    ],
)
def test_product_projectors(benchmark, name, state_name, projector, nbody):
    operator = TEST_SINGLE_TERMS[name]
    state = TEST_STATES[state_name]

    if projector == "_project_product_operator_to_m_body_recursive":

        def impl():
            return _project_product_operator_to_m_body_recursive(operator, nbody, state)

    else:

        def impl():
            return project_product_operator_as_n_body_operator(operator, nbody, state)

    benchmark.pedantic(impl, rounds=3, iterations=1)


@pytest.mark.parametrize(
    [
        "name",
        "state_name",
        "projector",
        "nbody",
    ],
    [
        (name, state_name, projector, nbody)
        for name in TEST_SINGLE_TERMS
        for state_name, sigma0 in TEST_STATES.items()
        for projector in (
            "_project_qutip_operator_to_m_body_recursive",
            "project_qutip_operator_as_n_body_operator",
        )
        for nbody in range(8)
        if hasattr(TEST_SINGLE_TERMS[name], "site_names")
    ],
)
def test_qutip_projectors(benchmark, name, state_name, projector, nbody):
    operator = TEST_SINGLE_TERMS[name]
    state = TEST_STATES[state_name]

    if projector == "_project_qutip_operator_to_m_body_recursive":

        def impl():
            return _project_qutip_operator_to_m_body_recursive(operator, nbody, state)

    else:

        def impl():
            return project_qutip_operator_as_n_body_operator(operator, nbody, state)

    benchmark.pedantic(impl, rounds=3, iterations=1)


@pytest.mark.parametrize(
    [
        "op_name",
        "projection_name",
        "projection_function",
        "state_name",
        "nbody",
        "sigma0",
    ],
    [
        (name, proj_name, proj_func, state_name, nbody, sigma0)
        for nbody in range(4)
        for name in TEST_OPERATORS
        for state_name, sigma0 in TEST_STATES.items()
        for proj_name, proj_func in (
            ("project_to_n_body_operator", project_to_n_body_operator),
            ("project_operator_to_m_body", project_operator_to_m_body),
        )
    ],
)
def test_benchmark_nbody_projection(
    benchmark, op_name, projection_name, projection_function, state_name, nbody, sigma0
):
    """Test the mean field projection over different states,
    and using both implementations"""
    print("testing the consistency of projection in", op_name)
    op_sq = TEST_OPERATORS_SQ[op_name]

    def impl():
        return projection_function(op_sq, nbody, sigma0)

    result = benchmark.pedantic(impl, rounds=3, iterations=1)

    if sigma0 is None:
        eval_orig, eval_proj = (
            op_sq.tr(),
            result.tr(),
        )
    else:
        eval_orig, eval_proj = sigma0.expect([op_sq, result])

    assert abs(eval_orig - eval_proj) < 1.0e-6
