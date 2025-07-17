import os
from test.helper import (
    CHAIN_SIZE,
    HAMILTONIAN,
    SX_A,
    SX_B,
    SX_TOTAL,
    SZ_C,
    TEST_CASES_STATES,
)

import pytest

from alpsqutip.operators.states.meanfield import (
    project_operator_to_m_body,
)
from alpsqutip.operators.states.meanfield.projections import (
    n_body_projector,
)

TEST_STATES = {"None": None}
TEST_OPERATORS = {}
TEST_OPERATORS_SQ = {}

if os.environ.get("BENCHMARKS", 0):
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
            ("n_body_projector", n_body_projector),
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
