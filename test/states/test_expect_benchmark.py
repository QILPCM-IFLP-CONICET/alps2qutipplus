"""
Basic unit test for states.
"""

import os
from test.helper import (
    HAMILTONIAN,
    OBSERVABLE_CASES,
    SX_TOTAL,
    SY_TOTAL,
    SZ_TOTAL,
    TEST_CASES_STATES,
)

import pytest

from alpsqutip.operators.states.gibbs import GibbsDensityOperator
from alpsqutip.operators.states.utils import compute_expectation_values

# from alpsqutip.settings import VERBOSITY_LEVEL

if os.environ.get("BENCHMARKS", 0):
    OBSERVABLE_CASES["sx_total^2"] = SX_TOTAL * SX_TOTAL
    OBSERVABLE_CASES["sz_total^2"] = SZ_TOTAL * SZ_TOTAL
    OBSERVABLE_CASES["HAMILTONIAN^2"] = HAMILTONIAN * HAMILTONIAN
    TOTAL_SPIN_COMPONENTS = [SX_TOTAL, SY_TOTAL, SZ_TOTAL]
    OBSERVABLE_CASES["total components of the spin"] = TOTAL_SPIN_COMPONENTS
    OBSERVABLE_CASES["global quadratic list"] = TOTAL_SPIN_COMPONENTS + [
        x * x for x in TOTAL_SPIN_COMPONENTS
    ]
    TEST_CASES_STATES["mixture"] = (
        0.6 * TEST_CASES_STATES["x semipolarized"]
        + 0.3 * TEST_CASES_STATES["first full polarized"]
        + 0.1 * TEST_CASES_STATES["gibbs_sz_bar"]
    )
    BENCHMARK_EXPECTATION_CASES = [
            (
                name_rho,
                name_obs,
            )
            for name_rho in TEST_CASES_STATES
            for name_obs in OBSERVABLE_CASES
        ]
else:
    BENCHMARK_EXPECTATION_CASES = []


@pytest.mark.skipif(
    not os.environ.get("BENCHMARKS", 0), reason="run only in benchmarks"
)
@pytest.mark.parametrize(["name_rho", "name_obs"], BENCHMARK_EXPECTATION_CASES)
def test_benchmark_expect(benchmark, name_rho, name_obs):

    rho = TEST_CASES_STATES[name_rho]
    obs = OBSERVABLE_CASES[name_obs]
    print(type(rho), type(obs))
    if isinstance(rho, (GibbsDensityOperator,)) and len(rho.system.sites) > 8:
        return

    if rho is None:

        def impl():
            return compute_expectation_values(obs, rho)

    else:

        def impl():
            return rho.expect(obs)

    benchmark.pedantic(impl, rounds=3, iterations=1)
