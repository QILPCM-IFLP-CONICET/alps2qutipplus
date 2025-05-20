"""
Test functions that implement the mean field approximation.
"""
import pytest

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
    variational_quadratic_mfa,
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
    "sx_total": sx_total,
    "sx_total - sx_total^2/(N-1)": (sx_total + sx_total * sx_total / (CHAIN_SIZE - 1)),
    "sx_A*sx_B": sx_A * sx_B,
}


# TODO: Study why the convergency fails for these cases.
SKIP_MEANFIELD_SEEDS = {
    "sx_total - sx_total^2/(N-1)": [],
    "sx_A*sx_B": [], # "x semipolarized"
}

EXPECTED_PROJECTIONS = {}
# sx_total is not modified
EXPECTED_PROJECTIONS["sx_total"] = {name: sx_total for name in TEST_STATES}

# TODO: build this analytically
SX_MF_AV = 0.5 * 1.0757657
EXPECTED_PROJECTIONS["sx_total - sx_total^2/(N-1)"] = {
    name: (sx_total * SX_MF_AV + (0.1197810663) * 3 / 4 * CHAIN_SIZE / (CHAIN_SIZE - 1))
    for name in TEST_STATES
}
EXPECTED_PROJECTIONS["sx_A*sx_B"] = {
    name: ScalarOperator(0, SYSTEM) for name in TEST_STATES
}


@pytest.mark.parametrize(
    ["state_name", "state", "generator_name", "generator"],
    [
        (state_name, state, generator_name, generator)
        for state_name, state in TEST_CASES_STATES.items()
        for generator_name, generator in TEST_OPERATORS.items()
        if isinstance(state, (GibbsProductDensityOperator, ProductDensityOperator))
    ],
)
def test_variational_meanfield(
        state_name, state, generator_name, generator
):
    print(
        "Check that the variational mean field for ",
        generator_name, "of type", type(generator),
        "converges to a self-consistent "
        "state starting from",
        state_name, "of type", type(state)
    )

    sigma = variational_quadratic_mfa(generator, sigma_ref=state)
