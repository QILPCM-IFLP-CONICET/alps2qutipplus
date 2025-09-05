"""
Test functions that implement the mean field approximation.
"""

from test.helper import CHAIN_SIZE, SX_A, SX_B, SX_TOTAL, SYSTEM, TEST_CASES_STATES

import numpy as np
import pytest

from alpsqutip import (
    graph_from_alps_xml,
    model_from_alps_xml,
)
from alpsqutip.meanfield import (
    variational_quadratic_mfa,
)
from alpsqutip.model import SystemDescriptor
from alpsqutip.operators import ScalarOperator
from alpsqutip.operators.states import (
    GibbsProductDensityOperator,
    ProductDensityOperator,
)
from alpsqutip.projections import n_body_projection
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
    "sx_total - sx_total^2/(N-1)": (SX_TOTAL + SX_TOTAL * SX_TOTAL / (CHAIN_SIZE - 1)),
    "sx_A*sx_B": SX_A * SX_B,
}


# TODO: Study why the convergency fails for these cases.
SKIP_MEANFIELD_SEEDS = {
    "sx_total - sx_total^2/(N-1)": [],
    "sx_A*sx_B": [],  # "x semipolarized"
}

EXPECTED_PROJECTIONS = {}
# sx_total is not modified
EXPECTED_PROJECTIONS["sx_total"] = {name: SX_TOTAL for name in TEST_STATES}

# TODO: build this analytically
SX_MF_AV = 0.5 * 1.0757657
EXPECTED_PROJECTIONS["sx_total - sx_total^2/(N-1)"] = {
    name: (SX_TOTAL * SX_MF_AV + (0.1197810663) * 3 / 4 * CHAIN_SIZE / (CHAIN_SIZE - 1))
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
def test_variational_meanfield(state_name, state, generator_name, generator):
    print(
        "Check that the variational mean field for ",
        generator_name,
        "of type",
        type(generator),
        "converges to a self-consistent " "state starting from",
        state_name,
        "of type",
        type(state),
    )

    sigma_var = variational_quadratic_mfa(
        generator, sigma_ref=state, max_self_consistent_steps=100
    )
    generator_1b_1st = n_body_projection(generator, 1, sigma_var)
    sigma_sc = GibbsProductDensityOperator(generator_1b_1st)
    rel_entropy_var = sigma_var.expect(sigma_var.logm() + generator)
    rel_entropy_sc = sigma_var.expect(sigma_sc.logm() + generator)
    assert (
        abs(rel_entropy_var - rel_entropy_sc) < ALPSQUTIP_TOLERANCE**0.5
    ), f"{rel_entropy_var}!={rel_entropy_sc}"


def test_mf_for_hamiltonians_with_loop_terms():
    latt_descr = graph_from_alps_xml(
        name="square lattice with loop", parms={"L": 3, "a": 1}
    )
    model_descr = model_from_alps_xml(name="spin loop")
    system = SystemDescriptor(
        latt_descr,
        model_descr,
        {
            "Jz": -1,
            "Jxy": -1,
            "Jz2": -1,
            "Jxy2": -1,
            "Jz3": -1,
            "Jxy3": -1,
            "Jz4": -1,
            "Jxy4": -1,
            "Wilson": 10,
        },
    )
    hamiltonian = system.global_operator("Hamiltonian") + system.global_operator("Sz")
    sigma = variational_quadratic_mfa(
        hamiltonian, numfields=3, max_self_consistent_steps=30
    )
    rel_entropy = np.real(sigma.expect(hamiltonian - sigma.logm()))
    print(rel_entropy)
    assert rel_entropy < -1.9
