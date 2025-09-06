"""
Test functions that implement the mean field approximation.
"""

from test.helper import (
    GLOBAL_IDENTITY,
    HAMILTONIAN,
    SX_A,
    SX_TOTAL,
    SY_TOTAL,
    SZ_A,
    SZ_TOTAL,
    TEST_CASES_STATES,
    check_equality,
    check_operator_equality,
)

import pytest

from alpsqutip.projections.symmetries import (
    project_conserved_quantity,
    project_parity_like,
)
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

TEST_STATES = {}

TEST_STATES.update(
    {
        name: TEST_CASES_STATES[name]
        for name in (
            "z semipolarized",
            "x semipolarized",
            "mixture of first and second partially polarized",
        )
    }
)


def test_parity_projection():
    observables = {
        "identity": GLOBAL_IDENTITY,
        "sx_A": SX_A,
        "sz_A": SZ_A,
        "sx_total": SX_TOTAL,
        "sy_total": SY_TOTAL,
        "sz_total": SZ_TOTAL,
        "hamiltonian": HAMILTONIAN,
    }

    for state_name, state in TEST_STATES.items():
        pre_averages = state.expect(observables)
        projected_state = project_parity_like(state, "Parity")
        expected_averages = pre_averages.copy()
        expected_averages["sx_total"] = 0.0
        expected_averages["sy_total"] = 0.0
        expected_averages["sx_A"] = 0.0
        projected_averages = projected_state.expect(observables)
        assert check_equality(projected_averages, expected_averages)

        projected_state = project_parity_like(state, "ParityX")
        expected_averages = pre_averages.copy()
        expected_averages["sz_total"] = 0.0
        expected_averages["sy_total"] = 0.0
        expected_averages["sz_A"] = 0.0
        projected_averages = projected_state.expect(observables)
        assert check_equality(projected_averages, expected_averages)

        projected_state = project_conserved_quantity(state, "Sx")
        assert check_operator_equality(
            SX_TOTAL * projected_state, projected_state * SX_TOTAL, 1e-6
        )
        expected_averages = pre_averages.copy()
        expected_averages["sz_total"] = 0.0
        expected_averages["sy_total"] = 0.0
        expected_averages["sz_A"] = 0.0
        projected_averages = projected_state.expect(observables)

        projected_state = project_conserved_quantity(state, "Sz")
        assert check_operator_equality(
            SZ_TOTAL * projected_state, projected_state * SZ_TOTAL, 1e-6
        )
        expected_averages = pre_averages.copy()
        expected_averages["sx_total"] = 0.0
        expected_averages["sy_total"] = 0.0
        expected_averages["sx_A"] = 0.0
        projected_averages = projected_state.expect(observables)


@pytest.mark.parametrize(["name"], ((name,) for name in TEST_CASES_STATES))
def test_symmetry_compatibility(name):
    state = TEST_CASES_STATES[name]
    symmetries = getattr(state, "symmetry_projections", None)
    if not symmetries:
        return
    state_qutip = state.to_qutip_operator()
    assert check_operator_equality(state, state_qutip, ALPSQUTIP_TOLERANCE)
    for symm in symmetries:
        print("checking ", symm, "on", name)
        assert check_operator_equality(symm(state), state_qutip, ALPSQUTIP_TOLERANCE)
