"""
Basic unit test for states.
"""

from test.helper import (
    OBSERVABLE_CASES,
    SUBSYSTEMS,
    SZ_TOTAL,
    TEST_CASES_STATES,
    alert,
    check_equality,
    expect_from_qutip,
)

import pytest

from alpsqutip.operators import OneBodyOperator

# from alpsqutip.settings import VERBOSITY_LEVEL

QT_TEST_CASES = {
    name: operator.to_qutip() for name, operator in TEST_CASES_STATES.items()
}


@pytest.mark.parametrize(
    ["name_rho", "rho", "name_sigma", "sigma"],
    [
        (
            name_rho,
            rho,
            name_sigma,
            sigma,
        )
        for name_rho, rho in TEST_CASES_STATES.items()
        for name_sigma, sigma in TEST_CASES_STATES.items()
    ],
)
def test_mixtures(name_rho, rho, name_sigma, sigma):
    rho_coeff = 0.99
    sigma_coeff = 0.01

    print(
        f"{rho_coeff}*",
        name_rho,
        f"[{type(rho)}] + {sigma_coeff} * ",
        name_sigma,
        f"[{type(sigma)}]",
    )
    mixture = rho_coeff * rho + sigma_coeff * sigma
    qutip_mixture = (
        rho_coeff * QT_TEST_CASES[name_rho] + sigma_coeff * QT_TEST_CASES[name_sigma]
    )
    assert check_equality(rho.tr(), 1)
    assert check_equality(sigma.tr(), 1)
    assert check_equality(mixture.tr(), 1)
    assert check_equality(qutip_mixture.tr(), 1)
    print("mixture:\n", mixture)
    print("qutip mixture:\n", qutip_mixture)
    check_equality(mixture.to_qutip(), qutip_mixture)


@pytest.mark.parametrize(["name", "rho"], list(TEST_CASES_STATES.items()))
def test_states(name, rho):
    """Tests for state objects"""
    # enumerate the name of each subsystem
    print(80 * "=", "\n")
    print("test states")
    print(80 * "=", "\n")
    assert isinstance(SZ_TOTAL, OneBodyOperator)

    print("\n     ", 120 * "@", "\n testing", name, f"({type(rho)})", "\n", 100 * "@")
    assert abs(rho.tr() - 1) < 1.0e-10, "la traza de rho no es 1"
    assert abs(1 - QT_TEST_CASES[name].tr()) < 1.0e-10, "la traza de rho.qutip no es 1"

    for subsystem in SUBSYSTEMS:
        print("   subsystem", subsystem)
        local_rho = rho.partial_trace(frozenset(subsystem))
        print(" type", local_rho)
        assert check_equality(local_rho.tr(), 1), "la traza del operador local no es 1"

    # Check Expectation Values
    print(" ??????????????? testing expectation values")
    print(rho.expect)
    expectation_values = rho.expect(OBSERVABLE_CASES)
    qt_expectation_values = expect_from_qutip(QT_TEST_CASES[name], OBSERVABLE_CASES)

    assert isinstance(expectation_values, dict)
    assert isinstance(qt_expectation_values, dict)
    for obs in expectation_values:
        alert(0, "\n     ", 80 * "*", "\n     ", name, " over ", obs)
        alert(0, "Native", expectation_values)
        alert(0, "QTip", qt_expectation_values)
        try:
            assert check_equality(expectation_values[obs], qt_expectation_values[obs])
        except AssertionError:
            assert (
                False
            ), f"the expectation value for the observable{obs} relative to {name} do not match ((result)={expectation_values[obs]} !=  {qt_expectation_values[obs]} (qutip))."


# test_load()
# test_all()
# test_eval_expr()
