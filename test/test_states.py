"""
Basic unit test for states.
"""

from alpsqutip.operators import OneBodyOperator

from .helper import (
    alert,
    check_equality,
    expect_from_qutip,
    observable_cases,
    subsystems,
    sz_total,
    test_cases_states,
)

# from alpsqutip.settings import VERBOSITY_LEVEL


def test_mixtures():
    qt_test_cases = {
        name: operator.to_qutip() for name, operator in test_cases_states.items()
    }

    for name_rho, rho in test_cases_states.items():
        for name_sigma, sigma in test_cases_states.items():
            print(".3*", name_rho, " + .7 * ", name_sigma)
            mixture = 0.3 * rho + 0.7 * sigma
            qutip_mixture = (
                0.3 * qt_test_cases[name_rho] + 0.7 * qt_test_cases[name_sigma]
            )
            assert check_equality(rho.tr(), 1)
            assert check_equality(sigma.tr(), 1)
            assert check_equality(mixture.tr(), 1)
            assert check_equality(qutip_mixture.tr(), 1)
            check_equality(mixture.to_qutip(), qutip_mixture)


def test_states():
    """Tests for state objects"""
    # enumerate the name of each subsystem
    print(80 * "=", "\n")
    print("test states")
    print(80 * "=", "\n")
    assert isinstance(sz_total, OneBodyOperator)
    qt_test_cases = {
        name: operator.to_qutip() for name, operator in test_cases_states.items()
    }

    for name, rho in test_cases_states.items():
        # TODO: Remove me
        if name != "mixture":
            continue

        print(
            "\n     ", 120 * "@", "\n testing", name, f"({type(rho)})", "\n", 100 * "@"
        )
        assert abs(rho.tr() - 1) < 1.0e-10, "la traza de rho no es 1"
        assert (
            abs(1 - qt_test_cases[name].tr()) < 1.0e-10
        ), "la traza de rho.qutip no es 1"

        for subsystem in subsystems:
            print("   subsystem", subsystem)
            local_rho = rho.partial_trace(subsystem)
            print(" type", local_rho)
            assert check_equality(
                local_rho.tr(), 1
            ), "la traza del operador local no es 1"

        # Check Expectation Values
        print(" ??????????????? testing expectation values")
        print(rho.expect)
        expectation_values = rho.expect(observable_cases)
        qt_expectation_values = expect_from_qutip(qt_test_cases[name], observable_cases)

        assert isinstance(expectation_values, dict)
        assert isinstance(qt_expectation_values, dict)
        for obs in expectation_values:
            alert(0, "\n     ", 80 * "*", "\n     ", name, " over ", obs)
            alert(0, "Native", expectation_values)
            alert(0, "QTip", qt_expectation_values)
            assert check_equality(
                expectation_values[obs], qt_expectation_values[obs]
            ), f"the expectation value for the observable{obs} do not match."


# test_load()
# test_all()
# test_eval_expr()
