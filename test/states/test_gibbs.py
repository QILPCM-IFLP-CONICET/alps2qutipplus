"""
Basic unit test for states.
"""

from test.helper import (
    GIBBS_GENERATOR_TESTS,
    OPERATOR_TYPE_CASES,
    PRODUCT_GIBBS_GENERATOR_TESTS,
    SITES,
    check_equality,
    check_operator_equality,
)

import pytest

from alpsqutip.operators import ProductOperator, QutipOperator
from alpsqutip.operators.states import (
    DensityOperatorMixin,
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    QutipDensityOperator,
)
from alpsqutip.operators.states.utils import safe_exp_and_normalize

# from alpsqutip.settings import VERBOSITY_LEVEL


def do_test_expect(rho, sigma_dict):
    """Compare expectation values"""

    for obs_name, obs_op in OPERATOR_TYPE_CASES.items():
        rho_obs_expect = rho.expect(obs_op)
        for name, sigma in sigma_dict.items():
            if sigma is None:
                continue

            if hasattr(sigma, "data"):
                rho_obs_result = (obs_op.to_qutip() * sigma).tr()
            else:
                rho_obs_result = sigma.expect(obs_op)

            assert abs(rho_obs_expect - rho_obs_result) < 1e-10, (
                f"Operator {obs_name} gives different results for "
                f"the reference (={rho_obs_expect})"
                f" and {name} (={rho_obs_result}). "
                f"Delta={rho_obs_expect-rho_obs_result}"
            )


def do_test_instance(rho) -> None:
    """

    Parameters
    ----------
    rho : GibbsDensityOperator |GibbsProductDensityOperator
        the state over which implement the tests.

    """
    rho_times_two = 2 * rho
    rho_qutip = rho.to_qutip()
    assert abs(rho.tr() - 1) < 1e-10
    assert abs(rho_qutip.tr() - 1) < 1e-10
    rho_0 = rho.partial_trace(frozenset({SITES[0]}))
    print("rho_0", type(rho_0))
    assert isinstance(rho_0, DensityOperatorMixin)
    check_equality(rho_0.to_qutip(), rho_qutip.ptrace([0]))
    assert isinstance(rho.to_qutip_operator(), QutipDensityOperator)
    rhosq = rho * rho
    assert rhosq.tr() <= 1
    assert isinstance(rhosq, (QutipOperator, ProductOperator))
    assert not isinstance(rhosq, DensityOperatorMixin)

    rho_ps = None
    if hasattr(rho, "to_product_state"):
        rho_ps = rho.to_product_state()
        assert abs(rho_ps.tr() - 1) < 1.0e-10

    do_test_expect(
        rho, {"rho_qutip": rho_qutip, "rho_ps": rho_ps, "rho * 2": rho_times_two}
    )


@pytest.mark.parametrize(("key", "k_gen"), list(GIBBS_GENERATOR_TESTS.items()))
def test_gibbs(key, k_gen):
    """Test GibbsDensityOperator"""
    if k_gen is None:
        return

    print(f"test gibbs state from {key}({type(k_gen)})")
    rho = GibbsDensityOperator(k_gen)
    do_test_instance(rho)
    do_test_log(rho)


@pytest.mark.parametrize(("key", "k_gen"), list(PRODUCT_GIBBS_GENERATOR_TESTS.items()))
def test_product_gibbs(key, k_gen):
    """Test GibbsProductDensityOperator"""
    if k_gen is None:
        return

    print(f"test gibbs state from {key} ({type(k_gen)})")
    rho = GibbsProductDensityOperator(k_gen)
    do_test_instance(rho)
    do_test_log(rho)


def do_test_log(rho):
    """Test the logm method."""
    rho_qutip = rho.to_qutip()
    ln_rho_neg = rho.logm()
    ln_rho_neg_qutip = ln_rho_neg.to_qutip()
    rho_g, ln_z = safe_exp_and_normalize(ln_rho_neg_qutip)
    assert abs(ln_z) < 1e-6
    check_operator_equality(rho_qutip, rho_g)
    return rho_g
