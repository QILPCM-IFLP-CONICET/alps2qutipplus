from test.helper import (
    HAMILTONIAN,
    SX_TOTAL,
    SY_TOTAL,
    SZ_TOTAL,
    TEST_CASES_STATES,
    check_equality,
)

import numpy as np

from alpsqutip.evolution import (
    heisenberg_solve,
    qutip_me_solve,
)

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


def test_rabi_evolution():
    rho0 = TEST_CASES_STATES[
        "ProductGibbs from local operator, hermitician"
    ].to_product_state()
    H = SY_TOTAL
    ts = np.linspace(0, 5, 100)
    e_ops = {
        "sxtotal": SX_TOTAL,
        "sytotal": SY_TOTAL,
        "sztotal": SZ_TOTAL,
        "Hamiltonian": HAMILTONIAN,
    }
    result = heisenberg_solve(H, rho0, ts, e_ops=e_ops, deep=4)
    result_qutip = qutip_me_solve(H, rho0, ts, e_ops=e_ops)
    print(len(result["sxtotal"]), len(result_qutip["sxtotal"]))
    check_equality(result, result_qutip, tolerance=1e-5)
