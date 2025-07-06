from test.helper import (
    GIBBS_GENERATOR_TESTS,
    HAMILTONIAN,
    OPERATOR_TYPE_CASES,
    SX_AB,
    SX_TOTAL,
    SZ_TOTAL,
    SY_TOTAL,
    TEST_CASES_STATES,
    check_equality,
    check_operator_equality,
)

import numpy as np
import pytest
import qutip

from alpsqutip.evolution import (
    build_hierarchical_basis,
    fn_hij_tensor_with_errors,
    projected_evolution,
    series_evolution,
    heisenberg_solve,
    qutip_me_solve,
)
from alpsqutip.operators.functions import commutator, spectral_norm
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.scalarprod import fetch_covar_scalar_product, orthogonalize_basis

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)




def test_rabi_evolution():
    rho0 = TEST_CASES_STATES["ProductGibbs from local operator, hermitician"].to_product_state()
    H = SY_TOTAL
    ts= np.linspace(0,5,100)
    e_ops={"sxtotal":SX_TOTAL, "sytotal":SY_TOTAL, "sztotal":SZ_TOTAL, "Hamiltonian": HAMILTONIAN}
    result = heisenberg_solve(H, rho0, ts, e_ops=e_ops, deep=4)
    result_qutip = qutip_me_solve(H, rho0, ts, e_ops=e_ops)
    print(len(result["sxtotal"]), len(result_qutip["sxtotal"]))
    check_equality(result, result_qutip)
    
