"""
Helper functions for pytests
"""

from numbers import Number
from typing import Iterable

import numpy as np
import qutip

from alpsqutip.model import SystemDescriptor, build_spin_chain
from alpsqutip.operators import (
    OneBodyOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.quadratic import build_quadratic_form_from_operator
from alpsqutip.operators.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    ProductDensityOperator,
)
from alpsqutip.settings import VERBOSITY_LEVEL

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)

CHAIN_SIZE = 4

SYSTEM: SystemDescriptor = build_spin_chain(CHAIN_SIZE)
SITES: tuple = tuple(s for s in SYSTEM.sites.keys())


global_identity: ScalarOperator = ScalarOperator(1.0, SYSTEM)


sx_A = SYSTEM.site_operator(f"Sx@{SITES[0]}")
sx_B = SYSTEM.site_operator(f"Sx@{SITES[1]}")
sx_AB = 0.7 * sx_A + 0.3 * sx_B


sy_A = SYSTEM.site_operator(f"Sy@{SITES[0]}")
sy_B = SYSTEM.site_operator(f"Sy@{SITES[1]}")

assert isinstance(sx_A * sx_B + sy_A * sy_B, Operator)


splus_A = SYSTEM.site_operator(f"Splus@{SITES[0]}")
splus_B = SYSTEM.site_operator(f"Splus@{SITES[1]}")
sminus_A = SYSTEM.site_operator(f"Sminus@{SITES[0]}")
sminus_B = SYSTEM.site_operator(f"Sminus@{SITES[1]}")


sz_A = SYSTEM.site_operator(f"Sz@{SITES[0]}")
sz_B = SYSTEM.site_operator(f"Sz@{SITES[1]}")
sz_C = SYSTEM.site_operator(f"Sz@{SITES[2]}")
sz_AB = 0.7 * sz_A + 0.3 * sz_B


sh_A = 0.25 * sx_A + 0.5 * sz_A
sh_B = 0.25 * sx_B + 0.5 * sz_B
sh_AB = 0.7 * sh_A + 0.3 * sh_B


sz_total: OneBodyOperator = SYSTEM.global_operator("Sz")
assert isinstance(sz_total, OneBodyOperator)

sx_total: OneBodyOperator = sum(SYSTEM.site_operator("Sx", s) for s in SITES)
sy_total: OneBodyOperator = sum(SYSTEM.site_operator("Sy", s) for s in SITES)
hamiltonian: SumOperator = SYSTEM.global_operator("Hamiltonian")

assert hamiltonian is not None

assert (sminus_A * sminus_B) is not None


splus0 = SYSTEM.site_operator(f"Splus@{SITES[0]}")
splus1 = SYSTEM.site_operator(f"Splus@{SITES[1]}")

spsp_hc = SumOperator(
    (
        splus0 * splus1,
        (splus0 * splus1).dag(),
    ),
    SYSTEM,
    True,
)


OPERATORS = {
    "Identity": global_identity,
    "sz_total": sz_total,
    "sx_total": sx_total,
    "sx_total_sq": sx_total * sx_total,
    "sx+ 3j sz": sx_total + (3 * 1j) * sz_total,
    "splus*splus": splus0 * splus1,
    "splus*splus+hc": spsp_hc,
    "hamiltonian": hamiltonian,
    "nonhermitician": hamiltonian + (3 * 1j) * sz_total,
}


SUBSYSTEMS = [
    (SITES[0],),
    (SITES[1],),
    (SITES[2],),
    (
        SITES[0],
        SITES[1],
    ),
    (
        SITES[0],
        SITES[2],
    ),
    (
        SITES[2],
        SITES[3],
    ),
]


OBSERVABLE_CASES = {
    "Identity": ScalarOperator(1.0, SYSTEM),
    "sz_total": sz_total,  # OneBodyOperator
    "sx_A": sx_A,  # LocalOperator
    "sy_A": sy_A,  # Local Operator
    "sz_B": sz_B,  # Diagonal local operator
    "sh_AB": sh_AB,  # ProductOperator
    "exchange_AB": sx_A * sx_B + sy_A * sy_B,  # Sum operator
    "hamiltonian": hamiltonian,  # Sum operator, hermitician
    "observable array": [[sh_AB, sh_A], [sz_A, sx_A]],
}


OPERATOR_TYPE_CASES = {
    "scalar, zero": ScalarOperator(0.0, SYSTEM),
    "product, zero": ProductOperator({}, prefactor=0.0, system=SYSTEM),
    "product, 1": ProductOperator({}, prefactor=1.0, system=SYSTEM),
    "scalar, real": ScalarOperator(1.0, SYSTEM),
    "scalar, complex": ScalarOperator(1.0 + 3j, SYSTEM),
    "local operator, hermitician": sx_A,  # LocalOperator
    "local operator, non hermitician": sx_A + sy_A * 1j,
    "One body, hermitician": sz_total,
    "One body, non hermitician": sx_total + sy_total * 1j,
    "three body, hermitician": (sx_A * sy_B * sz_C),
    "three body, non hermitician": ((sminus_A * sminus_B + sy_A * sy_B) * sz_total),
    "product operator, hermitician": sh_AB,
    "product operator, non hermitician": sminus_A * splus_B,
    "sum operator, hermitician": sx_A * sx_B + sy_A * sy_B,  # Sum operator
    "sum operator, hermitician from non hermitician": splus_A * splus_B
    + sminus_A * sminus_B,
    "sum operator, anti-hermitician": splus_A * splus_B - sminus_A * sminus_B,
    "sum local operators": splus_A + sminus_A,
    "sum local qutip operators": 2.0 * splus_A.to_qutip_operator()
    + sminus_A.to_qutip_operator() * 2.0,
    "sum local qutip operator and local operator": (
        2.0 * splus_A.to_qutip_operator()
        + sminus_A * 2.0
        + splus_B.to_qutip_operator() * 2
        + 2 * sminus_B
    ),
    "sum two-body qutip operators": 0.25
    * (splus_A.to_qutip_operator() * splus_B.to_qutip_operator())
    + (sminus_A * sminus_B) * 0.25,
    "qutip operator": hamiltonian.to_qutip_operator(),
    "hermitician quadratic operator": build_quadratic_form_from_operator(hamiltonian),
    "non hermitician quadratic operator": build_quadratic_form_from_operator(
        hamiltonian - sz_total * 1j
    ),
    "log unitary": build_quadratic_form_from_operator(hamiltonian * 1j),
    "single interaction term": build_quadratic_form_from_operator(sx_A * sx_B),
}


TEST_CASES_STATES = {}

TEST_CASES_STATES["fully mixed"] = ProductDensityOperator({}, system=SYSTEM)

TEST_CASES_STATES["z semipolarized"] = ProductDensityOperator(
    {name: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmaz() for name in SYSTEM.dimensions},
    1.0,
    system=SYSTEM,
)

TEST_CASES_STATES["x semipolarized"] = ProductDensityOperator(
    {name: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmax() for name in SYSTEM.dimensions},
    1.0,
    system=SYSTEM,
)


TEST_CASES_STATES["first full polarized"] = ProductDensityOperator(
    {sx_A.site: 0.5 * qutip.qeye(2) + 0.5 * qutip.sigmaz()}, 1.0, system=SYSTEM
)

TEST_CASES_STATES[
    "mixture of first and second partially polarized"
] = 0.5 * ProductDensityOperator(
    {sx_A.site: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmaz()}, 1.0, system=SYSTEM
) + 0.5 * ProductDensityOperator(
    {sx_B.site: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmaz()}, 1.0, system=SYSTEM
)


TEST_CASES_STATES["gibbs_sz"] = GibbsProductDensityOperator(sz_total, system=SYSTEM)

TEST_CASES_STATES["gibbs_sz_as_product"] = GibbsProductDensityOperator(
    sz_total, system=SYSTEM
).to_product_state()
TEST_CASES_STATES["gibbs_sz_bar"] = GibbsProductDensityOperator(
    sz_total * (-1), system=SYSTEM
)
TEST_CASES_STATES["gibbs_H"] = GibbsDensityOperator(hamiltonian, system=SYSTEM)
TEST_CASES_STATES["gibbs_H"] = (
    TEST_CASES_STATES["gibbs_H"] / TEST_CASES_STATES["gibbs_H"].tr()
)
TEST_CASES_STATES["mixture"] = (
    0.5 * TEST_CASES_STATES["gibbs_H"]
    + 0.25 * TEST_CASES_STATES["gibbs_sz"]
    + 0.25 * TEST_CASES_STATES["gibbs_sz_bar"]
)


FULL_TEST_CASES = {}
FULL_TEST_CASES.update(OPERATOR_TYPE_CASES)
FULL_TEST_CASES.update(TEST_CASES_STATES)


def alert(verbosity, *args):
    """Print a message depending on the verbosity level"""
    if verbosity < VERBOSITY_LEVEL:
        print(*args)


def check_equality(lhs, rhs):
    """
    Compare lhs and rhs and raise an assertion error if they are
    different.
    """
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        assert abs(lhs - rhs) < 1.0e-10, f"{lhs}!={rhs} + O(10^-10)"
        return True

    if isinstance(lhs, Operator) and isinstance(rhs, Operator):
        assert check_operator_equality(lhs, rhs)
        return True

    if isinstance(lhs, dict) and isinstance(rhs, dict):
        assert len(lhs) == rhs
        assert all(key in rhs for key in lhs)
        assert all(check_equality(lhs[key], rhs[key]) for key in lhs)
        return True

    if isinstance(lhs, np.ndarray) and isinstance(rhs, np.ndarray):
        diff = abs(lhs - rhs)
        assert (diff < 1.0e-10).all()
        return True

    if isinstance(lhs, Iterable) and isinstance(rhs, Iterable):
        assert len(lhs) != len(rhs)
        assert all(
            check_equality(lhs_item, rhs_item) for lhs_item, rhs_item in zip(lhs, rhs)
        )
        return True

    assert lhs == rhs
    return True


def check_operator_equality(op1, op2):
    """check if two operators are numerically equal"""

    if isinstance(op2, qutip.Qobj):
        op1, op2 = op2, op1

    if isinstance(op1, qutip.Qobj) and isinstance(op2, Operator):
        op2 = op2.to_qutip()

    op_diff = op1 - op2
    return abs((op_diff.dag() * op_diff).tr()) < 1.0e-9


def expect_from_qutip(rho, obs):
    """Compute expectation values or Qutip objects or iterables"""
    if isinstance(obs, Operator):
        return qutip.expect(rho, obs.to_qutip(tuple(obs.system.sites)))
    if isinstance(obs, dict):
        return {name: expect_from_qutip(rho, op) for name, op in obs.items()}
    return np.array([expect_from_qutip(rho, op) for op in obs])


def is_one_body_operator(operator) -> bool:
    """Check if the operator is a one-body operator"""
    if isinstance(operator, SumOperator):
        return all(is_one_body_operator(term) for term in operator.terms)
    return len(operator.acts_over()) < 2


GIBBS_GENERATOR_TESTS = {
    key: val for key, val in OPERATOR_TYPE_CASES.items() if val.isherm
}

PRODUCT_GIBBS_GENERATOR_TESTS = {
    key: val for key, val in GIBBS_GENERATOR_TESTS.items() if is_one_body_operator(val)
}


for key, val in GIBBS_GENERATOR_TESTS.items():
    name = "Gibbs from " + key
    TEST_CASES_STATES[name] = GibbsDensityOperator(val, SYSTEM)

for key, val in PRODUCT_GIBBS_GENERATOR_TESTS.items():
    name = "ProductGibbs from " + key
    TEST_CASES_STATES[name] = GibbsProductDensityOperator(val, SYSTEM)

print("loaded")
