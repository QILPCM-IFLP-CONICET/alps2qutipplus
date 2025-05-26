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


GLOBAL_IDENTITY: ScalarOperator = ScalarOperator(1.0, SYSTEM)


SX_A = SYSTEM.site_operator(f"Sx@{SITES[0]}")
SX_B = SYSTEM.site_operator(f"Sx@{SITES[1]}")
SX_AB = 0.7 * SX_A + 0.3 * SX_B


SY_A = SYSTEM.site_operator(f"Sy@{SITES[0]}")
SY_B = SYSTEM.site_operator(f"Sy@{SITES[1]}")

assert isinstance(SX_A * SX_B + SY_A * SY_B, Operator)


SPLUS_A = SYSTEM.site_operator(f"Splus@{SITES[0]}")
SPLUS_B = SYSTEM.site_operator(f"Splus@{SITES[1]}")
SMINUS_A = SYSTEM.site_operator(f"Sminus@{SITES[0]}")
SMINUS_B = SYSTEM.site_operator(f"Sminus@{SITES[1]}")


SZ_A = SYSTEM.site_operator(f"Sz@{SITES[0]}")
SZ_B = SYSTEM.site_operator(f"Sz@{SITES[1]}")
SZ_C = SYSTEM.site_operator(f"Sz@{SITES[2]}")
SZ_AB = 0.7 * SZ_A + 0.3 * SZ_B


SH_A = 0.25 * SX_A + 0.5 * SZ_A
sh_B = 0.25 * SX_B + 0.5 * SZ_B
SH_AB = 0.7 * SH_A + 0.3 * sh_B


SZ_TOTAL: OneBodyOperator = SYSTEM.global_operator("Sz")
assert isinstance(SZ_TOTAL, OneBodyOperator)

SX_TOTAL: OneBodyOperator = sum(SYSTEM.site_operator("Sx", s) for s in SITES)
SY_TOTAL: OneBodyOperator = sum(SYSTEM.site_operator("Sy", s) for s in SITES)
HAMILTONIAN: SumOperator = SYSTEM.global_operator("Hamiltonian")

assert HAMILTONIAN is not None

assert (SMINUS_A * SMINUS_B) is not None


SPLUS0 = SYSTEM.site_operator(f"Splus@{SITES[0]}")
SPLUS1 = SYSTEM.site_operator(f"Splus@{SITES[1]}")

SPSP_HC = SumOperator(
    (
        SPLUS0 * SPLUS1,
        (SPLUS0 * SPLUS1).dag(),
    ),
    SYSTEM,
    True,
)


OPERATORS = {
    "Identity": GLOBAL_IDENTITY,
    "sz_total": SZ_TOTAL,
    "sx_total": SX_TOTAL,
    "sx_total_sq": SX_TOTAL * SX_TOTAL,
    "sx+ 3j sz": SX_TOTAL + (3 * 1j) * SZ_TOTAL,
    "splus*splus": SPLUS0 * SPLUS1,
    "splus*splus+hc": SPSP_HC,
    "hamiltonian": HAMILTONIAN,
    "nonhermitician": HAMILTONIAN + (3 * 1j) * SZ_TOTAL,
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
    "sz_total": SZ_TOTAL,  # OneBodyOperator
    "Sx_A": SX_A,  # LocalOperator
    "sy_A": SY_A,  # Local Operator
    "sz_B": SZ_B,  # Diagonal local operator
    "sh_AB": SH_AB,  # ProductOperator
    "exchange_AB": SX_A * SX_B + SY_A * SY_B,  # Sum operator
    "hamiltonian": HAMILTONIAN,  # Sum operator, hermitician
    "observable array": [[SH_AB, SH_A], [SZ_A, SX_A]],
}


OPERATOR_TYPE_CASES = {
    "scalar, zero": ScalarOperator(0.0, SYSTEM),
    "product, zero": ProductOperator({}, prefactor=0.0, system=SYSTEM),
    "product, 1": ProductOperator({}, prefactor=1.0, system=SYSTEM),
    "scalar, real": ScalarOperator(1.0, SYSTEM),
    "scalar, complex": ScalarOperator(1.0 + 3j, SYSTEM),
    "local operator, hermitician": SX_A,  # LocalOperator
    "local operator, non hermitician": SX_A + SY_A * 1j,
    "One body, hermitician": SZ_TOTAL,
    "One body, non hermitician": SX_TOTAL + SY_TOTAL * 1j,
    "three body, hermitician": (SX_A * SY_B * SZ_C),
    "three body, non hermitician": ((SMINUS_A * SMINUS_B + SY_A * SY_B) * SZ_TOTAL),
    "product operator, hermitician": SH_AB,
    "product operator, non hermitician": SMINUS_A * SPLUS_B,
    "sum operator, hermitician": SX_A * SX_B + SY_A * SY_B,  # Sum operator
    "sum operator, hermitician from non hermitician": SPLUS_A * SPLUS_B
    + SMINUS_A * SMINUS_B,
    "sum operator, anti-hermitician": SPLUS_A * SPLUS_B - SMINUS_A * SMINUS_B,
    "sum local operators": SPLUS_A + SMINUS_A,
    "sum local qutip operators": 2.0 * SPLUS_A.to_qutip_operator()
    + SMINUS_A.to_qutip_operator() * 2.0,
    "sum local qutip operator and local operator": (
        2.0 * SPLUS_A.to_qutip_operator()
        + SMINUS_A * 2.0
        + SPLUS_B.to_qutip_operator() * 2
        + 2 * SMINUS_B
    ),
    "sum two-body qutip operators": 0.25
    * (SPLUS_A.to_qutip_operator() * SPLUS_B.to_qutip_operator())
    + (SMINUS_A * SMINUS_B) * 0.25,
    "qutip operator": HAMILTONIAN.to_qutip_operator(),
    "hermitician quadratic operator": build_quadratic_form_from_operator(HAMILTONIAN),
    "non hermitician quadratic operator": build_quadratic_form_from_operator(
        HAMILTONIAN - SZ_TOTAL * 1j
    ),
    "log unitary": build_quadratic_form_from_operator(HAMILTONIAN * 1j),
    "single interaction term": build_quadratic_form_from_operator(SX_A * SX_B),
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
    {SX_A.site: 0.5 * qutip.qeye(2) + 0.5 * qutip.sigmaz()}, 1.0, system=SYSTEM
)

TEST_CASES_STATES[
    "mixture of first and second partially polarized"
] = 0.5 * ProductDensityOperator(
    {SX_A.site: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmaz()}, 1.0, system=SYSTEM
) + 0.5 * ProductDensityOperator(
    {SX_B.site: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmaz()}, 1.0, system=SYSTEM
)


TEST_CASES_STATES["gibbs_sz"] = GibbsProductDensityOperator(SZ_TOTAL, system=SYSTEM)

TEST_CASES_STATES["gibbs_sz_as_product"] = GibbsProductDensityOperator(
    SZ_TOTAL, system=SYSTEM
).to_product_state()
TEST_CASES_STATES["gibbs_sz_bar"] = GibbsProductDensityOperator(
    SZ_TOTAL * (-1), system=SYSTEM
)
TEST_CASES_STATES["gibbs_H"] = GibbsDensityOperator(HAMILTONIAN, system=SYSTEM)
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
