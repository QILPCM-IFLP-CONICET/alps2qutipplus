"""
Helper functions for pytests
"""

from numbers import Number
from typing import Iterable

import numpy as np
import qutip

from alpsqutip.model import SystemDescriptor, build_spin_chain
from alpsqutip.operators import OneBodyOperator, Operator, ScalarOperator, SumOperator
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

system: SystemDescriptor = build_spin_chain(CHAIN_SIZE)
sites: tuple = tuple(s for s in system.sites.keys())


global_identity: ScalarOperator = ScalarOperator(1.0, system)


sx_A = system.site_operator(f"Sx@{sites[0]}")
sx_B = system.site_operator(f"Sx@{sites[1]}")
sx_AB = 0.7 * sx_A + 0.3 * sx_B


sy_A = system.site_operator(f"Sy@{sites[0]}")
sy_B = system.site_operator(f"Sy@{sites[1]}")

assert isinstance(sx_A * sx_B + sy_A * sy_B, Operator)


splus_A = system.site_operator(f"Splus@{sites[0]}")
splus_B = system.site_operator(f"Splus@{sites[1]}")
sminus_A = system.site_operator(f"Sminus@{sites[0]}")
sminus_B = system.site_operator(f"Sminus@{sites[1]}")


sz_A = system.site_operator(f"Sz@{sites[0]}")
sz_B = system.site_operator(f"Sz@{sites[1]}")
sz_C = system.site_operator(f"Sz@{sites[2]}")
sz_AB = 0.7 * sz_A + 0.3 * sz_B


sh_A = 0.25 * sx_A + 0.5 * sz_A
sh_B = 0.25 * sx_B + 0.5 * sz_B
sh_AB = 0.7 * sh_A + 0.3 * sh_B


sz_total: OneBodyOperator = system.global_operator("Sz")
assert isinstance(sz_total, OneBodyOperator)

sx_total: OneBodyOperator = sum(system.site_operator("Sx", s) for s in sites)
sy_total: OneBodyOperator = sum(system.site_operator("Sy", s) for s in sites)
hamiltonian: SumOperator = system.global_operator("Hamiltonian")


assert (sminus_A * sminus_B) is not None

subsystems = [
    [sites[0]],
    [sites[1]],
    [sites[2]],
    [sites[0], sites[1]],
    [sites[0], sites[2]],
    [sites[2], sites[3]],
]


observable_cases = {
    "Identity": ScalarOperator(1.0, system),
    "sz_total": sz_total,  # OneBodyOperator
    "sx_A": sx_A,  # LocalOperator
    "sy_A": sy_A,  # Local Operator
    "sz_B": sz_B,  # Diagonal local operator
    "sh_AB": sh_AB,  # ProductOperator
    "exchange_AB": sx_A * sx_B + sy_A * sy_B,  # Sum operator
    "hamiltonian": hamiltonian,  # Sum operator, hermitician
    "observable array": [[sh_AB, sh_A], [sz_A, sx_A]],
}


operator_type_cases = {
    "scalar, real": ScalarOperator(1.0, system),
    "scalar, complex": ScalarOperator(1.0 + 3j, system),
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
    "qutip operator": hamiltonian.to_qutip_operator(),
    "hermitician quadratic operator": build_quadratic_form_from_operator(hamiltonian),
    "non hermitician quadratic operator": build_quadratic_form_from_operator(
        hamiltonian - sz_total * 1j
    ),
    "log unitary": build_quadratic_form_from_operator(hamiltonian * 1j),
    "single interaction term": build_quadratic_form_from_operator(sx_A * sx_B),
}


test_cases_states = {}

test_cases_states["fully mixed"] = ProductDensityOperator({}, system=system)

test_cases_states["z semipolarized"] = ProductDensityOperator(
    {name: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmaz() for name in system.dimensions},
    1.0,
    system=system,
)

test_cases_states["x semipolarized"] = ProductDensityOperator(
    {name: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmax() for name in system.dimensions},
    1.0,
    system=system,
)


test_cases_states["first full polarized"] = ProductDensityOperator(
    {sx_A.site: 0.5 * qutip.qeye(2) + 0.5 * qutip.sigmaz()}, 1.0, system=system
)

test_cases_states[
    "mixture of first and second partially polarized"
] = 0.5 * ProductDensityOperator(
    {sx_A.site: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmaz()}, 1.0, system=system
) + 0.5 * ProductDensityOperator(
    {sx_B.site: 0.5 * qutip.qeye(2) + 0.25 * qutip.sigmaz()}, 1.0, system=system
)


test_cases_states["gibbs_sz"] = GibbsProductDensityOperator(sz_total, system=system)

test_cases_states["gibbs_sz_as_product"] = GibbsProductDensityOperator(
    sz_total, system=system
).to_product_state()
test_cases_states["gibbs_sz_bar"] = GibbsProductDensityOperator(
    sz_total * (-1), system=system
)
test_cases_states["gibbs_H"] = GibbsDensityOperator(hamiltonian, system=system)
test_cases_states["gibbs_H"] = (
    test_cases_states["gibbs_H"] / test_cases_states["gibbs_H"].tr()
)
test_cases_states["mixture"] = (
    0.5 * test_cases_states["gibbs_H"]
    + 0.25 * test_cases_states["gibbs_sz"]
    + 0.25 * test_cases_states["gibbs_sz_bar"]
)


full_test_cases = {}
full_test_cases.update(operator_type_cases)
full_test_cases.update(test_cases_states)


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
        return qutip.expect(rho, obs.to_qutip())
    if isinstance(obs, dict):
        return {name: expect_from_qutip(rho, op) for name, op in obs.items()}
    return np.array([expect_from_qutip(rho, op) for op in obs])


print("loaded")
