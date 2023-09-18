"""
Helper functions for pytests
"""

from numbers import Number
from typing import Iterable

import numpy as np
import qutip

from alpsqutip.model import Operator, SystemDescriptor, build_spin_chain
from alpsqutip.operators import OneBodyOperator, ProductOperator, SumOperator
from alpsqutip.settings import VERBOSITY_LEVEL
from alpsqutip.states import (GibbsDensityOperator,
                              GibbsProductDensityOperator,
                              ProductDensityOperator)

CHAIN_SIZE = 6

system: SystemDescriptor = build_spin_chain(CHAIN_SIZE)
sites: tuple = tuple(s for s in system.sites.keys())

sz_total: OneBodyOperator = system.global_operator("Sz")
hamiltonian: SumOperator = system.global_operator("Hamiltonian")

global_identity: ProductOperator = ProductOperator({}, 1.0, system)

sx_A = system.site_operator(f"Sx@{sites[0]}")
sx_B = system.site_operator(f"Sx@{sites[1]}")
sx_AB = 0.7 * sx_A + 0.3 * sx_B


sy_A = system.site_operator(f"Sy@{sites[0]}")
sy_B = system.site_operator(f"Sy@{sites[1]}")


sz_A = system.site_operator(f"Sz@{sites[0]}")
sz_B = system.site_operator(f"Sz@{sites[1]}")
sz_C = system.site_operator(f"Sz@{sites[2]}")
sz_AB = 0.7 * sz_A + 0.3 * sz_B


sh_A = 0.25 * sx_A + 0.5 * sz_A
sh_B = 0.25 * sx_B + 0.5 * sz_B
sh_AB = 0.7 * sh_A + 0.3 * sh_B


subsystems = [
    [sites[0]],
    [sites[1]],
    [sites[2]],
    [sites[0], sites[1]],
    [sites[0], sites[2]],
    [sites[2], sites[3]],
]


observable_cases = {
    "Identity": ProductOperator({}, 1.0, system),
    "sz_total": sz_total,
    "sx_A": sx_A,
    "sy_A": sy_A,
    "sz_B": sz_B,
    "sh_AB": sh_AB,
    "exchange_AB": sx_A * sx_B + sy_A * sy_B,
    "hamiltonian": hamiltonian,
    "observable array": [[sh_AB, sh_A], [sz_A, sx_A]],
}


test_cases_states = {}

test_cases_states["fully mixed"] = ProductDensityOperator({}, 1.0, system=system)

test_cases_states["gibbs_sz"] = GibbsProductDensityOperator(sz_total, system=system)

test_cases_states["gibbs_sz_as_product"] = GibbsProductDensityOperator(
    sz_total, system=system
).to_product_state()
test_cases_states["gibbs_sz_bar"] = GibbsProductDensityOperator(
    -sz_total, system=system
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
        assert abs(lhs - rhs) < 1.0e-10
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
    return (op_diff.dag() * op_diff).tr() < 1.0e-9


def expect_from_qutip(rho, obs):
    """Compute expectation values or Qutip objects or iterables"""
    if isinstance(obs, Operator):
        return qutip.expect(rho, obs.to_qutip())
    if isinstance(obs, dict):
        return {name: expect_from_qutip(rho, op) for name, op in obs.items()}
    return np.array([expect_from_qutip(rho, op) for op in obs])
