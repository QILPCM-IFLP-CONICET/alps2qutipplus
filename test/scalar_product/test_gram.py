"""
Tests and benchmarks for `gram_matrix`.
"""

import os
from test.helper import HAMILTONIAN, SX_A, SX_TOTAL, SZ_TOTAL, TEST_CASES_STATES

import pytest

from alpsqutip.scalarprod.basis import HierarchicalOperatorBasis
from alpsqutip.scalarprod.build import fetch_covar_scalar_product
from alpsqutip.scalarprod.gram import (
    gram_matrix,
)

REFERENCE_STATE_NAMES = (
    "ProductGibbs from scalar, zero",
    "ProductGibbs from product operator, hermitician",
    "ProductGibbs from sum local qutip operator and local operator",
    "ProductGibbs from One body, hermitician",
)

SCALAR_PRODUCTS = {
    key: fetch_covar_scalar_product(TEST_CASES_STATES[key])
    for key in REFERENCE_STATE_NAMES
}

if os.environ.get("BENCHMARKS", 0):
    HIERARCHICAL_BASIS_CASES = {
        f"{sp_name}_{k_name}_{deep}": HierarchicalOperatorBasis(
            k, HAMILTONIAN + SZ_TOTAL, deep, sp_
        )
        for k_name, k in {
            "SX_A": SX_A,
            "SX_TOTAL": SX_TOTAL,
            "SZ_TOTAL": SZ_TOTAL,
        }.items()
        for deep in (
            2,
            3,
            4,
        )
        for sp_name, sp_ in SCALAR_PRODUCTS.items()
    }
else:
    HIERARCHICAL_BASIS_CASES = {}


@pytest.mark.parametrize(["case", "basis"], list(HIERARCHICAL_BASIS_CASES.items()))
def test_gram_matrix_benchmark(benchmark, case, basis):
    """
    Benchmark scalar products
    """
    sp = basis.sp
    ops = basis.operator_basis

    def impl():
        return gram_matrix(ops, sp)

    benchmark.pedantic(impl, rounds=3, iterations=1)
