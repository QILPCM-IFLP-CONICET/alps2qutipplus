"""
Tests and benchmarks for `gram_matrix`.
"""

import os
from test.helper import HAMILTONIAN, SX_A, SX_TOTAL, SZ_TOTAL, TEST_CASES_STATES

import numpy as np
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
<<<<<<< HEAD
=======
SCALAR_PRODUCTS["HS scalar product"] = lambda x, y: np.real((x * y).tr())
>>>>>>> origin/main

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
<<<<<<< HEAD
            6,
            8,
=======
            3,
            4,
>>>>>>> origin/main
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
<<<<<<< HEAD
=======


@pytest.mark.parametrize(["case", "basis"], list(HIERARCHICAL_BASIS_CASES.items()))
def test_brute_force_gram_matrix_benchmark(benchmark, case, basis):
    """
    Benchmark scalar products
    """
    sp = basis.sp
    ops = basis.operator_basis

    def impl():
        gram = np.empty(
            (
                len(ops),
                len(ops),
            )
        )
        for i, op1 in enumerate(ops):
            for j, op2 in enumerate(ops):
                if i == j:
                    gram[i, j] = np.real(sp(op1, op2))
                elif i > j:
                    continue
                gram[i, j] = gram[j, i] = np.real(sp(op1, op2))

        return gram

    benchmark.pedantic(impl, rounds=3, iterations=1)


@pytest.mark.parametrize(["case", "basis"], list(HIERARCHICAL_BASIS_CASES.items()))
def test_cross_gram_matrix_benchmark(benchmark, case, basis):
    """
    Benchmark scalar products
    """
    sp = basis.sp
    ops = basis.operator_basis

    if hasattr(sp, "cross_gram_matrix"):

        def impl():
            return sp.cross_gram_matrix(ops)

    else:

        def impl():
            co_gram = np.empty(
                (
                    len(ops),
                    len(ops),
                )
            )
            for i, op1 in enumerate(ops):
                for j, op2 in enumerate(ops):
                    co_gram[i, j] = np.real(sp(op1, op2))
            return co_gram

    benchmark.pedantic(impl, rounds=3, iterations=1)
>>>>>>> origin/main
