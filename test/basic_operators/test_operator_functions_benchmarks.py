"""
Tests and benchmarks for `gram_matrix`.
"""

from test.helper import HAMILTONIAN, SX_A, SX_TOTAL, SZ_TOTAL

import pytest

from alpsqutip.operators.functions import commutator


@pytest.mark.parametrize(
    [
        "case",
        "op1",
        "op2",
        "deep",
    ],
    [
        (
            "[H+Sz_Total, Sx_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SX_TOTAL,
            2,
        ),
        (
            "[H+Sz_Total, Sz_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SZ_TOTAL,
            2,
        ),
        (
            "[H+Sz_Total, SX_A]",
            HAMILTONIAN,
            SX_A,
            2,
        ),
        (
            "[H+Sz_Total, Sx_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SX_TOTAL,
            6,
        ),
        (
            "[H+Sz_Total, Sz_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SZ_TOTAL,
            6,
        ),
        (
            "[H+Sz_Total, SX_A]",
            HAMILTONIAN,
            SX_A,
            6,
        ),
        (
            "[H+Sz_Total, Sx_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SX_TOTAL,
            8,
        ),
        (
            "[H+Sz_Total, Sz_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SZ_TOTAL,
            8,
        ),
        (
            "[H+Sz_Total, SX_A]",
            HAMILTONIAN,
            SX_A,
            8,
        ),
    ],
)
def test_iterated_commutator_benchmark(benchmark, case, op1, op2, deep):
    """
    Benchmark scalar products
    """

    def impl():
        result = op2
        for i in range(deep):
            result = commutator(op1, result)
        return result

    benchmark.pedantic(impl, rounds=3, iterations=1)
