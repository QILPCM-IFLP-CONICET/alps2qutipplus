"""
Functions for operators.
"""

import logging

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple
from typing import Union

from qutip import Qobj

import alpsqutip.settings as alpsqutip_settings
from alpsqutip.operators.arithmetic import SumOperator, iterable_to_operator
from alpsqutip.operators.basic import (
    Operator,
    ScalarOperator,
)

# from alpsqutip.operators.simplify import simplify_sum_operator


# from datetime import datetime


MAX_WORKERS = alpsqutip_settings.PARALLEL_MAX_WORKERS
USE_THREADS = alpsqutip_settings.PARALLEL_USE_THREADS

if alpsqutip_settings.USE_PARALLEL:
    try:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        logging.info("Using parallel routines to compute commutators.")
    except ModuleNotFoundError:
        alpsqutip_settings.USE_PARALLEL = False
        logging.warning(
            "ProcessPoolExecutor/ThreadPoolExecutor cannot be loaded. Using serial routines."
        )
else:
    logging.info("Using serial routines to compute commutators.")


def anticommutator(
    op_1: Union[Qobj, Operator], op_2: Union[Qobj, Operator]
) -> Union[Qobj, Operator]:
    """
    Computes the anticommutator of two operators, defined as {op1, op2} = op1 * op2 + op2 * op1.

    Parameters:
        op1, op2: operators (can be a matrix or a quantum operator object).

    Returns:
        The anticommutator of op1 and op2.
    """
    if isinstance(op_1, Qobj):
        if not isinstance(op_2, Qobj):
            op_2 = op_2.to_qutip()
        return op_1 * op_2 + op_2 * op_1
    if isinstance(op_2, Qobj):
        op_1 = op_1.to_qutip()
        return op_1 * op_2 + op_2 * op_1

    return anticommutator_alps2qutip(op_1, op_2)


def anticommutator_alps2qutip_serial(op_1: Operator, op_2: Operator) -> Operator:
    """
    Computes the anticommutator of two operators, defined as {op1, op2} = op1 * op2 + op2 * op1.

    Parameters:
        op1, op2: operators (can be a matrix or a quantum operator object).

    Returns:
        The anticommutator of op1 and op2.
    """
    system = op_1.system or op_2.system
    if isinstance(op_1, SumOperator):
        return SumOperator(
            tuple((anticommutator_alps2qutip(term, op_2) for term in op_1.terms)),
            system,
        ).simplify()
    if isinstance(op_2, SumOperator):
        return SumOperator(
            tuple((anticommutator_alps2qutip(op_1, term) for term in op_2.terms)),
            system,
        ).simplify()

    # TODO: Handle fermions...
    acts_over_1, acts_over_2 = op_1.acts_over(), op_2.acts_over()
    if acts_over_1 is not None:
        if len(acts_over_1) == 0:
            return op_2 * (op_1 * 2)
        if acts_over_2 is not None:
            if len(acts_over_2) == 0:
                return op_1 * (op_2 * 2)
            elif len(acts_over_1.intersection(acts_over_2)) == 0:
                return (op_1 * op_2).simplify() * 2
    return (op_1 * op_2 + op_2 * op_1).simplify()


def commutator(
    op_1: Union[Operator, Qobj], op_2: Union[Operator, Qobj]
) -> Union[Qobj, Operator]:
    """
    Commutator of two operators
    """
    if isinstance(op_1, Qobj):
        if not isinstance(op_2, Qobj):
            op_2 = op_2.to_qutip()
        return op_1 * op_2 - op_2 * op_1
    if isinstance(op_2, Qobj):
        op_1 = op_1.to_qutip()
        return op_1 * op_2 - op_2 * op_1

    return commutator_alps2qutip(op_1, op_2)


def _commutator_term_worker(entries):
    """
    Compute the commutator [hi, kj] = hi * kj - kj * hi and simplify the result.

    Parameters:
        hi_kj (tuple): A tuple (hi, kj) of SumOperator objects.

    Returns:
        Operator: The simplified commutator operator with small terms removed (threshold 1e-5).
    """
    op_1, op_2 = entries
    acts_over_1, acts_over_2 = op_1.acts_over(), op_2.acts_over()
    if acts_over_1 is not None:
        if len(acts_over_1) == 0:
            return None
        if acts_over_2 is not None:
            if len(acts_over_2) == 0 or len(acts_over_1.intersection(acts_over_2)) == 0:
                return None
    return (op_1 * op_2 - op_2 * op_1).simplify()._set_system_()


def commutator_alps2qutip_parallel(
    op_1: Operator, op_2: Operator, use_threads=USE_THREADS, num_workers=MAX_WORKERS
) -> Operator:
    """
    The commutator of two Operator objects. Parallel implementation.
    """
    if op_1 is op_2:
        return ScalarOperator(0, op_1.system)

    system = op_1.system.union(op_2.system)
    op_1 = op_1.flat()
    op_2 = op_2.flat()
    op_1_terms = op_1.terms if isinstance(op_1, SumOperator) else (op_1,)
    op_2_terms = op_2.terms if isinstance(op_2, SumOperator) else (op_2,)
    terms_entries = (
        (
            t1,
            t2,
        )
        for t1 in op_1_terms
        for t2 in op_2_terms
    )

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor_cls(max_workers=num_workers) as executor:
        terms = tuple(
            (
                val
                for val in executor.map(_commutator_term_worker, terms_entries)
                if val is not None
            )
        )

    for t in terms:
        t._set_system_(system)
    return iterable_to_operator(terms, system).simplify()


def commutator_alps2qutip_serial(op_1: Operator, op_2: Operator) -> Operator:
    """
    The commutator of two Operator objects.
    Serial implementation.
    """
    if op_1 is op_2:
        return ScalarOperator(0, op_1.system)
    system = op_1.system or op_2.system
    if isinstance(op_1, SumOperator):
        return SumOperator(
            tuple((commutator_alps2qutip(term, op_2) for term in op_1.terms)), system
        ).simplify()
    if isinstance(op_2, SumOperator):
        return SumOperator(
            tuple((commutator_alps2qutip(op_1, term) for term in op_2.terms)), system
        ).simplify()

    acts_over_1, acts_over_2 = op_1.acts_over(), op_2.acts_over()
    if acts_over_1 is not None:
        if len(acts_over_1) == 0:
            return ScalarOperator(0, system)
        if acts_over_2 is not None:
            if len(acts_over_2) == 0 or len(acts_over_1.intersection(acts_over_2)) == 0:
                return ScalarOperator(0, system)
    return (op_1 * op_2 - op_2 * op_1).simplify()


anticommutator_alps2qutip = anticommutator_alps2qutip_serial

commutator_alps2qutip = (
    commutator_alps2qutip_parallel
    if False and alpsqutip_settings.USE_PARALLEL
    else commutator_alps2qutip_serial
)
