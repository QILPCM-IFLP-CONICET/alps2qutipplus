"""
Parallel routines

"""

import logging
from functools import partial
from typing import Optional, Tuple

import alpsqutip.settings as alpsqutip_settings
from alpsqutip.operators import Operator
from alpsqutip.operators.arithmetic import iterable_to_operator
from alpsqutip.operators.simplify import collect_nbody_terms
from alpsqutip.operators.states.basic import (
    ProductDensityOperator,
)
from alpsqutip.operators.states.gibbs import (
    GibbsProductDensityOperator,
)

USE_PARALLEL = alpsqutip_settings.USE_PARALLEL
MAX_WORKERS = alpsqutip_settings.PARALLEL_MAX_WORKERS
USE_THREADS = alpsqutip_settings.PARALLEL_USE_THREADS


DISPATCH_PROJECTION_METHOD_PARALLEL = {}


if alpsqutip_settings.USE_PARALLEL:
    try:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        logging.info("Using parallel routines for large objects.")
    except ModuleNotFoundError:
        USE_PARALLEL = False
        logging.warning(
            "ProcessPoolExecutor/ThreadPoolExecutor cannot be loaded. Using serial routines."
        )
        MAX_WORKERS = 1
        USE_THREADS = False
else:
    logging.info("Using serial routines for large objects.")
    MAX_WORKERS = 1
    USE_THREADS = False


def _commutator_term_worker(entries):
    """
    Compute the commutator [hi, kj] = hi * kj - kj * hi and simplify the result.
    The `system` attribute of the result is set to `None` to reduce the
    serializing cost.

    Parameters:
        hi_kj (tuple): A tuple (hi, kj) of SumOperator objects.

    Returns:
        Operator: The simplified commutator operator with small terms removed (threshold 1e-5).
    """
    op_1, op_2 = entries
    return (op_1 * op_2 - op_2 * op_1).simplify()._set_system_()


def commutator_alps2qutip_parallel(
    op_1: Operator,
    op_2: Operator,
    use_threads: bool = USE_THREADS,
    num_workers: int = MAX_WORKERS,
) -> Operator:
    """
    The commutator of two Operator objects `op_1` and  `op_2`.
    Parallel implementation.
    """
    system = op_1.system.union(op_2.system)
    op_1_terms = collect_nbody_terms(op_1.flat())
    op_2_terms = collect_nbody_terms(op_2.flat())

    def fetch_terms():
        for block_1, terms_1 in op_1_terms.items():
            for block_2, terms_2 in op_2_terms.items():
                if (
                    block_1 is not None
                    and block_2 is not None
                    and not block_1.intersection(block_2)
                ):
                    continue
                for term_1 in terms_1:
                    for term_2 in terms_2:
                        if term_1 is term_2:
                            continue
                        yield (term_1, term_2)

    terms_pairs = tuple(pair for pair in fetch_terms())
    len_terms_pairs = len(terms_pairs)
    # The wall-time estimated for a task can be estimated as
    # WallTime =   N_TASKS / NUM_PROC * TIME_SINGLE_TASK + NUM_PROC * PARALLELIZING_OVERHEAD_TIME
    # The optimal number of processes is given by
    #  NUM_PROC = sqrt( N_TASKS *  TIME_SINGLE_TASK/PARALLELIZING_OVERHEAD_TIME )

    num_workers = min(1, max(num_workers, int((0.001 * len_terms_pairs) ** 0.5)))
    if num_workers == 1 or not USE_PARALLEL:
        terms = tuple(op_1 * op_2 - op_2 * op_1 for op_1, op_2 in terms_pairs)
        return iterable_to_operator(terms, system).simplify()

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    chunksize = max(1, int(len_terms_pairs / num_workers))
    with executor_cls(max_workers=num_workers) as executor:
        terms = tuple(
            (
                val
                for val in executor.map(
                    _commutator_term_worker, terms_pairs, chunksize=chunksize
                )
                if val is not None
            )
        )
    return iterable_to_operator(terms, system)._set_system_(system).simplify()


def _project_monomial_worker(operator, nmax, sigma):
    """Worker"""
    return (
        DISPATCH_PROJECTION_METHOD_PARALLEL[type(operator)](operator, nmax, sigma)
        .simplify()
        ._set_system_(None)
    )


def parallel_process_non_dispatched_terms(
    terms: Tuple[Operator],
    nmax: int,
    sigma: Optional[ProductDensityOperator | GibbsProductDensityOperator] = None,
    use_threads=USE_THREADS,
    max_workers=MAX_WORKERS,
) -> Operator:
    """
    Project each operator in `terms` to the nmax subspace, relative
    to the state `sigma`.
    """
    system = terms[0].system
    non_dispatched_length = len(terms)
    project_worker = partial(_project_monomial_worker, nmax=nmax, sigma=sigma)
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    # The wall-time estimated for a task can be estimated as
    # WallTime =   N_TASKS / NUM_PROC * TIME_SINGLE_TASK + NUM_PROC * PARALLELIZING_OVERHEAD_TIME
    # The optimal number of processes is given by
    #  NUM_PROC = sqrt( N_TASKS *  TIME_SINGLE_TASK/PARALLELIZING_OVERHEAD_TIME )

    num_workers = min(1, max(max_workers, int((0.1 * non_dispatched_length) ** 0.5)))
    if num_workers > 1:
        chunksize = max(1, int(non_dispatched_length / num_workers))
        with executor_cls(max_workers=num_workers) as executor:
            terms = tuple(
                term
                for term in executor.map(project_worker, terms, chunksize=chunksize)
            )
    else:
        terms = tuple(_project_monomial_worker(term, nmax, sigma) for term in terms)
    for term in terms:
        term._set_system_(system)
    return terms
