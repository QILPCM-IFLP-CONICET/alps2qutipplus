"""
Parallel routines

"""

import logging

import alpsqutip.settings as alpsqutip_settings
from alpsqutip.operators import Operator
from alpsqutip.operators.arithmetic import iterable_to_operator
from alpsqutip.operators.simplify import collect_nbody_terms

USE_PARALLEL = alpsqutip_settings.USE_PARALLEL
MAX_WORKERS = alpsqutip_settings.PARALLEL_MAX_WORKERS
USE_THREADS = alpsqutip_settings.PARALLEL_USE_THREADS

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
    # For few terms, use the serial version.
    if len(terms_pairs) < 100000 or not USE_PARALLEL:
        terms = tuple(op_1 * op_2 - op_2 * op_1 for op_1, op_2 in terms_pairs)
        return iterable_to_operator(terms, system).simplify()

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    chunksize = max(1, int(len(terms_pairs) / num_workers))
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
