from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from itertools import combinations_with_replacement, product

import numpy as np
import qutip as qutip
import scipy.linalg as linalg

from alpsqutip.operators.arithmetic import (
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)

from .operators.simplify import simplify_sum_operator
from .optimized_projections import opt_project_to_n_body_operator

### Parallelization functions employed using multithreading 


### locally defined functions and operator classes



### Workers functions at a ,,low"-level implementation, to be used in the 
### parallelization of future tasks.

def _commutator_term_worker(hi_kj):
    hi, kj = hi_kj
    if frozenset(hi.acts_over()).isdisjoint(kj.acts_over()):
        return ScalarOperator(0, system=hi.system) 
    comm = simplify_sum_operator(hi * kj - kj * hi)
    return comm
    
def _projection_worker(args):
    term, nmax, sigma_0 = args
    return opt_project_to_n_body_operator(
        operator=term,
        nmax=nmax,
        sigma=sigma_0,
    ).simplify().tidyup(1e-10)
    
def _scalar_product_task(args):
    """Helper function for multiprocessing (must be top-level to be picklable)."""
    i, q, v, sp = args
    return i, sp(q, v)
    
def _hij_worker_explicit(args):
    """
    Calcule (i, val) avec val = Re⟨b_i | [H, b_last]_proj⟩.

    Étapes :
        - Calculer le commutateur : [H, b_last] = -i H b_last + i b_last H
        - Projeter le commutateur sur le sous-espace à m corps
        - Calculer le produit scalaire avec b_i

    Paramètres :
        args (tuple) : Contient (i, b_i, b_last, H, sp, sigma_0, m_max)

    Retour :
        tuple (i, val) où val est le produit scalaire ⟨b_i | [H, b_last]_proj⟩.
    """
    i, basis_i, b_last, H, sp, sigma_0, nmax = args

    comm = (-1j * H * b_last + 1j * b_last * H).simplify()
    comm_proj = opt_project_to_n_body_operator(operator=comm, 
                                               nmax=nmax,
                                               sigma=sigma_0)
    val = sp(basis_i, comm_proj)
    return i, val

def _gram_matrix_worker(args):
    i, j, op_i, op_j, sp = args
    return i, j, sp(op_i, op_j)

### 

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations

import numpy as np


def parallelized_real_time_projection_of_hierarchical_basis(
    generator, 
    seed_op,
    sigma_ref,
    nmax,
    deep,
    num_workers=None,
    ell_prime=None,
    chunksize=None,
    tidy_thresh=1e-5,
):
    if seed_op is None or deep == 0:
        return []
    if ell_prime is None: 
        ell_prime = deep

    basis = [seed_op]
    gen_terms = (-1j * generator).as_sum_of_products().terms
    system = generator.system

    for i in range(1, deep):
        current_op = basis[-1]
        basis_last_terms = (
            current_op.terms if isinstance(current_op, SumOperator)
            else current_op.as_sum_of_products().terms
        )

        # Precompute supports
        gen_term_supports = {g: frozenset(g.acts_over()) for g in gen_terms}
        basis_term_supports = {b: frozenset(b.acts_over()) for b in basis_last_terms}

        # Filtered term pairs (already avoids unnecessary commutators)
        term_pairs = [
            (g, b)
            for g in gen_terms
            for b in basis_last_terms
            if gen_term_supports[g].intersection(basis_term_supports[b])
        ]

        # Compute dynamic chunksize
        total_tasks = len(term_pairs)
        if chunksize is None:
            num_workers_eff = num_workers or os.cpu_count() or 4
            chunksize = max(1, total_tasks // (8 * num_workers_eff))  # factor 8 gives smallish chunks

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            commutator_terms = list(
                executor.map(_commutator_term_worker, term_pairs, chunksize=chunksize)
            )

        # Group by support
        grouped_terms = defaultdict(list)
        for term in commutator_terms:
            support = frozenset(term.acts_over())
            grouped_terms[support].append(term)

        # Sum terms with same support
        merged_terms = [
            sum(group, ScalarOperator(0, system=system)) for group in grouped_terms.values()
        ]

        # Prepare projection
        if i <= ell_prime: 
            projection_args = [(term, nmax, sigma_ref) for term in merged_terms]
        else: 
            projection_args = [(term, 2, sigma_ref) for term in merged_terms]
            
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            projected_terms = list(
                executor.map(_projection_worker, projection_args, chunksize=chunksize)
            )

        # Clean-up
        merged_terms = commutator_terms = None

        local_op = sum(projected_terms, ScalarOperator(0, system=system))
        local_op = simplify_sum_operator(local_op)

        norm_val = local_op.to_qutip().norm()
        if norm_val < 1e-10:
            break

        basis.append(local_op)

    return basis

def compute_Hij_tensor_non_orth(basis, generator, sp, sigma_ref, nmax, Gram=None, num_workers=4, chunksize=32):
    """
    Construit la matrice Hij_tensor_explicit.

    Args:
        basis (List[Operator]): Base non orthonormée {b_i}
        H (Operator): Hamiltonien du système
        sp (callable): Produit scalaire
        sigma0 (Operator): Référence pour la projection
        m_max (int): Nombre maximal de corps à garder
        Gram (np.ndarray): Matrice de Gram facultative (utilisée pour éviter recalculs)
        num_threads (int): Nombre de threads pour la parallélisation

    Returns:
        Hij_tensor (np.ndarray): Matrice Hij complète
    """
    n = len(basis)
    Hij_tensor = np.zeros((n, n), dtype=np.complex128)

    # Calcul des entrées Hij[i, j] pour j < n-1 (en supposant Gram disponible)
    if Gram is not None:
        for i in range(n):
            for j in range(n - 1):
                Hij_tensor[i, j] = Gram[i, j + 1]
    else:
        for i in range(n):
            for j in range(n - 1):
                Hij_tensor[i, j] = sp(basis[i], basis[j + 1])

    # Préparation des arguments pour la dernière colonne
    args = [
        (i, basis[i], basis[-1], generator, sp, sigma_ref, nmax)
        for i in range(n)
    ]

    # Parallélisation du calcul de la dernière colonne
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_hij_worker_explicit, args, chunksize=chunksize)

    for i, val in results:
        Hij_tensor[i, n - 1] = val

    return Hij_tensor

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np


def _scalar_product_task(args):
    i, q, v, sp = args
    return i, sp(q, v)

def orthogonalize_basis_parallel_process(
    basis, sp: callable, tol=1e-6, max_workers=None, return_orth_basis=False
):
    """
    Orthogonalizes a basis using Gram-Schmidt with parallel scalar product
    computation, and returns the transformation matrix and Gram matrix.

    Args:
        basis (List[Operator]): Initial operator basis {b_j}
        sp (callable): Scalar product function (must be picklable!)
        tol (float): Threshold to discard near-zero vectors
        max_workers (int or None): Max processes to use
        return_orth_basis (bool): If False, skips returning orthonormal basis

    Returns:
        orth_basis (List[Operator]) or None: If return_orth_basis is True
        R (np.ndarray): Transformation matrix, b_j = sum_i R[i,j] * q_i
        G (np.ndarray): Gram matrix, G = R† R
    """
    n = len(basis)
    system = getattr(basis[0], "system", None)

    def ensure_consistent(op):
        op.system = system
        if hasattr(op, "terms"):
            for t in op.terms:
                t.system = system
        return op

    orth_basis = []
    R = np.zeros((n, n), dtype=np.complex128)

    for j in range(n):
        v = deepcopy(basis[j])
        ensure_consistent(v)

        # Parallel scalar products <q_i, v>
        args = [(i, q, v, sp) for i, q in enumerate(orth_basis)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_scalar_product_task, args))

        for i, val in results:
            R[i, j] = val
            v = v - val * orth_basis[i]

        norm = np.real(sp(v, v)) ** 0.5
        if norm < tol:
            R[:, j] = 0.0
            continue

        R[len(orth_basis), j] = norm
        v = v / norm
        ensure_consistent(v)
        if return_orth_basis:
            orth_basis.append(v)

    R = R[:len(orth_basis), :]
    G = R.conj().T @ R

    return (orth_basis if return_orth_basis else None), R, G

def parallel_gram_matrix_process(basis, sp, num_workers=None, chunksize=16):
    """
    Computes the Hermitian Gram matrix G[i,j] = sp(b_i, b_j) in parallel using processes.

    Args:
        basis (List[Operator]): List of operator objects.
        sp (Callable): Pickleable scalar product function.
        num_workers (int or None): Number of parallel processes.

    Returns:
        G (np.ndarray): Hermitian Gram matrix.
    """
    n = len(basis)
    G = np.zeros((n, n), dtype=np.complex128)

    tasks = [(i, j, basis[i], basis[j], sp) for i in range(n) for j in range(i, n)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, j, val in executor.map(_gram_matrix_worker, tasks, chunksize=16):
            G[i, j] = val
            if i != j:
                G[j, i] = np.conj(val)

    return G

def _fine_sp_worker(args):
    i, j, op1, op2, sp = args
    if frozenset(op1.acts_over()).isdisjoint(op2.acts_over()):
        return i, j, 0.0  
    return i, j, sp(op1, op2)

def parallel_gram_matrix_fine(basis, sp, num_workers=None, chunksize=32):
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    n = len(basis)
    flat_basis = [op.as_sum_of_products().terms for op in basis]

    tasks = [
        (i, j, a, b, sp) 
        for i in range(n) for j in range(i, n)
        for a in flat_basis[i] for b in flat_basis[j]
    ]

    partial_sums = defaultdict(complex)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, j, val in executor.map(_fine_sp_worker, tasks, chunksize=chunksize):
            partial_sums[(i, j)] += val

    # Assemble final G matrix
    G = np.zeros((n, n), dtype=np.complex128)
    for (i, j), val in partial_sums.items():
        G[i, j] = val
        if i != j:
            G[j, i] = np.conj(val)

    return G
