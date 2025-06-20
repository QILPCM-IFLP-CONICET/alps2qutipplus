from copy import deepcopy
import numpy as np
import qutip as qutip
import scipy.linalg as linalg

### Parallelization functions employed using multithreading 

from itertools import product, combinations_with_replacement
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

### locally defined functions and operator classes

from .operators.states.meanfield.projections import project_to_n_body_operator
from .operators.simplify import simplify_sum_operator

from alpsqutip.operators.arithmetic import (
    ScalarOperator,
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
    QutipOperator
)

### Workers functions at a ,,low"-level implementation, to be used in the 
### parallelization of future tasks.

def _commutator_term_worker(hi_kj):
    hi, kj = hi_kj
    comm = simplify_sum_operator(hi * kj - kj * hi)
    return comm
    
def _projection_worker(args):
    term, m_max, sigma_0 = args
    return project_to_n_body_operator(
        operator=term,
        nmax=m_max,
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
    i, basis_i, b_last, H, sp, sigma_0, m_max = args

    comm = (-1j * H * b_last + 1j * b_last * H).simplify()
    comm_proj = project_to_n_body_operator(operator=comm, nmax=m_max, sigma=sigma_0)
    val = sp(basis_i, comm_proj)
    return i, val

### 

from itertools import permutations
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def parallelized_real_time_projection_of_hierarchical_basis(
    generator, 
    seed_op,
    sigma_ref,
    m_max, 
    deep,
    num_workers=None,
    tidy_thresh=1e-5,
):
    if seed_op is None or deep == 0:
        return []

    basis = [seed_op]
    gen_terms = (-1j * generator).as_sum_of_products().terms
    system = generator.system

    for i in range(1, deep):
        current_op = basis[-1]
        basis_last_terms = (
            current_op.terms if isinstance(current_op, SumOperator)
            else current_op.as_sum_of_products().terms
        )

        term_pairs = list(product(gen_terms, basis_last_terms))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            commutator_terms = list(executor.map(_commutator_term_worker, term_pairs, chunksize=128))

        commutator_terms = [term for term in commutator_terms if term is not None]
        projection_args = [(term, m_max, sigma_ref) for term in commutator_terms]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            projected_terms = list(executor.map(_projection_worker,
                                                projection_args, chunksize=128))

        local_op = sum(projected_terms, ScalarOperator(0, system=system))
        local_op = simplify_sum_operator(local_op)

        # --- Debug check: small norm triggers early exit ---
        norm_val = local_op.to_qutip().norm()
        #print(f"[DEBUG] Depth {i}: projected norm = {norm_val:.3e}")
        if norm_val < 1e-10:
        #    print("[DEBUG] Basis update terminated early: projected operator norm too small.")
            break

        basis.append(local_op)

    return basis

def compute_Hij_tensor_non_orth(basis, generator, sp, sigma_ref, m_max, Gram=None, num_workers=4):
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
        (i, basis[i], basis[-1], generator, sp, sigma_ref, m_max)
        for i in range(n)
    ]

    # Parallélisation du calcul de la dernière colonne
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_hij_worker_explicit, args)

    for i, val in results:
        Hij_tensor[i, n - 1] = val

    return Hij_tensor

def orthogonalize_basis_parallel_process(basis, sp: callable, tol=1e-6, max_workers=None):
    """
    Orthogonalizes a basis using classical Gram-Schmidt with parallel scalar product
    computation using ProcessPoolExecutor (multi-core true parallelism).

    Args:
        basis (List[Operator]): Initial operator basis {b_j}
        sp (callable): Scalar product function (must be picklable!)
        tol (float): Threshold to discard near-zero vectors
        max_workers (int or None): Max processes to use

    Returns:
        orth_basis (List[Operator]): Orthonormalized basis {q_i}
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

        # Compute all <q_i, v> in parallel
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
        orth_basis.append(v)

    R = R[:len(orth_basis), :]
    G = R.conj().T @ R
    return orth_basis, R, G




### not optimal ###
def parallel_gram_schmidt_with_gram(basis, sp: callable, tol=1e-6, num_threads=4):
    """
    Parallel Gram-Schmidt orthogonalization with Gram matrix computation.

    Args:
        basis (List[Operator]): Initial (non-orthogonal) operator basis.
        sp (Callable): Scalar product function.
        tol (float): Threshold to drop near-zero vectors.
        num_threads (int): Threads for parallel scalar product computation.

    Returns:
        orth_basis (List[Operator]): Orthonormalized operator basis {q_k}.
        R (np.ndarray): Coefficients matrix s.t. q_k = sum_j R[k,j] * b_j.
        G (np.ndarray): Gram matrix G[i,j] = sp(b_i, b_j).
    """

    def ensure_consistent_system(op, system):
        op.system = system
        if hasattr(op, "terms"):
            for term in op.terms:
                term.system = system
        return op

    n = len(basis)
    system = getattr(basis[0], "system", None)

    # Compute full Gram matrix G in parallel
    G = np.zeros((n, n), dtype=np.complex128)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Upper triangular indices only
        pairs = [(i, j) for i in range(n) for j in range(i, n)]
        results = list(executor.map(lambda ij: (ij[0], ij[1], sp(basis[ij[0]], basis[ij[1]])), pairs))
    for i, j, val in results:
        G[i, j] = val
        if i != j:
            G[j, i] = np.conj(val)  # Hermitian

    orth_basis = []
    R_rows = []

    for k, b_k in enumerate(basis):
        q_k = deepcopy(b_k)
        ensure_consistent_system(q_k, system)

        Rk = np.zeros(n, dtype=np.complex128)
        Rk[k] = 1.0

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            projections = list(executor.map(
                lambda q_j: sp(q_j, q_k),
                orth_basis
            ))

        for j, (proj, q_j) in enumerate(zip(projections, orth_basis)):
            q_k = q_k - proj * q_j
            Rk -= proj * R_rows[j]

        norm = np.real(sp(q_k, q_k))**0.5
        if norm < tol:
            continue

        q_k = q_k / norm
        Rk = Rk / norm

        ensure_consistent_system(q_k, system)
        orth_basis.append(q_k)
        R_rows.append(Rk)

    R = np.array(R_rows)

    return orth_basis, R, G
