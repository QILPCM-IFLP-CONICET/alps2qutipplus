import numpy as np
import qutip as qutip
import scipy.linalg as linalg

### Parallelization functions employed using multithreading 

from itertools import product, combinations_with_replacement
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

### locally defined functions and operator classes

from alpsqutip.operators.states.meanfield.projections import (project_operator_to_m_body, 
                                                               project_qutip_operator_to_m_body)

from alpsqutip.operators import (
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
    """
    Compute the commutator [hi, kj] = hi * kj - kj * hi and simplify the result.

    Parameters:
        hi_kj (tuple): A tuple (hi, kj) of SumOperator objects.

    Returns:
        Operator: The simplified commutator operator with small terms removed (threshold 1e-5).
    """
    hi, kj = hi_kj
    return (hi * kj - kj * hi).simplify().tidyup(1e-5)


def _sp_worker(pair, basis, sp):
    """
    Compute the real-valued part of the scalar product between two SumOperators belonging to some basis,
    provided these are hermitian.

    Parameters:
        pair (tuple): A tuple (i, j) indicating indices in the basis list.
        basis (list): A list of operator objects forming a basis.
        sp (callable): A scalar product function sp(op1, op2) -> real.

    Returns:
        tuple: A tuple (i, j, val) where val is the real part of sp(basis[i], basis[j]).
    """
    i, j = pair
    val = float(np.real(sp(basis[i], basis[j])))
    return (i, j, val)


def _hij_worker_explicit(args):
    """
    Compute (i, val) where val = Re⟨b_i | [H, b_last]_proj⟩.

    Steps:
        - Compute commutator: [H, b_last] = -i H b_last + i b_last H
        - Project the commutator onto the m_max-body subspace using sigma_0 as reference
        - Take the scalar product between b_i and the projected commutator

    Parameters:
        args (tuple): Contains the following elements:
            i (int): Index of the current basis element.
            basis_i (Operator): The i-th basis element b_i.
            b_last (Operator): The operator b_j being commuted with H.
            H (Operator): The system Hamiltonian.
            sp (callable): Scalar product function sp(op1, op2) -> complex.
            sigma_0 (Operator or State): Reference operator for projection.
            m_max (int): Max number of bodies to retain in the projection.

    Returns:
        tuple: (i, val) where val is Re⟨b_i | [H, b_last]_proj⟩.
    """
    i, basis_i, b_last, H, sp, sigma_0, m_max = args

    # Compute commutator [H, b_last]
    comm = (-1j * H * b_last + 1j * b_last * H).simplify().tidyup(1e-5)

    # Project onto the m-body subspace
    comm_proj = parallel_project_operator_to_m_body(
        full_operator=comm,
        m_max=m_max,
        sigma_0=sigma_0
    )

    # Compute scalar product with basis_i
    val = sp(basis_i, comm_proj).real
    return i, val

def _project_single_term(term_m_sigma):
    """
    Worker function for projecting a single operator term to the m-body subalgebra.

    Used for parallelization inside `parallel_project_operator_to_m_body`.

    Args:
        term_m_sigma (tuple): A tuple (term, m_max, sigma_0), where
            - term (Operator): An operator to project, typically a SumOperator.
            - m_max (int): Maximum number of correlations retained in the projection.
            - sigma_0 (Operator or None): Local reference state for the projection.

    Returns:
        Operator: The projected SumOperator term.
    """
    term, m_max, sigma_0 = term_m_sigma
    return parallel_project_operator_to_m_body(term, m_max, sigma_0, parallel=True)

### 

from itertools import permutations
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def parallel_project_operator_to_m_body(full_operator, m_max=2, sigma_0=None, parallel=True):
    """
    Project an operator onto the subalgebra of at most m_max-body operators,
    relative to the local reference state `sigma_0`.

    If sigma_0 is not provided, maximally mixed states are assumed.

    Args:
        full_operator (Operator): The operator to be projected.
        m_max (int): The maximum number of bodies in the projected operator.
        sigma_0 (Operator or None): A reference state used to define the projection.
        parallel (bool): Whether to parallelize over SumOperator terms.

    Returns:
        Operator: The projected operator within the m-body operator subalgebra.
    """
    assert sigma_0 is None or hasattr(sigma_0, "expect"), f"Invalid sigma_0 of type {type(sigma_0)}"

    if m_max == 0:
        # Only scalar part survives
        if sigma_0:
            return ScalarOperator(np.real_if_close(sigma_0.expect(full_operator)), full_operator.system)
        return ScalarOperator(np.real_if_close(full_operator.tr()), full_operator.system)

    # Base case: already within m-body scope
    if isinstance(full_operator, OneBodyOperator) or len(full_operator.acts_over()) <= m_max:
        return full_operator

    full_operator = full_operator.simplify()
    system = full_operator.system

    # Handle SumOperator
    if isinstance(full_operator, SumOperator):
        if parallel and len(full_operator.terms) > 1:
            with ProcessPoolExecutor() as executor:
                args = [(term, m_max, sigma_0) for term in full_operator.terms]
                terms = tuple(op for op in executor.map(_project_single_term, args) if op is not None)
        else:
            terms = tuple(
                parallel_project_operator_to_m_body(term, m_max, sigma_0, parallel=False)
                for term in full_operator.terms
            )

        # Filter and rebuild result
        if not terms:
            return ScalarOperator(0.0, system)
        if len(terms) == 1:
            return terms[0]
        return SumOperator(terms, system).simplify()

    # Handle ProductOperator
    if isinstance(full_operator, ProductOperator):
        sites_op = full_operator.sites_op
        if len(sites_op) <= m_max:
            return full_operator

        # Peel off first site
        first_site, *rest = tuple(sites_op)
        op_first = sites_op[first_site]
        weight_first = op_first
        sigma_rest = sigma_0

        if sigma_0 is not None:
            try:
                sigma_first = sigma_0.partial_trace(frozenset({first_site})).to_qutip()
                sigma_rest = sigma_0.partial_trace(frozenset(rest))
                weight_first = op_first * sigma_first
            except Exception as e:
                raise ValueError(f"Failed to partial trace sigma_0: {e}")
        else:
            dim = getattr(op_first, 'dims', [[2]])[0][0]  # fallback to 2 if not found
            weight_first = weight_first / dim

        raw_tr = weight_first.tr()
        first_av = np.real_if_close(raw_tr)
        delta_op = LocalOperator(first_site, op_first - first_av, system)

        sites_op_rest = {site: op for site, op in sites_op.items() if site != first_site}
        rest_prod_operator = ProductOperator(
            sites_op_rest, prefactor=full_operator.prefactor, system=system
        )

        # Recursive decomposition
        result = delta_op * parallel_project_operator_to_m_body(rest_prod_operator, m_max - 1, sigma_rest, parallel)

        if not np.isclose(raw_tr, 0, atol=1e-12):  # or your preferred tolerance
            result += (ScalarOperator(first_av, system) 
                       * parallel_project_operator_to_m_body(rest_prod_operator, m_max, sigma_rest, parallel))

        return result.simplify()

    # Handle QutipOperator or fallback
    if isinstance(full_operator, QutipOperator):
        return project_qutip_operator_to_m_body(full_operator, m_max, sigma_0)

    return project_qutip_operator_to_m_body(full_operator.to_qutip_operator(), m_max, sigma_0)


### Max-Ent & Heisenberg functions 

def parallelized_real_time_projection_of_hierarchical_basis(generator, 
                                               seed_op,
                                               sigma_ref,
                                               m_max, 
                                               deep,
                                               num_workers=None,
                                               tidy_thresh=1e-5):
    """
    Construct a hierarchical basis of projected commutators using a generator operator.

    This function builds a basis of operators by iteratively computing commutators
    with the generator and projecting each new operator onto the m-body subalgebra,
    using a reference state for the projection. The commutator evaluations are
    parallelized for efficiency.

    Args:
        generator (Operator): The generator (e.g., Hamiltonian) for time evolution,
                                expressed as a sum of product operators.
        seed_op (Operator): The initial operator from which to build the basis.
        sigma_ref (Operator or None): The reference state for projecting onto the
                                        restricted subalgebra.
        m_max (int): The maximal body-order for the projection.
        deep (int): The number of iterated commutators used in the basis (depth of basis).
        num_workers (int or None): Number of parallel workers. Defaults to os.cpu_count().
        tidy_thresh (float): Threshold used in `tidyup()` to discard small terms.

    Returns:
        List[Operator]: A list of projected operators forming the hierarchical basis.
    """
    if seed_op is None or deep == 0:
        return []

    basis = [seed_op]
    
    gen_terms = (-1j*generator).as_sum_of_products().terms

    for i in range(1, deep):
        # Cache last basis terms only once
        basis_last_terms = basis[-1].as_sum_of_products().terms
        term_pairs = list(product(gen_terms, basis_last_terms))

        # Parallelize acá 
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            commutator_terms = list(executor.map(_commutator_term_worker, term_pairs, chunksize=128))

        # Assemble everything in the local op
        local_op = sum(commutator_terms)
        if True: 
            local_op = parallel_project_operator_to_m_body(
                            full_operator=local_op.tidyup(1e-5).as_sum_of_products(),
                            m_max=m_max, 
                            sigma_0=sigma_ref,
                            parallel=True
                        ).simplify().tidyup(tidy_thresh)
            
        basis.append(local_op)

    return basis

### Linear algebra functions involving a scalar product sp

def parallel_gram_matrix(basis, sp, num_workers=None, use_threads=False):
    """
    Compute the Gram matrix of a set of operators in parallel using a scalar product.

    This function evaluates all inner products ⟨b_i | b_j⟩ for i, j in `basis`
    and returns a real symmetric Gram matrix. Parallelization can be done via
    threads or processes.

    Parameters:
        basis (List[Operator]): List of basis operators.
        sp (Callable): Scalar product function taking two operators and returning a scalar.
                       Must be a top-level, pickleable function if using processes.
        num_workers (int or None): Number of worker threads or processes to use.
                                   Defaults to the number of cores.
        use_threads (bool): If True, uses threading instead of multiprocessing.
                            Useful when the scalar product is I/O-bound or GIL-friendly.

    Returns:
        np.ndarray: Symmetric real-valued Gram matrix of shape (len(basis), len(basis)).
    """
    size = len(basis)
    result = np.zeros((size, size), dtype=float)

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    index_pairs = list(combinations_with_replacement(range(size), 2))

    # Pre-bind basis and sp to _sp_worker
    worker = partial(_sp_worker, basis=basis, sp=sp)

    with executor_cls(max_workers=num_workers) as executor:
        for i, j, val in executor.map(worker, index_pairs):
            result[i, j] = val
            if i != j:
                result[j, i] = val

    return result.round(14)

def parallel_orthogonalize_basis_gs(basis, sp: callable, tol=1e-5, num_threads=4):
    """
    Orthogonalizes a list of operators using the Gram-Schmidt process with a given scalar product.

    This function performs Gram-Schmidt orthogonalization on a given operator basis using a 
    user-defined scalar product function `sp`. The process is partially parallelized 
    using threads for computing scalar products.

    Parameters:
        basis (List[Any]): List of operators or objects forming the initial basis.
        sp (Callable): Scalar product function, taking two basis elements and returning a scalar.
        tol (float): Threshold below which vectors are discarded as linearly dependent or too small.
        num_threads (int): Number of threads used to parallelize the scalar product evaluations.

    Returns:
        orth_basis (List[Any]): List of orthonormalized basis elements (same type as input basis).
        T (np.ndarray): Matrix of coefficients such that each orthonormalized basis vector q_k satisfies:
                        q_k = sum_j T[k, j] * b_j
                        The shape of T is (len(orth_basis), len(basis)).
    """
    orth_basis = []
    T = []

    for k, op_orig in enumerate(basis):
        op = op_orig
        coeffs = np.zeros(len(basis), dtype=np.complex128)
        coeffs[k] = 1.0  # Initially op = 1 * b_k

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            projections = list(executor.map(
                lambda prev: sp(prev, op),
                orth_basis
            ))

        for j, (proj, prev) in enumerate(zip(projections, orth_basis)):
            op -= proj * prev
            coeffs -= proj * T[j]  # No slicing here

        norm = np.real(sp(op, op)) ** 0.5
        if norm < tol:
            continue

        op /= norm
        coeffs /= norm
        orth_basis.append(op)
        T.append(coeffs)

    T = np.array(T)

    # Optional orthonormality check
    def check_orthonorm(pair):
        i, j = pair
        val = sp(orth_basis[i], orth_basis[j])
        if i == j:
            assert abs(val - 1.0) < tol, f"Norm not 1 at {i}: {val}"
        else:
            assert abs(val) < tol, f"Not orthogonal: {i}, {j} = {val}"

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(check_orthonorm, [(i, j) for i in range(len(orth_basis)) for j in range(i, len(orth_basis))])

    return orth_basis, T

def parallel_Hij_tensor(basis, generator, sigma_0, sp, m_max, 
                        is_basis_orthogonal=True, QR_matrix=None,
                        use_threads=False, max_workers=None):
    """
    Computes Hij = (op_i, [H, op_j]) for a given basis.

    If the basis is orthogonal and the QR_matrix is provided, computes the Hij
    for the orthonormal basis c_i = sum_{alpha} T**-1_{iα} b_α via a similarity transform:

        H^(c) = T**-1 @ H^(b) @ T**-1.T

    Otherwise, returns Hij in the original basis.
    
    The parameters generator, sigma_0, sp, m_max must be given in order to treat the last cases of the Hij-tensor,
    in particular to treat the:
    
        H_{i ell} = (bi, [H, b_{ell}]),
        
        since the b_{ell+1} = Proj_{m_max, sigma_0}[H, b_{ell}] is not included in the basis, 
        and cannot be computed from recurrence of the (orthogonalized or not) Hierarchical Basis.
        
    This procedure is similar if, instead, the basis is not orthogonal.
    """
    ell = len(basis)
    Hij_b = np.zeros((ell, ell), dtype=np.float64)
    sp_local = sp

    # Case 1: Use (b_i, b_{j+1}) for j < ell - 1
    for i in range(ell):
        for j in range(ell - 1):
            Hij_b[i, j] = sp_local(basis[i], basis[j + 1]).real

    # Case 2: Explicitly compute [H, b_{ell-1}]
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    tasks = [(i, basis[i], basis[-1], generator, sp_local, sigma_0, m_max) for i in range(ell)]

    with executor_cls(max_workers=max_workers) as executor:
        for i, val in executor.map(_hij_worker_explicit, tasks):
            Hij_b[i, ell - 1] = val

    # If orthogonal, transform using Tinv
    if is_basis_orthogonal and QR_matrix is not None:
        Tinv = np.linalg.inv(QR_matrix)
        return Tinv @ Hij_b @ Tinv.T

    return Hij_b
