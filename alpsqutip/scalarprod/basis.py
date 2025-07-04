"""
Basis of Operator metric sub-spaces
"""

import logging
from typing import Callable, Iterable, Optional, Tuple, cast

import numpy as np
from numpy.linalg import LinAlgError, cholesky, inv
from numpy.typing import NDArray
from scipy.linalg import expm as linalg_expm

from alpsqutip.operators.basic import Operator, ScalarOperator
from alpsqutip.operators.functions import commutator
from alpsqutip.scalarprod.build import fetch_HS_scalar_product
from alpsqutip.scalarprod.gram import gram_matrix
from alpsqutip.scalarprod.utils import find_linearly_independent_rows


class OperatorBasis:
    """
    Represent a basis of a subspace of the operator algebra with a
    metric given by a scalar product function.

    If a generator is given, the basis stores an array hij, which
    defines the evolution of the coefficients `coeff_a` in the
    expansion of an operator $K$

    K = sum_a coeff_a(t) Q_a

    in a way that Q

    dK
    -- = -i [H, K]
    dt

    The __add__ operator allows to extend the basis by adding
    more operators.
    """

    operator_basis: Tuple[Operator, ...]
    sp: Callable
    generator: Optional[Operator]
    gram: NDArray
    gram_inv: NDArray
    errors: np.ndarray
    gen_matrix: np.ndarray

    def __init__(
        self,
        operators: Tuple[Operator, ...],
        generator: Optional[Operator] = None,
        sp: Optional[Callable] = None,
        n_body_projection: Callable = lambda x: x,
        precomputed_tensors: Optional[dict] = None,
    ):

        if generator is not None:
            if generator.isherm:
                generator = generator * 1j
            self.generator = cast(Operator, generator).simplify()

        if sp is None:
            sp = fetch_HS_scalar_product()

        self.sp = sp

        if n_body_projection is not None:
            operators = tuple((n_body_projection(op_b) for op_b in operators))

        assert all(op_b.isherm for op_b in operators)
        self.operator_basis = operators

        if precomputed_tensors is not None:
            # Directly inject precomputed tensors (no recomputation)
            self.gram = precomputed_tensors["gram"]
            self.gram_inv = precomputed_tensors["gram_inv"]
            self.errors = precomputed_tensors["errors"]
            self.gen_matrix = precomputed_tensors["gen_matrix"]
        else:
            self.build_tensors()

    def __add__(self, other_basis):
        return append_basis(self, other_basis)

    def __radd__(self, other_basis):
        return prepend_basis(self, other_basis)

    def build_tensors(
        self, generator: Optional[Operator] = None, sp: Optional[Callable] = None
    ):
        """
        Build the arrays required to compute projections, expansions
        and evolutions

        Parameters
        ----------
        generator : Optional[Operator], optional
            The operator that generates the evolution. The default is None.
        sp : Optional[Callable], optional
            A scalar product. The default is None.

        Raises
        ------
        ValueError
            Raised if the basis elements does not span a non-trivial subspace.

        """

        if generator is not None:
            self.generator = generator
        else:
            generator = self.generator
        if sp is not None:
            self.sp = sp
        else:
            sp = self.sp

        operator_basis = self.operator_basis

        gram = gram_matrix(operator_basis, self.sp)

        # Cholesky decomposition
        # G = L . L^\dagger
        while operator_basis:
            try:
                l_gram = cholesky(gram)
                if all(abs(row[i]) > 1e-6 for i, row in enumerate(l_gram)):
                    break
            except LinAlgError:
                pass

            logging.warning(
                (
                    "using a non-independent set of operators. "
                    "Reduce it to a linearly independent set..."
                )
            )
            li_indx = find_linearly_independent_rows(gram)
            operator_basis_it = (operator_basis[i] for i in li_indx)
            operator_basis = tuple((op_b for op_b in operator_basis_it if op_b))
            gram = np.array([[gram[i, j] for i in li_indx] for j in li_indx])

            if not operator_basis:
                raise ValueError("No linear independent elements.")

            self.operator_basis = operator_basis

        self.gram = gram
        size = len(operator_basis)
        hij = np.zeros(
            (
                size,
                size,
            )
        )
        errors = np.zeros((size,))
        if self.generator is None:
            return

        # G^{-1} = (L^{-1})^\dagger . L^{-1}
        l_inv = inv(l_gram)
        self.gram_inv = l_inv.T @ l_inv

        def build_j_coefficients(op_2: Operator) -> Tuple[np.ndarray, np.float64]:
            comm = commutator(op_2, generator)
            error_sq = np.real(sp(comm, comm))
            hj = np.array([sp(op_1, comm) for op_1 in operator_basis])
            # |Pi_{\parallel} A|^2 = h^*_{ji}g^{-1}_{ik} h_{kj}
            # = |L^{-1}_{ik} h_{kj}|^2
            proj_coeffs = l_inv @ hj
            # errors_j = |Pi_{\perp} [H,Q_j]| =
            # sqrt(|[H,Q_j]|^2- | L_{ki} h_{ij}|^2)
            norm_par = proj_coeffs @ proj_coeffs
            error_sq = (max(error_sq - norm_par, 0)) ** 0.5
            return hj, error_sq

        # This loop is parallelizable:
        for j, op_2 in enumerate(operator_basis):
            hij[:, j], errors[j] = build_j_coefficients(op_2)

        self.gen_matrix = self.gram_inv @ hij
        self.errors = errors

    def coefficient_expansion(self, operator: Operator) -> NDArray:
        """
        Get the coefficients a_i s.t. the orthogonal projection
        of `operator` onto the basis is
        sum(a_i*b_i)

        Parameters
        ----------
        operator : Operator
            The operator to be decomposed on the basis elements.

        Returns
        -------
        NDArray
            the coeffients of the expansion.

        """
        sp = self.sp
        return self.gram_inv @ np.array(
            [sp(op, operator) for op in self.operator_basis]
        )

    def operator_from_coefficients(self, phi) -> Operator:
        """
        Build an operator from coefficients

        Parameters
        ----------
        phi : TYPE
            The coefficients of the expansion.

        Returns
        -------
        Operator
            The operator obtained from the components.

        """

        return sum(op_i * a_i for op_i, a_i in zip(self.operator_basis, phi))

    def project_onto(self, operator) -> Operator:
        """
        Project operator onto the subspace

        Parameters
        ----------
        operator : TYPE
            The operator to be projected.

        Returns
        -------
        Operator
            The projection of the operator in the subspace spanned by
            the basis.

        """

        return self.operator_from_coefficients(self.coefficient_expansion(operator))

    def evolve(self, t: float, a_0: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the coefficients for the expansion of the operator
        operator(t) = sum a_i(t) b_i
        evolving according the projected evolution,
        given its expansion at t=0, and the estimated error induced by
        the projection.

        Parameters
        ----------
        t : float
            DESCRIPTION.
        a_0 : np.ndarray
            DESCRIPTION.

        Returns
        -------
        Tuple(ndarray, float)
            Returns two ndarrays: the first with the evolved coefficient, and
            the second with the estimated error.

        """
        a_t = linalg_expm(t * self.gen_matrix) @ a_0
        # The error is estimated by
        # |\Delta K| = |\int_0^t \sum_a \Pi_{\perp}[H,Q_a] phi_a(\tau)d \tau  |
        #            <= \sum_a |\Pi_{\perp}[H,Q_a]| |phi_a(t)| t
        #
        return a_t, t * self.errors @ np.abs(a_t)


class HierarchicalOperatorBasis(OperatorBasis):
    """
    A HierarchicalOperatorBasis is a basis where
    the elements are linear combinations of iterated commutators
    of a seed element and the generator of the evolutions.
    """

    deep: int

    def __init__(
        self,
        seed: Operator,
        generator: Operator,
        deep: int = 1,
        sp: Optional[Callable] = None,
        n_body_projection: Callable = lambda x: x,
    ):
        if generator.isherm:
            generator = 1j * generator

        if sp is None:
            sp = fetch_HS_scalar_product()

        self.sp = sp
        self.generator = generator.simplify()
        self._build_basis(seed, deep, n_body_projection)
        self.build_tensors()

    def __add__(self, other):
        logging.warning(
            "Adding a HierarchicalBasis to another basis "
            "requires an explicit conversion."
        )
        return OperatorBasis(self.operator_basis, self.generator, self.sp) + other

    def _build_basis(self, seed, deep, projection_function=None):
        elements = [seed.simplify()]
        sp = self.sp
        generator = self.generator
        errors = np.zeros((deep,))
        for i in range(deep):
            new_elem = commutator(elements[-1], generator).simplify()
            comm_norm = np.abs(sp(new_elem, new_elem))
            if np.abs(comm_norm) < 1e-12:
                logging.warning(
                    (
                        f"""A commutator got (almost) zero norm. deep->"""
                        f"""{len(elements)}"""
                    )
                )
                deep = len(elements)
                elements.append(ScalarOperator(0, new_elem.system))
                errors = errors[:deep]
                break
            errors[i] = comm_norm
            new_elem = projection_function(new_elem)
            elements.append(new_elem)

        self.operator_basis = tuple(elements[:deep])
        gram = gram_matrix(elements, sp)
        self._hij = gram[:deep, 1:]
        self.gram = gram[:deep, :deep]
        self.errors = errors

    def build_tensors(
        self, generator: Optional[Operator] = None, sp: Optional[Callable] = None
    ):
        """
        Build the tensors required to compute projections and evolutions.

        Parameters
        ----------
        generator : Optional[Operator], optional
            The generator of the time evolution. The default is None.
        sp : Optional[Callable], optional
            The scalar product. The default is None.

        Returns
        -------
        None.

        """
        if generator is not None or sp is not None:
            logging.warning("A HierarchicalBasis cannot regenerate its elements.")

        # Loop to ensure that all the elements
        # in the basis are linearly independent.
        while self.operator_basis:
            try:
                gram = self.gram
                l_gram = cholesky(gram)
                break
            except LinAlgError:
                logging.warning(
                    (
                        "using a non-independent set of operators. "
                        "Reduce it to a linearly independent set..."
                    )
                )
            # Remove the last element and try again
            self.operator_basis = self.operator_basis[:-1]
            self.gram = gram[:-1, :-1]
            self._hij = self._hij[:-1, :-1]
            self.errors = self.errors[:-1]

        hij = self._hij
        errors = self.errors

        l_inv = inv(l_gram)
        self.gram_inv = l_inv.T @ l_inv

        for j, row in enumerate(hij):
            proj_coeffs = l_inv @ row
            norm_par = proj_coeffs @ proj_coeffs
            errors[j] = (max(errors[j] - norm_par, 0)) ** 0.5

        self.errors = errors
        self.gen_matrix = self.gram_inv @ hij


def append_basis(basis_1: OperatorBasis, basis_2: OperatorBasis | Iterable[Operator]):
    """
    Build a new basis with the elements of basis_1 and the
    elements of basis_2, given preference to the elements in basis_1.
    Efficiently reuses precomputed tensors from basis_1.
    """

    # If both basis are identical, return one of them.
    if basis_1 is basis_2:
        return basis_1

    sp: Callable = basis_1.sp

    ops1 = basis_1.operator_basis
    basis_1_generator = basis_1.generator
    if isinstance(basis_2, OperatorBasis):
        ops2 = basis_2.operator_basis
        basis_2_generator = basis_2.generator
        basis_2_gram = basis_2.gram
        basis_2_sp = basis_2.sp
        basis_2_gen_matrix = basis_2.gen_matrix
        basis_2_errors = basis_2.errors
    else:
        ops2 = tuple(basis_2)
        basis_2_generator = None
        basis_2_gram = None
        basis_2_sp = None
        basis_2_gen_matrix = None
        basis_2_errors = None

    same_sp: bool = sp is basis_2_sp
    generator = basis_1_generator or basis_2_generator
    operators = ops1 + ops2

    # --- Gram matrix blocks ---
    g11 = basis_1.gram  # (n1, n1)
    g11_inv = basis_1.gram_inv

    if same_sp:
        g22 = basis_2_gram
    else:
        n2 = len(ops2)
        g22 = np.empty((n2, n2), dtype=g11.dtype)
        for i_idx, o2a in enumerate(ops2):
            for j_idx, o2b in enumerate(ops2):
                g22[i_idx, j_idx] = sp(o2a, o2b)

    def merge_gram(
        g11, g11_inv, g22
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, Tuple[int]]:
        """
        Build the gram and gram_inv tensors from the available information.
        """

        # Build the new Gram matrix
        n_1 = len(g11)
        n_2 = len(g22)
        n_total = n_1 + n_2
        g12 = np.empty(
            (
                n_1,
                n_2,
            ),
            dtype=g11.dtype,
        )
        for i_idx, o1 in enumerate(ops1):
            for j_idx, o2 in enumerate(ops2):
                g12[i_idx, j_idx] = sp(o1, o2)
        g21 = g12.T
        gram_full = np.block([[g11, g12], [g21, g22]])

        # If gram is singular, reduce it and remove the
        # linearly dependent elements.
        li_indices = find_linearly_independent_rows(gram_full)
        if len(li_indices) != n_total:
            n_total = len(li_indices)
            if n_total == n_1:
                return g11, g11_inv, g11, g11, li_indices
        n_2 = n_total - n_1
        gram_full = gram_full[li_indices, :][:, li_indices]
        g12 = gram_full[:n_1, n_1:]
        g21 = g12.T
        g22 = gram_full[n_1:, n_1:]

        # --- Gram inverse (block inversion, Schur complement) ---
        # Should not be singular, because we ensure that gram is not
        # singular...
        shur = g22 - g21 @ g11_inv @ g12
        shur_inv = np.linalg.inv(shur)
        # Build the inverse
        top_left = g11_inv + g11_inv @ g12 @ shur_inv @ g21 @ g11_inv
        top_right = -g11_inv @ g12 @ shur_inv
        bottom_left = -shur_inv @ g21 @ g11_inv
        bottom_right = shur_inv
        gram_full_inv = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        return gram_full, gram_full_inv, g11, g22, li_indices

    gram, gram_inv, g11, g22, li_indices = merge_gram(g11, g11_inv, g22)
    n1, n2, n = len(g11), len(g22), len(gram)
    if n == n1:
        return basis_1
    if len(operators) != n:
        operators = tuple((operators[idx] for idx in li_indices))

    # Now, if generator is None, build the basis and return
    if generator is None:
        return OperatorBasis(
            operators,
            generator,
            sp,
            precomputed_tensors=dict(
                gram=gram,
                gram_inv=gram_inv,
                errors=np.zeros((n,)),
                gen_matrix=np.zeros(
                    (
                        n,
                        n,
                    )
                ),
                hij=np.zeros(
                    (
                        n,
                        n,
                    )
                ),
            ),
        )

    # Build gen_matrix and errors

    def prepare_blocks(
        gen_block: Optional[NDArray],
        ops: Tuple[Operator, ...],
        gram_block: NDArray,
        errors: Optional[NDArray],
        n_block: int,
        reuse: bool,
        rows_it,
    ) -> Tuple[NDArray, NDArray, Tuple[Operator, ...]]:
        """Prepare the diagonal blocks"""
        if reuse:
            if n_block != len(gram_block):
                rows_li = tuple(rows_it)
                ops = tuple(ops[idx] for idx in rows_li)
                gen_block = cast(NDArray, gen_block)[rows_li, :][:, rows_li]

            hij_block = gram_block @ gen_block
            error_sq = cast(NDArray, errors) ** 2 + np.array(
                [
                    cast(NDArray, gen_block)[:, idx] @ hij_block[:, idx]
                    for idx in range(n_block)
                ]
            )
            return hij_block, error_sq, ops

        # If not reuse, just remove the ld operators from ops and return empty blocks.
        if n_block != len(gram_block):
            ops = tuple(ops[idx] for idx in rows_it)
        return (
            np.empty(
                (
                    n_block,
                    n_block,
                ),
                dtype=g11.dtype,
            ),
            np.empty((n_block,), dtype=g11.dtype),
            ops,
        )

    def fill_h_blocks(b_1, b_2, h_diag, error_sq, reuse):
        "Compute h12, err_1_sq and h11 if needed."
        hij_off = np.empty(
            (
                len(b_2),
                len(b_1),
            ),
            dtype=g11.dtype,
        )
        for j_idx, op_j in enumerate(b_1):
            comm = commutator(op_j, generator)
            if not reuse:
                error_sq[j_idx] = sp(comm, comm)
                for i_idx, op_i in enumerate(b_1):
                    h_diag[i_idx, j_idx] = sp(op_i, comm)
            for i_idx, op_i in enumerate(b_2):
                hij_off[i_idx, j_idx] = sp(op_i, comm)
        return hij_off

    reuse_h11 = generator is basis_1.generator
    reuse_h22 = same_sp and generator is basis_2_generator

    hij11, error_1_sq, ops1 = prepare_blocks(
        basis_1.gen_matrix,
        ops1,
        g11,
        basis_1.errors,
        n1,
        reuse_h11,
        (idx for idx in li_indices if idx < n1),
    )
    hij22, error_2_sq, ops2 = prepare_blocks(
        basis_2_gen_matrix,
        ops2,
        g22,
        basis_2_errors,
        n2,
        reuse_h22,
        (idx - n1 for idx in li_indices if idx >= n1),
    )
    hij21 = fill_h_blocks(ops1, ops2, hij11, error_1_sq, reuse_h11)
    hij12 = fill_h_blocks(ops2, ops1, hij22, error_2_sq, reuse_h22)

    hij = np.block([[hij11, hij12], [hij21, hij22]])
    genij = gram_inv @ hij
    errors = (
        np.block([error_1_sq, error_2_sq])
        - np.array([genij[:, i] @ hij[:, i] for i in range(n)])
    ) ** 0.5

    return OperatorBasis(
        operators,
        generator,
        sp,
        precomputed_tensors=dict(
            gram=gram, gram_inv=gram_inv, errors=errors, gen_matrix=genij, hij=hij
        ),
    )


def append_basis_old(
    basis_1: OperatorBasis, basis_2: OperatorBasis | Iterable[Operator]
):
    """
    Build a new basis with the elements of basis_1 and the
    elements of basis_2, given preference to the elements in
    basis_1.
    """
    # TODO: reuse the already build `basis_1.gram`,
    # `basis_1.gram_inv`, `basis_1.gen_matrix`, and `basis_1.errors`
    # to avoid recompute scalar products and projections.
    sp = basis_1.sp
    operators = basis_1.operator_basis
    generator = basis_1.generator

    if isinstance(basis_2, OperatorBasis):
        operators = operators + basis_2.operator_basis
    else:
        operators = operators + tuple(basis_2)
    return OperatorBasis(operators, generator, sp)


def prepend_basis(basis_1: OperatorBasis, basis_2: OperatorBasis | Iterable[Operator]):
    """
    Build a new basis with the elements of basis_1 and the
    elements of basis_2, given preference to the elements in
    basis_2.
    """
    # TODO: reuse the already build `basis_1.gram`,
    # `basis_1.gram_inv`, `basis_1.gen_matrix`, and `basis_1.errors`
    # to avoid recompute scalar products and projections.
    sp = basis_1.sp
    operators = basis_1.operator_basis
    generator = basis_1.generator

    if isinstance(basis_2, OperatorBasis):
        operators = basis_2.operator_basis + operators
    else:
        operators = tuple(basis_2) + operators
    return OperatorBasis(operators, generator, sp)
