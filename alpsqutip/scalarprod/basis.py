"""
Basis of Operator metric sub-spaces
"""

import logging
from typing import Callable, Iterable, Optional, Tuple

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
        operators: Tuple[Operator],
        generator: Optional[Operator] = None,
        sp: Optional[Callable] = None,
        n_body_projection: Callable = lambda x: x,
        precomputed_tensors: Optional[dict] = None,
    ):

        if generator is not None and generator.isherm:
            generator = generator * 1j

        self.generator = generator.simplify()
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

        self.operator_basis = elements[:deep]
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
    elements of basis_2, given preference to the elements in
    basis_1.
    """
    # TODO: reuse the already build `basis_1.gram`,
    # `basis_1.gram_inv`, `basis_1.gen_matrix`, and `basis_1.errors`
    # to avoid recompute scalar products and projections.
    sp = basis_1.sp
    operators = basis_1.operators
    generator = basis_1.generator

    if isinstance(basis_2, OperatorBasis):
        operators = operators + basis_2.operators
    else:
        operators = operators + basis_2
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
    operators = basis_1.operators
    generator = basis_1.generator

    if isinstance(basis_2, OperatorBasis):
        operators = basis_2.operators + operators
    else:
        operators = basis_2 + operators
    return OperatorBasis(operators, generator, sp)
