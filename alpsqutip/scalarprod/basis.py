"""
Basis of Operator metric sub-spaces
"""

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError, cholesky, inv, qr
from scipy.linalg import expm as linalg_expm

from alpsqutip.operators import Operator, ScalarOperator
from alpsqutip.operators.functions import commutator
from alpsqutip.scalarprod.build import fetch_HS_scalar_product
from alpsqutip.scalarprod.gram import gram_matrix


def find_linearly_independent_rows(mat: np.ndarray, tol: float = 1e-12) -> Tuple[int]:
    """
    Find indices of a maximal subset of linearly independent columns of the matrix.
    """
    _, inds = qr(mat, mode="r", pivoting=True)
    rank = np.linalg.matrix_rank(mat, tol=tol)
    # The first `rank` indices are linearly independent columns
    return tuple(sorted(inds[:rank]))


class OperatorBasis:
    """
    Represent a basis of a subspace of the operator algebra with a
    metric given by a scalar product function.

    If a generator is given,



    The __add__ operator allows to extend the basis by adding
    more operators.
    """

    operator_basis: Tuple[Operator]
    sp: Callable
    generator: Optional[Operator]
    gram: np.ndarray
    gram_inv: np.ndarray
    errors: np.ndarray
    gen: np.ndarray

    def __init__(
        self,
        operators: Tuple[Operator],
        generator: Optional[Operator] = None,
        sp: Optional[Callable] = None,
        n_body_projection: Callable = lambda x: x,
    ):

        if generator is not None and generator.isherm:
            generator = generator * 1j

        self.generator = generator
        if sp is None:
            sp = fetch_HS_scalar_product()

        self.sp = sp

        if n_body_projection is not None:
            operators = tuple((n_body_projection(op_b) for op_b in operators))

        assert all(op_b.isherm for op_b in operators)
        self.operator_basis = operators
        self.build_tensors()

    def __add__(self, other_basis):
        if isinstance(other_basis, OperatorBasis):
            other_basis = other_basis.operators
        elif isinstance(other_basis, Operator):
            other_basis = (other_basis,)

        return OperatorBasis(self.operators + other_basis, self.generator, self.sp)

    def build_tensors(
        self, generator: Optional[Operator] = None, sp: Optional[Callable] = None
    ):

        if generator is not None:
            self.generator = generator
        else:
            generator = self.generator
        if sp is not None:
            self.sp = sp
        else:
            sp = self.sp

        operator_basis = self.operator_basis

        self.gram = gram = gram_matrix(operator_basis, self.sp)
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

        # Cholesky decomposition
        # G = L . L^\dagger
        try:
            l_gram = cholesky(gram)
        except LinAlgError:
            logging.warning(
                (
                    "using a non-independent set of operators. "
                    "Reduce it to a linearly independent set..."
                )
            )
            ld_indx = find_linearly_independent_rows(gram)
            self.operator_basis = tuple((operator_basis[i] for i in ld_indx))
            return self.build_tensors()

        # G^{-1} = (L^{-1})^\dagger . L^{-1}
        l_inv = inv(l_gram)
        self.gram_inv = l_inv.T @ l_inv

        def build_j_coefficients(op_2: Operator) -> Tuple[np.ndarray, np.float64]:
            comm = commutator(op_2, generator)
            error_sq = np.real(sp(comm, comm))
            hj = np.array([sp(op_1, comm) for op_1 in operator_basis])
            # |Pi_{\parallel} A|^2 = h^*_{ji}g^{-1}_{ik} h_{kj} = |L^{-1}_{ik} h_{kj}|^2
            proj_coeffs = l_inv @ hj
            # errors_j = |Pi_{\perp} [H,Q_j]| = sqrt(|[H,Q_j]|^2- | L_{ki} h_{ij}|^2)
            norm_par = proj_coeffs @ proj_coeffs
            error_sq = (max(error_sq - norm_par, 0)) ** 0.5
            return hj, error_sq

        # This loop is parallelizable:
        for j, op_2 in enumerate(operator_basis):
            hij[:, j], errors[j] = build_j_coefficients(op_2)

        self.gen_matrix = self.gram_inv @ hij
        self.errors = errors

    def coefficient_expansion(self, operator: Operator):
        """
        Get the coefficients a_i s.t. the orthogonal projection
        of `operator` onto the basis is
        sum(a_i*b_i)
        """
        sp = self.sp
        return self.gram_inv @ np.array(
            [sp(op, operator) for op in self.operator_basis]
        )

    def operator_from_coefficients(self, phi):
        """Build an operator from coefficients"""
        return sum(op_i * a_i for op_i, a_i in zip(self.operator_basis, phi))

    def project_onto(self, operator):
        """
        Project operator onto the subspace
        """
        return self.operator_from_coefficients(self.coefficient_expansion(operator))

    def evolve(self, t: float, a_0: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the coefficients for the expansion of the operator
        operator(t) = sum a_i(t) b_i
        evolving according the projected evolution,
        given its expansion at t=0, and the estimated error induced by
        the projection.
        """
        a_t = linalg_expm(t * self.gen_matrix) @ a_0
        # The error is estimated by
        # |\Delta K| = |\int_0^t \sum_a \Pi_{\perp}[H,Q_a] phi_a(\tau)d \tau  |
        #            <= \sum_a |\Pi_{\perp}[H,Q_a]| |phi_a(t)| t
        #
        return a_t, t * self.errors @ abs(a_t)


class HierarchicalOperatorBasis(OperatorBasis):
    """
    A HierarchicalOperatorBasis is a basis where
    the elements are linear combinations of iterated commutators
    of a seed element and the generator of the evolutions.
    """

    comm_norms: np.ndarray
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
        self.generator = generator
        self._build_basis(seed, deep, n_body_projection)
        self.build_tensors()

    def __add__(self, other):
        logging.warning(
            "Adding a HierarchicalBasis to another basis requires an explicit conversion."
        )
        return OperatorBasis(self.operator_basis, self.generator, self.sp) + other

    def _build_basis(self, seed, deep, projection_function=None):
        print("deep", deep)
        elements = [seed]
        sp = self.sp
        generator = self.generator
        errors = np.zeros((deep,))
        for i in range(deep):
            new_elem = commutator(elements[-1], generator)
            comm_norm = abs(sp(new_elem, new_elem))
            if abs(comm_norm) < 1e-16:
                logging.warning(
                    f"A commutator got (almost) zero norm. deep->{len(elements)}"
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
        print("gram extended\n", gram)
        self._hij = gram[:deep, 1:]
        self.gram = gram[:deep, :deep]
        self.errors = errors

    def build_tensors(
        self, generator: Optional[Operator] = None, sp: Optional[Callable] = None
    ):
        if generator is not None or sp is not None:
            logging.warning("A HierarchicalBasis cannot regenerate its elements.")

        gram = self.gram
        hij = self._hij
        errors = self.errors
        try:
            l_gram = cholesky(gram)
        except LinAlgError:
            logging.warning(
                (
                    "using a non-independent set of operators. "
                    "Reduce it to a linearly independent set..."
                )
            )
            # Remove the last element and try again
            self.operator_basis = self.operator_basis[:-1]
            self.gram = self.gram[:-1, :-1]
            self._hij = self._hij[:-1, :-1]
            return self.build_tensors()

        l_inv = inv(l_gram)
        self.gram_inv = l_inv.T @ l_inv
        print("l_inv\n:", l_inv)
        print("hij\n:", hij)

        for j, row in enumerate(hij):
            print(f"row {j}:", row)
            proj_coeffs = l_inv @ row
            norm_par = proj_coeffs @ proj_coeffs
            errors[j] = (max(errors[j] - norm_par, 0)) ** 0.5

        self.errors = errors
        self.gen_matrix = self.gram_inv @ hij
