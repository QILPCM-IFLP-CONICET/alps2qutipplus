"""
Define SystemDescriptors and different kind of operators
"""

# from numbers import Number
from time import time
from typing import Callable, List, Union

import numpy as np
from numpy.linalg import eigh, svd
from numpy.random import random

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.states import (
    GibbsProductDensityOperator,
    MixtureDensityOperator,
    ProductDensityOperator,
)
from alpsqutip.scalarprod import gram_matrix

# from typing import Union


class QuadraticFormOperator(Operator):
    """
    Represents a two-body operator of the form
    sum_alpha w_alpha * Q_alpha^2
    with Q_alpha a local operator or a One body operator.
    """

    system: SystemDescriptor
    terms: list
    weights: list

    def __init__(self, basis, weights, system=None, offset=None):
        # If the system is not given, infer it from the terms
        if offset:
            offset = offset.simplify()
        assert isinstance(basis, tuple)
        assert isinstance(weights, tuple)
        assert (
            isinstance(offset, (OneBodyOperator, LocalOperator, ScalarOperator))
            or offset is None
        ), f"{type(offset)} should be an Operator"
        if system is None:
            for term in basis:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

        # If check_and_simplify, ensure that all the terms are
        # one-body operators and try to use the simplified forms
        # of the operators.

        self.weights = weights
        self.basis = basis
        self.system = system
        self.offset = offset
        self._simplified = False

    def __bool__(self):
        return len(self.weights) > 0 and any(self.weights) and any(self.basis)

    def __add__(self, other):
        assert isinstance(other, Operator), "other must be an operator."
        system = self.system or other.system
        if isinstance(other, QuadraticFormOperator):
            basis = self.basis + other.basis
            weights = self.weights + other.weights
            offset = self.offset
            if offset is None:
                offset = other.offset
            else:
                if other.offset is not None:
                    offset = offset + other.offset
            return QuadraticFormOperator(basis, weights, system, offset)
        if isinstance(
            other,
            (
                ScalarOperator,
                LocalOperator,
                OneBodyOperator,
            ),
        ):
            offset = self.offset
            if offset is None:
                offset = other
            else:
                offset = offset + other
            basis = self.basis
            weights = self.weights
            return QuadraticFormOperator(basis, weights, system, offset)
        return SumOperator(
            (
                self,
                other,
            ),
            system,
        )

    def __mul__(self, other):
        system = self.system
        if isinstance(other, ScalarOperator):
            other = other.prefactor
            system = system or other.system
        if isinstance(other, (float, complex)):
            offset = self.offset
            if offset is not None:
                offset = offset * other
            return QuadraticFormOperator(
                self.basis,
                tuple(w * other for w in self.weights),
                system,
                offset,
            )
        standard_repr = self.to_sum_operator(False).simplify()
        return standard_repr * other

    def __neg__(self):
        offset = self.offset
        if offset is not None:
            offset = -offset
        return QuadraticFormOperator(
            self.basis, tuple(-w for w in self.weights), self.system, offset
        )

    def acts_over(self):
        """
        Set of sites over the state acts.
        """
        offset = self.offset
        result = set() if offset is None else set(offset.acts_over())
        for term in self.basis:
            term_acts_over = term.acts_over()
            if term_acts_over is None:
                return None
            result = result.union(term_acts_over)
        return result

    def flat(self):
        return self

    @property
    def isdiagonal(self):
        """True if the operator is diagonal in the product basis."""
        offset = self.offset
        if offset:
            isdiag = offset.isdiagonal
            if isdiag is not True:
                return isdiag
        if all(term.isdiagonal for term in self.basis):
            return True
        return False

    @property
    def isherm(self):
        offset = self.offset
        if offset:
            isherm = offset.isherm
            if isherm is not True:
                return isherm

        return all(abs(np.imag(weight)) < 1e-10 for weight in self.weights)

    def partial_trace(self, sites: Union[list, SystemDescriptor]):
        if not isinstance(sites, SystemDescriptor):
            sites = self.system.subsystem(sites)

        if len(self.basis) == 0:
            if offset:
                return offset.partial_trace(sites)
            return ScalarOperator(0, sites)

        terms = tuple(
            w * (op_term * op_term).partial_trace(sites)
            for w, op_term in zip(self.weights, self.basis)
        )
        offset = self.offset
        if offset:
            terms = terms + (offset.partial_trace(sites),)
        return SumOperator(
            terms,
            sites,
        ).simplify()

    def simplify(self):
        """
        Simplify the operator.
        Build a new representation with a smaller basis.
        """
        operator = self
        if not all(b.isherm for b in self.basis):
            operator = ensure_hermitician_basis(operator)
        result = simplify_hermitician_quadratic_form(operator)
        if (
            len(result.basis) > len(self.basis)
            or len(result.basis) == len(self.basis)
            and self.offset is result.offset
        ):
            return self
        return result

    def to_qutip(self):
        result = sum(
            (w * op_term.dag() * op_term).to_qutip()
            for w, op_term in zip(self.weights, self.basis)
        )
        offset = self.offset
        if offset:
            result += offset.to_qutip()
        return result

    def to_sum_operator(self, simplify: bool = True) -> SumOperator:
        """Convert to a linear combination of quadratic operators"""
        isherm = self.isherm
        isdiag = self.isdiagonal
        if all(b_op.isherm for b_op in self.basis):
            terms = tuple(
                (
                    ((op_term * op_term) * w)
                    for w, op_term in zip(self.weights, self.basis)
                )
            )
        else:
            terms = tuple(
                (
                    ((op_term.dag() * op_term) * w)
                    for w, op_term in zip(self.weights, self.basis)
                )
            )

        offset = self.offset
        if offset:
            terms = terms + (offset,)
        result = SumOperator(terms, self.system, isherm, isdiag)
        if simplify:
            return result.simplify()
        return result


def build_quadratic_form_from_operator(operator, isherm=None, simplify=True):
    """
    Simplify the operator and try to decompose it
    as a Quadratic Form.
    """
    operator = operator.simplify()
    if isinstance(operator, QuadraticFormOperator):
        if isherm and not operator.isherm:
            offset = operator.offset
            if offset is not None and not offset.isherm:
                offset = SumOperator(
                    (
                        offset * 0.5,
                        offset.dag() * 0.5,
                    ),
                    operator.system,
                    isherm,
                )
                if simplify:
                    offset = offset.simplify()
            result = QuadraticFormOperator(
                operator.basis,
                tuple((np.real(w) for w in operator.weights)),
                operator.system,
                offset,
            )
        else:
            return operator.simplify() if simplify else operator
        return result

    # First, classify terms
    terms = operator.flat().terms if isinstance(operator, SumOperator) else [operator]
    terms_dict = {
        None: [],
        "1": [
            term
            for term in terms
            if isinstance(term, (ScalarOperator, LocalOperator, OneBodyOperator))
        ],
        "2": [],
    }
    terms = (
        term
        for term in terms
        if not isinstance(term, (ScalarOperator, LocalOperator, OneBodyOperator))
    )

    def key_func(t_op):
        if not isinstance(t_op, ProductOperator):
            return None
        acts_over = t_op.acts_over()
        if acts_over is None:
            return None
        size = len(acts_over)
        if size < 2:
            return "1"
        if size == 2:
            return "2"
        return None

    for term in terms:
        terms_dict[key_func(term)].append(term)
    terms = None

    # Process terms

    # If no two-body terms are collected, return the original operator
    if len(terms_dict["2"]) == 0:
        if isherm and not operator.isherm:
            operator = SumOperator(
                (
                    operator * 0.5,
                    operator.dag() * 0.5,
                ),
                operator.system,
                isherm=True,
            )
        if simplify:
            operator = operator.simplify()
        return operator

    # parameters
    if isherm or operator.isherm:
        isherm = True
    system = operator.system

    # Decomposing two-body terms
    basis: list = []
    weights: list = []
    basis_h: list = []
    weights_h: list = []
    other_terms: list = []

    for term in terms_dict["2"]:
        if not isinstance(term, ProductOperator):
            other_terms.append(term)
            continue

        prefactor = term.prefactor
        if prefactor == 0:
            continue

        op1, op2 = (
            LocalOperator(site, l_op, system) for site, l_op in term.sites_op.items()
        )
        op2_dag = op2 if op2.isherm else op2.dag()
        weights_h.extend(
            (
                prefactor * 0.25,
                -prefactor * 0.25,
            )
        )
        basis_h.extend(
            (
                op1 + op2_dag,
                op1 - op2_dag,
            )
        )
        if not isherm:
            weights.extend(
                (
                    prefactor * 0.25j,
                    -prefactor * 0.25j,
                )
            )
            basis.extend(
                (
                    op1 + op2_dag * 1j,
                    op1 - op2_dag * 1j,
                )
            )

    # Anti-hermitician terms at the begining...
    if isherm:
        basis = basis_h
        weights = weights_h
    else:
        basis.extend(basis_h)
        weights.extend(weights_h)

    # if the basis includes antihermitician terms, rewrite them
    # by a canonical transformation
    basis_h, weights_h = [], []
    for b_elem, w_factor in zip(basis, weights):
        if b_elem.isherm:
            basis_h.append(b_elem)
            weights_h.append(w_factor)
        else:
            b_h = (b_elem + b_elem.dag()).simplify()
            b_a = ((b_elem.dag() - b_elem) * 1j).simplify()
            if bool(b_h):
                basis_h.append(b_h)
                weights_h.append(0.25 * w_factor)
                if bool(b_a):
                    basis_h.append(b_a)
                    weights_h.append(0.25 * w_factor)
                    comm = ((b_h * b_a - b_a * b_h) * 0.25j).simplify()
                    if bool(comm):
                        terms_dict["1"].append(comm)
            elif bool(b_a):
                basis_h.append(b_h)
                weights_h.append(0.25 * w_factor)

    basis, weights = basis_h, weights_h
    basis_h, weights_h = None, None

    # Add all one body terms
    one_body_terms = (
        [OneBodyOperator(tuple(terms_dict["1"]), system)]
        if len(terms_dict["1"]) > 1
        else terms_dict["1"]
    )
    offset = one_body_terms[0] if one_body_terms else None

    if isherm and offset and not offset.isherm:
        offset = (offset + offset.dag()) * 0.5

    result = QuadraticFormOperator(
        tuple(basis), tuple(weights), system=system, offset=offset
    )

    if simplify:
        result = result.simplify()

    other_terms = (
        [SumOperator(tuple(terms_dict[None]), system)]
        if len(terms_dict[None]) > 1
        else terms_dict[None]
    )
    rest = other_terms[0] if other_terms else None
    if rest:
        if isherm and rest and not rest.isherm:
            rest = SumOperator(
                (
                    rest * 0.5,
                    rest.dag() * 0.5,
                ),
                system,
                isherm,
            )
            if simplify:
                rest = rest.simplify()
        result = result + rest
    return result


def quadratic_form_expect(sq_op, state):
    """
    Compute the expectation value of op, taking advantage
    of its structure.
    """
    sq_op = sq_op.to_sum_operator(False)
    return state.expect(sq_op)


def selfconsistent_meanfield_from_quadratic_form(
    quadratic_form: QuadraticFormOperator, max_it, logdict=None
):
    """
    Build a self-consistent mean field approximation
    to the gibbs state associated to the quadratic form.
    """
    #    quadratic_form = simplify_quadratic_form(quadratic_form)
    system = quadratic_form.system
    terms = quadratic_form.terms
    weights = quadratic_form.weights

    operators = [2 * w * b for w, b in zip(weights, terms)]
    basis = [b for w, b in zip(weights, terms)]

    phi = [(2.0 * random() - 1.0)]

    evolution = []
    timestamps = []

    if isinstance(logdict, dict):
        logdict["states"] = evolution
        logdict["timestamps"] = timestamps

    remaining_iterations = max_it
    while remaining_iterations:
        remaining_iterations -= 1
        k_exp = OneBodyOperator(
            tuple(phi_i * operator for phi_i, operator in zip(phi, basis)),
            system,
        )
        k_exp = ((k_exp + k_exp.dag()).simplify()) * 0.5
        assert k_exp.isherm
        rho = GibbsProductDensityOperator(k_exp, 1.0, system)
        new_phi = -rho.expect(operators).conj()
        if isinstance(logdict, dict):
            evolution.append(new_phi)
            timestamps.append(time())

        change = sum(
            abs(old_phi_i - new_phi_i) for old_phi_i, new_phi_i in zip(new_phi, phi)
        )
        if change < 1.0e-10:
            break
        phi = new_phi

    return rho


def ensure_hermitician_basis(self: QuadraticFormOperator):
    """
    Ensure that the quadratic form is expanded using a
    basis of hermitician operators.
    """
    basis = self.basis
    if all(b.isherm for b in basis):
        return self

    coeffs = self.weights
    system = self.system
    offset = self.offset

    # Reduce the basis to an hermitician basis
    new_basis = []
    new_coeffs = []
    local_terms = []
    for la_coeff, b_op in zip(coeffs, basis):
        if b_op.isherm:
            new_basis.append(b_op)
            new_coeffs.append(la_coeff)
            continue
        # Not hermitician. Decompose as two hermitician terms
        # and an offset
        if la_coeff == 0:
            continue
        b_h = ((b_op + b_op.dag()) * 0.5).simplify()
        b_a = ((b_op - b_op.dag()) * 0.5j).simplify()
        if b_h:
            new_basis.append(b_h)
            new_coeffs.append(la_coeff)
            if b_a:
                new_basis.append(b_a)
                new_coeffs.append(la_coeff)
                comm = ((b_h * b_a - b_a * b_h) * (1j * la_coeff)).simplify()
                if comm:
                    local_terms.append(comm)
        elif b_a:
            new_basis.append(b_a)
            new_coeffs.append(la_coeff)

    local_terms = [term for term in local_terms if term]
    if offset is not None:
        local_terms = [offset] + local_terms
    if local_terms:
        new_offset = sum(local_terms).simplify()

    if not bool(new_offset):
        new_offset = None
    return QuadraticFormOperator(
        tuple(new_basis), tuple(new_coeffs), system, new_offset
    )


def one_body_operator_hermitician_hs_sp(x_op: OneBodyOperator, y_op: OneBodyOperator):
    """
    Hilbert Schmidt scalar product optimized for OneBodyOperators
    """
    result = 0
    terms_x = x_op.terms if isinstance(x_op, OneBodyOperator) else (x_op,)
    terms_y = y_op.terms if isinstance(y_op, OneBodyOperator) else (y_op,)

    for t_1 in terms_x:
        for t_2 in terms_y:
            if isinstance(t_1, ScalarOperator):
                result += t_2.tr() * t_1.prefactor
            elif isinstance(t_2, ScalarOperator):
                result += t_1.tr() * t_2.prefactor
            elif t_1.site == t_2.site:
                result += (t_1.operator * t_2.operator).tr()
            else:
                result += t_1.operator.tr() * t_2.operator.tr()
    return result


def simplify_quadratic_form(
    operator: QuadraticFormOperator,
    hermitic: bool = True,
    scalar_product: Callable = one_body_operator_hermitician_hs_sp,
):
    """
    Takes a 2-body operator and returns lists weights, ops
    such that the original operator is
    sum(w * op.dag()*op for w,op in zip(weights,ops))
    """
    operator = ensure_hermitician_basis(operator)
    if hermitic:
        operator = QuadraticFormOperator(
            operator.basis,
            tuple((np.real(f) for f in operator.weights)),
            operator.system,
            offset=operator.offset,
        )
    return simplify_hermitician_quadratic_form(operator, scalar_product)


def simplify_hermitician_quadratic_form(
    self: QuadraticFormOperator, sp_call: Callable = one_body_operator_hermitician_hs_sp
):
    """
    Assuming that the basis is hermitician, and the coefficients are real,
    find another representation using a smaller basis.
    """
    basis = tuple((b.simplify() for b in self.basis))
    coeffs = self.weights
    system = self.system
    offset = self.offset
    if offset:
        offset = offset.simplify()

    def reduce_real_case(real_basis, real_coeffs):
        # remove null coeffs:
        real_basis = tuple((b for c, b in zip(real_coeffs, real_basis) if c))
        real_coeffs = [c for c in real_coeffs if abs(c) > 1e-10]
        if len(real_coeffs) == 0:
            return tuple(), tuple()

        # first build the gramm matrix
        gram_mat = gram_matrix(real_basis, sp_call)

        # and its SVD decomposition
        u_mat, s_mat, vd_mat = svd(gram_mat, full_matrices=False, hermitian=True)

        # t is a matrix that builds a reduced basis of hermitician operators
        # from the original generators

        t_mat = np.array(
            [row * s ** (-0.5) for row, s in zip(vd_mat, s_mat) if s > 1e-10]
        )

        if len(t_mat) == 0:
            return tuple(), tuple()

        # Then, we build the change back to the hermitician basis
        q_mat = np.array(
            [row * s ** (0.5) for row, s in zip(u_mat.T, s_mat) if s > 1e-10]
        ).T
        # Now, we build the (non-diaognal) quadratic form in the new reduced basis
        new_qf = (q_mat.T * real_coeffs).dot(q_mat)
        # and diagonalize it
        real_coeffs, evecs = eigh(new_qf)
        evecs = evecs[:, abs(real_coeffs) > 1e-10]
        real_coeffs = real_coeffs[abs(real_coeffs) > 1e-10]
        # the optimized expansion is the product of the matrix of eigenvectors times
        # the basis reduction
        t_mat = evecs.T.dot(t_mat)

        # finally, rebuild the basis. This is the most expensive part
        real_basis = tuple(
            (
                OneBodyOperator(
                    tuple((op * c for c, op in zip(row, real_basis) if abs(c) > 1e-10)),
                    system,
                ).simplify()
                for row in t_mat
            )
        )
        return real_basis, tuple(real_coeffs)

    if any(hasattr(z, "imag") for z in coeffs):
        coeffs_r = tuple((np.real(z) for z in coeffs))
        coeffs_i = tuple((np.imag(z) for z in coeffs))
        basis_r, coeffs_r = reduce_real_case(basis, coeffs_r)
        basis_i, coeffs_i = reduce_real_case(basis, coeffs_i)
        basis = basis_r + basis_i
        coeffs = coeffs_r + tuple((1j * c for c in coeffs_i))
    else:
        basis, coeffs = reduce_real_case(basis, coeffs)

    result = QuadraticFormOperator(basis, coeffs, system, offset)

    return result


# #####################
#
#  Arithmetic
#
# #######################


@Operator.register_add_handler(
    (
        ScalarOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_add_handler(
    (
        LocalOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_add_handler(
    (
        ProductOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_add_handler(
    (
        SumOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_add_handler(
    (
        OneBodyOperator,
        QuadraticFormOperator,
    )
)
def _(op1: Operator, op2: QuadraticFormOperator):
    return op2 + op1


# Products right products


@Operator.register_mul_handler(
    (
        ScalarOperator,
        QuadraticFormOperator,
    )
)
def _(op1: ScalarOperator, op2: QuadraticFormOperator):
    return op2 * op1


@Operator.register_mul_handler(
    (
        LocalOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        SumOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        OneBodyOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductDensityOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        GibbsProductDensityOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        MixtureDensityOperator,
        QuadraticFormOperator,
    )
)
def _(op1: Operator, op2: QuadraticFormOperator):
    return op1 * op2.to_sum_operator()
