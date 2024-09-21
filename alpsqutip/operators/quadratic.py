"""
Define SystemDescriptors and different kind of operators
"""

# from numbers import Number
from time import time
from typing import Callable

import numpy as np
from numpy.linalg import eigh, svd
from numpy.random import random

from alpsqutip.model import SystemDescriptor
from alpsqutip.operator_functions import (
    hermitian_and_antihermitian_parts,
    simplify_sum_operator,
)
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.states import (
    GibbsProductDensityOperator,
    ProductDensityOperator,
    MixtureDensityOperator,
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
        assert isinstance(offset, (OneBodyOperator, LocalOperator, ScalarOperator)) or offset is None, f"{type(offset)} should be an Operator"
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
        standard_repr = self.to_sum_operator().simplify()
        return standard_repr * other

    def __neg__(self):
        offset = self.offset
        if offset is not None:
            offset = -offset
        return QuadraticFormOperator(
            self.basis, tuple(-w for w in self.weights), self.system, offset
        )

    def act_over(self):
        """
        Set of sites over the state acts.
        """
        offset = self.offset
        result = set() if offset is None else set(offset.act_over())
        for term in self.basis:
            term_act_over = term.act_over()
            if term_act_over is None:
                return None
            result = result.union(term_act_over)
        return result

    @property
    def isdiag(self):
        offset = self.offset
        if offset:
            isdiag = offset.isdiag
            if isdiag is not True:
                return isdiag
        if all(term.isdiag for term in self.basis):
            return True
        return None

    @property
    def isherm(self):
        offset = self.offset
        if offset:
            isherm = offset.isherm
            if isherm is not True:
                return isherm

        return all(abs(np.imag(weight))<1e-10 for weight in self.weights)

    def partial_trace(self, sites):
        terms = tuple(
            w * (op_term * op_term).partial_trace(sites)
            for w, op_term in zip(self.weights, self.basis)
        )
        offset = self.offset
        if offset:
            terms = terms + (offset.partial_trace(sites),)
        return SumOperator(
            terms,
            self.system,
        ).simplify()

    def simplify(self):
        """
        Simplify the operator.
        Build a new representation with a smaller basis.
        """
        operator = self
        assert all(b.isherm for b in self.basis)
        if not all(b.isherm for b in self.basis):
            return simplify_hermitician_quadratic_form(ensure_hermitician_basis(self))
        result = simplify_hermitician_quadratic_form(self)
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

    def to_sum_operator(self, symplify: bool = True) -> SumOperator:
        """Convert to a linear combination of quadratic operators"""
        result = sum(
            (w * op_term.dag() * op_term)
            for w, op_term in zip(self.weights, self.basis)
        )
        offset = self.offset
        if offset:
            result = result + offset
        if symplify:
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
                offset = SumOperator((offset*.5, offset.dag()*.5,), operator.system, isherm)
                if simplify:
                    offset = offset.simplify()
            result = QuadraticFormOperator(operator.basis, tuple((np.real(w) for w in operator.weights)) , operator.system, offset)
        else:
            return operator.simplify() if simplify else operator
        return result

    # First, classify terms
    terms = operator.flat().terms if isinstance(operator, SumOperator) else [operator]
    terms_dict = {
        None: [],
        "1": [
            t
            for t in terms
            if isinstance(t, (ScalarOperator, LocalOperator, OneBodyOperator))
        ],
        "2": [],
    }
    terms = (
        t
        for t in terms
        if not isinstance(t, (ScalarOperator, LocalOperator, OneBodyOperator))
    )

    def key_func(t):
        if not isinstance(t, ProductOperator):
            return None
        act_over = t.act_over()
        if act_over is None:
            return None
        size = len(act_over)
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
            operator = SumOperator((operator*.5,operator.dag()*.5,),operator.system, isherm=True)
        if simplify:
            operator = operator.simplify()
        return operator

    # parameters
    if isherm or operator.isherm:
        isherm = True
    system = operator.system


    # Decomposing two-body terms
    basis = []
    weights = []
    basis_h = []
    weights_h = []

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

    # Anti-hermitician terms at the end...
    if isherm:
        basis = basis_h
        weights = weights_h
    else:
        basis.extend(basis_h)
        weights.extend(weights_h)

    # if the basis includes antihermitician terms, rewrite them
    # by a canonical transformation
    basis_h, weights_h = [], []
    for b, w in zip(basis, weights):
        if b.isherm:
            basis_h.append(b)
            weights_h.append(w)
        else:
            b_h = (b + b.dag()).simplify()
            b_a = ((b.dag()-b)*1j).simplify()
            if bool(b_h):
                basis_h.append(b_h)
                weights_h.append(.25*w)
                if bool(b_a):
                    basis_h.append(b_h)
                    weights_h.append(.25*w)
                    comm = ((b_h*b_a-b_a*b_h)*.25j).simplify()
                    if bool(comm):
                        terms_dict["1"].append(comm)
            elif bool(b_a):
                basis_h.append(b_h)
                weights_h.append(.25*w)

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
        offset = (offset+offset.dag())*.5

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
            rest = SumOperator((rest*.5,rest.dag()*.5,), system, isherm)
            if simplify:
                rest = rest.simplify()
        result = result + rest
    return result


def hs_scalar_product(o_1, o_2):
    """HS scalar product"""
    return (o_1.dag() * o_2).tr()


def matrix_change_to_orthogonal_basis(
    basis: list, scalar_product=hs_scalar_product, threshold=1.0e-10
):
    """
    Build the coefficient matrix of the base change to an orthogonal base.
    """
    gram = gram_matrix(basis, scalar_product)
    # pylint: disable=unused-variable
    # TODO: Check if there is a way to obtain just the left singular vectors.
    left_sv, s_diag, v_h = svd(gram, hermitian=True, full_matrices=False)
    kappa = len([sv for sv in s_diag if sv > threshold])
    v_h = v_h[:kappa]
    return v_h.conj()


def simplify_quadratic_form(
    operator: QuadraticFormOperator,
    hermitic: bool = True,
    scalar_product: Callable = hs_scalar_product,
):
    """
    Takes a 2-body operator and returns lists weights, ops
    such that the original operator is
    sum(w * op.dag()*op for w,op in zip(weights,ops))
    """
    local_ops = operator.terms
    coeffs = operator.weights
    system = operator.system
    offset = operator.offset
    hermitic = hermitic or operator.isherm

    # Orthogonalize the basis
    u_transp = matrix_change_to_orthogonal_basis(
        local_ops, scalar_product=scalar_product
    )
    u_dag = u_transp.conj()

    # reduced_basis = [ sum(c*old_op  for c, old_op in zip(row,local_ops) )
    #                   for row in u_transp]
    # Build the coefficients of the quadratic form

    def sort_by_weight(weights, rows):
        if len(rows) < 2:
            return list(weights), rows
        w_and_rows = zip(
            *sorted(
                ([weight, row] for weight, row in zip(weights, rows)),
                key=lambda x: x[0],
            )
        )
        return tuple(list(data) for data in w_and_rows)

    def new_weight_and_basis(coeff_matrix):
        weights, eig_vecs = eigh(coeff_matrix)
        # Remove null eigenvalues
        support = abs(weights) > 1.0e-10
        v_transp = eig_vecs.transpose()[support]
        weights = weights[support]

        # Build the new set of operators as the composition
        # of the two basis changes: the one that reduces the basis
        # by orthogonalizing a metric (u_transp) and the one
        # that diagonalizes the quadratic form in the new basis
        # (v_transp):
        rows = v_transp.conj().dot(u_transp)
        weights, rows = sort_by_weight(weights, rows)

        new_basis = tuple(
            OneBodyOperator(
                tuple(c * old_op for c, old_op in zip(row, local_ops)), system
            )
            for row in rows
        )
        return tuple(weights), new_basis

    weights, new_basis = new_weight_and_basis(
        (u_dag * np.real(coeffs)).dot(u_transp.transpose())
    )
    if not hermitic:
        weights_imag, new_basis_imag = new_weight_and_basis(
            (u_dag * np.imag(coeffs)).dot(u_transp.transpose())
        )
        weights = weights + tuple(weight * 1j for weight in weights_imag)
        new_basis = new_basis + new_basis_imag

    return QuadraticFormOperator(new_basis, weights, system, offset=offset)


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
    assert all(b.isherm for b in basis)
    return self

    
    coeffs = self.weights
    system = self.system
    offset = self.offset

    # Reduce the basis to an hermitician basis
    new_basis = []
    new_coeffs = []
    commut_terms = []
    local_terms = []
    for la, b in zip(coeffs, basis):
        if b.isherm:
            new_basis.append(b)
            new_coeffs.append(la)
            continue
        # Not hermitician. Decompose as two hermitician terms
        # and an offset
        if la == 0:
            continue
        b_h = ((b + b.dag()) * 0.5).simplify()
        b_a = ((b - b.dag()) * 0.5j).simplify()
        if b_h:
            new_basis.append(b_h)
            new_coeffs.append(la)
            if b_a:
                new_basis.append(b_a)
                new_coeffs.append(la)
                comm = ((b_h * b_a - b_a * b_h) * (1j * la)).simplify()
                if comm:
                    local_terms.append(comm)
        elif b_a:
            new_basis.append(b_a)
            new_coeffs.append(la)

    local_terms = [term for term in local_terms if term]
    if offset is not None:
        local_terms = [offset] + local_terms
    if local_terms:
        new_offset = sum(local_terms).simplify()

    if not (bool(new_offset)):
        new_offset = None
    return QuadraticFormOperator(
        tuple(new_basis), tuple(new_coeffs), system, new_offset
    )


def one_body_operator_hermitician_hs_sp(x: OneBodyOperator, y: OneBodyOperator):
    """
    Hilbert Schmidt scalar product optimized for OneBodyOperators
    """
    result = 0
    terms_x = x.terms if isinstance(x, OneBodyOperator) else (x,)
    terms_y = y.terms if isinstance(y, OneBodyOperator) else (y,)
    for t1, t2 in zip(terms_x, terms_y):
        if isinstance(t1, ScalarOperator):
            result +=t2.tr() * t1.prefactor
        elif isinstance(t2, ScalarOperator):
            result += t1.tr()*t2.prefactor 
        elif t1.site == t2.site:
            result += (t1.operator * t2.operator).tr()
        else:
            result += t1.operator.tr() * t2.operator.tr()
    return result


def simplify_hermitician_quadratic_form(
    self: QuadraticFormOperator, sp: Callable = one_body_operator_hermitician_hs_sp
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

    def reduce_real_case(real_basis, real_coeffs, sp):
        # remove null coeffs:
        real_basis = tuple((b for c, b in zip(real_coeffs, real_basis) if c))
        real_coeffs = [c for c in real_coeffs if abs(c) > 1e-10]
        if len(real_coeffs) == 0:
            return tuple(), tuple()

        # first build the gramm matrix
        gram_mat = gram_matrix(real_basis, sp)

        # and its SVD decomposition
        u_mat, s_mat, vd_mat = svd(gram_mat, full_matrices=False, hermitian=True)

        # t is a matrix that builds a reduced basis of hermitician operators
        # from the original generators

        t = np.array([row * s ** (-0.5) for row, s in zip(vd_mat, s_mat) if s > 1e-10])

        if len(t) == 0:
            return tuple(), tuple()

        # Then, we build the change back to the hermitician basis
        q = np.array(
            [row * s ** (0.5) for row, s in zip(u_mat.T, s_mat) if s > 1e-10]
        ).T
        # Now, we build the (non-diaognal) quadratic form in the new reduced basis
        new_qf = (q.T * real_coeffs).dot(q)
        # and diagonalize it
        real_coeffs, evecs = np.linalg.eigh(new_qf)
        evecs = evecs[:, abs(real_coeffs) > 1e-10]
        real_coeffs = real_coeffs[abs(real_coeffs) > 1e-10]
        # the optimized expansion is the product of the matrix of eigenvectors times
        # the basis reduction
        t = evecs.T.dot(t)

        # finally, rebuild the basis. This is the most expensive part
        real_basis = tuple(
            (
                OneBodyOperator(
                    tuple((op * c for c, op in zip(row, real_basis) if abs(c) > 1e-10)),
                    system,
                ).simplify()
                for row in t
            )
        )

        return real_basis, tuple(real_coeffs)

    if any(hasattr(z, "imag") for z in coeffs):
        coeffs_r = tuple((z.real if hasattr(z, "real") else z for z in coeffs))
        coeffs_i = tuple((z.imag if hasattr(z, "imag") else 0 for z in coeffs))
        basis_r, coeffs_r = reduce_real_case(basis, coeffs_r, sp)
        basis_i, coeffs_i = reduce_real_case(basis, coeffs_i, sp)
        basis = basis_r + basis_i
        coeffs = coeffs_r + tuple((1j * c for c in coeffs_i))
    else:
        basis, coeffs = reduce_real_case(basis, coeffs, sp)

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
