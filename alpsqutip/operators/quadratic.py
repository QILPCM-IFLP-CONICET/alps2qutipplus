"""
Define SystemDescriptors and different kind of operators
"""

# from numbers import Number
from time import time

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
        assert isinstance(basis, tuple)
        assert isinstance(weights, tuple)
        assert isinstance(offset, Operator) or offset is None
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
        if isinstance(other, ScalarOperator):
            prefactor = other.prefactor
            offset = self.offset
            if offset is not None:
                offset = offset * prefactor
            return QuadraticFormOperator(
                self.basis,
                tuple(w * prefactor for w in self.weights),
                self.system,
                offset,
            )
        standard_repr = self.to_sum_operator().simplify()
        return  standard_repr * other

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
        return all(isinstance(weight, (int, float)) for weight in self.weights)

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
        """Simplify the operator"""
        return self

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


def build_quadratic_operator(operator, isherm=None):
    """
    Simplify the operator and try to decompose it
    as a Quadratic Form.
    """
    operator = operator.simplify()
    if isinstance(operator, QuadraticFormOperator):
        return operator

    # First, classify terms
    terms = operator.flat().terms if isinstance(operator, SumOperator) else [operator]
    terms_dict = {None:[], "1":[t for t in terms if isinstance(t, (ScalarOperator, LocalOperator, OneBodyOperator))], "2":[]}
    terms = (t for t in terms if not isinstance(t, (ScalarOperator, LocalOperator, OneBodyOperator)))

    def key_func(t):
        if not isinstance(t, ProductOperator):
            return None
        act_over = t.act_over()
        if act_over is None:
            return None
        size = len(act_over)
        if size<2:
            return "1"
        if size==2:
            return "2"
        return None
    
    for term in terms:
        terms_dict[key_func(term)].append(term)
    terms = None

    # Process terms
    
    # If no two-body terms are collected, return the original operator
    if len(terms_dict["2"])==0:
        return operator

    # parameters
    if isherm or operator.isherm:
        isherm = True
    system = operator.system

    one_body_terms = ([OneBodyOperator(tuple(terms_dict["1"]), system)]
                     if len(terms_dict["1"])>1 else terms_dict["1"])

    # Decomposing two-body terms
    basis = []
    weights = []
    basis_a = []
    weights_a = []
    
    for term in terms_dict["2"]:
        if not isinstance(term, ProductOperator):
            other_terms.append(term)
            continue
        
        prefactor = term.prefactor
        if prefactor == 0:
            continue
            
        op1, op2 = (LocalOperator(site, l_op, system) for site, l_op in term.sites_op.items())
        op2_dag = op2.dag()
        weights.extend((prefactor*.25,-prefactor*.25,))
        basis.extend((op1+op2_dag,op1-op2_dag,))
        if not isherm:
            weights_a.extend((prefactor*.25j,-prefactor*.25j,))
            basis_a.extend((op1+op2_dag*1j, op1-op2_dag*1j,))

    # Anti-hermitician terms at the end...
    if not isherm:
        basis.extend(basis_a)
        weights_a.extend(weights_a)
        basis_a, weighs_a = None, None
    
    
    if one_body_terms:
        result = QuadraticFormOperator(tuple(basis), tuple(weights), system=system, offset=one_body_terms[0])
    else:
        result = QuadraticFormOperator(tuple(basis), tuple(weights), system=system)

    result = result.simplify()
        
    other_terms = ([SumOperator(tuple(terms_dict[None]), system)]
                     if len(terms_dict[None])>1 else terms_dict[None])
    
    if other_terms:
        result = result + other_terms[0]

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
    hermitic=True,
    scalar_product=hs_scalar_product,
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


def build_quadratic_form_from_operator(
    operator: Operator, system=None, simplify: bool = True, herm: bool = True
):
    """
    Try to bring operator to a quadratic form object.
    """
    return build_quadratic_operator(operator, herm)

    def scalar_to_quadratic_form(scalar):
        """process scalars"""
        if operator == 0:
            return QuadraticFormOperator([], [], system, False)

        if isinstance(scalar, complex) and (herm or operator.imag == 0):
            scalar = scalar.real

        return QuadraticFormOperator(
            (ScalarOperator(1, system),), (scalar,), system, None
        )

    def local_to_quadratic_form(local_operator):
        """process local operators"""
        site = local_operator.site
        loc_op = local_operator.operator

        return QuadraticFormOperator(
            (
                LocalOperator(site, loc_op + 0.5, system),
                LocalOperator(site, loc_op - 0.5, system),
            ),
            (
                0.5,
                -0.5,
            ),
            system,
        )

    def one_body_to_quadratic_form(ob_op):
        """process local operators"""
        return QuadraticFormOperator(
            (
                ob_op + 0.5,
                ob_op.dag() - 0.5,
            ),
            (
                0.5,
                -0.5,
            ),
            system,
            None,
        )

    def product_to_quadratic_form(prod_op):
        """Process product operators"""
        sites_op = prod_op.sites_op
        system = prod_op.system
        if len(sites_op) > 2:
            raise ValueError(
                "argument is not a quadratic form on local operators",
                type(prod_op),
                " with ",
                len(sites_op),
                "factors",
            )

        prefactor = prod_op.prefactor
        if len(sites_op) == 0:
            return scalar_to_quadratic_form(prefactor)
        if len(sites_op) == 1:
            site, loc_op = next(iter(sites_op.items()))
            return local_to_quadratic_form(LocalOperator(site, loc_op, system))

        factor1, factor2 = tuple(
            (
                site,
                op,
            )
            for site, op in sites_op.items()
        )

        prefactor = prefactor * 0.5
        if not factor1[1].isherm:
            if factor2[1].isherm:
                factor1, factor2 = factor2, factor1
            else:
                factor1 = factor1[0], factor1[1].dag()

        terms = (
            OneBodyOperator(
                (
                    LocalOperator(factor1[0], factor1[1], system),
                    LocalOperator(factor2[0], factor2[1] * prefactor, system),
                ),
                system,
                None,
            ),
            OneBodyOperator(
                (
                    LocalOperator(factor1[0], factor1[1], system),
                    LocalOperator(factor2[0], factor2[1] * (-prefactor), system),
                ),
                system,
                None,
            ),
        )
        weights = (
            0.5,
            -0.5,
        )
        return QuadraticFormOperator(terms, weights, system, None)

    def sum_to_quadratic_form(operator):
        terms = []
        weights = []
        offset = []
        for term in operator.terms:
            q_term = build_quadratic_form_from_operator(term, system, False, True)
            if q_term is None:
                return None
            terms.extend(q_term.terms)
            weights.extend(q_term.weights)
            if q_term.offset:
                offset.extend(q_term.offset)
        return QuadraticFormOperator(
            tuple(terms), tuple(weights), system, SumOperator(tuple(offset), system)
        )

    def subclass_to_quadratic_form(operator):
        for base_type, func in lookup_table_methods.items():
            if isinstance(operator, base_type):
                return func(operator)

        raise ValueError(
            "argument is not a quadratic form on local operators",
            type(operator),
        )

    def handle_non_hermitic_case(operator):
        real_part, imag_part = hermitian_and_antihermitian_parts(operator)
        real_part = simplify_sum_operator(real_part)
        imag_part = simplify_sum_operator(imag_part)

        if not bool(imag_part):
            return build_quadratic_form_from_operator(real_part, system, simplify, True)

        if not bool(real_part):
            return (
                build_quadratic_form_from_operator(imag_part, system, simplify, True)
                * 1j
            )

        result_re = build_quadratic_form_from_operator(
            real_part, system, simplify, True
        )

        result_im = build_quadratic_form_from_operator(
            imag_part, system, simplify, True
        )

        terms = result_re.terms + result_im.terms
        weights = result_re.weights + tuple(weight * 1j for weight in result_im.weights)
        return QuadraticFormOperator(terms, weights, system, None)

    lookup_table_methods = {
        int: scalar_to_quadratic_form,
        float: scalar_to_quadratic_form,
        complex: scalar_to_quadratic_form,
        QuadraticFormOperator: (lambda x: x),
        LocalOperator: local_to_quadratic_form,
        OneBodyOperator: one_body_to_quadratic_form,
        ProductOperator: product_to_quadratic_form,
        SumOperator: sum_to_quadratic_form,
    }

    if isinstance(operator, Operator):
        if system is None:
            system = operator.system

        # Reduce the general case to the hermitician case:
        herm = herm or operator.isherm
        if not herm:
            return handle_non_hermitic_case(operator)

        if simplify:
            operator = operator.simplify()
    # Now, process assuming operator is hermitician
    return lookup_table_methods.get(type(operator), subclass_to_quadratic_form)(
        operator
    )


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
