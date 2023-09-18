"""
Define SystemDescriptors and different kind of operators
"""

# from typing import List, Optional
from numbers import Number

import numpy as np
from numpy.linalg import eigh, svd
from numpy.random import random

from alpsqutip.operators import (LocalOperator, OneBodyOperator, Operator,
                                 ProductOperator, SumOperator,
                                 SystemDescriptor)
from alpsqutip.states import GibbsProductDensityOperator


class QuadraticFormOperator(Operator):
    """
    Represents a two-body operator of the form
    sum_alpha w_alpha * Q_alpha^2
    with Q_alpha a local operator or a One body operator.
    """

    system: SystemDescriptor
    terms: list
    weights: list

    def __init__(
        self, terms, weights, system=None, offset=None, check_and_simplify=True
    ):
        # If the system is not given, infer it from the terms
        if system is None:
            for term in terms:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

        # If check_and_simplify, ensure that all the terms are one-body operators
        # and try to use the simplified forms of the operators.

        assert all(isinstance(term, (OneBodyOperator, LocalOperator, Number))
                   for term in terms)
        if check_and_simplify:
            tested_terms = []
            tested_weights = []
            for term, weight in zip(terms, weights):
                if isinstance(term, QuadraticFormOperator):
                    tmp_term = term
                else:
                    tmp_term = self.build_from_operator(term, system, True)

                tested_terms.extend(tmp_term.terms)
                tested_weights.extend([w * weight for w in tmp_term.weights])

            terms, weights = tested_terms, tested_weights

        self.weights = weights
        self.terms = terms
        self.system = system
        self.offset = offset

    @staticmethod
    def build_from_operator(operator, system=None, simplify=True):
        """
        Try to build a quadratic form from an operator
        """
        if isinstance(operator, (int, float, complex)):
            if operator != 0:
                return QuadraticFormOperator([1], [operator], system, False)
            # Empty operator
            return QuadraticFormOperator([], [], system, False)

        if simplify:
            operator.simplify()

        if system is None:
            system = operator.system

        if isinstance(operator, QuadraticFormOperator):
            return operator

        if isinstance(operator, ProductOperator):
            sites_op = operator.sites_op
            num_factors = len(sites_op)
            if num_factors == 0:
                return QuadraticFormOperator.build_from_operator(
                    operator.prefactor, system, False
                )
            if num_factors == 2:
                factors = tuple(
                    LocalOperator(site, op_factor, system)
                    for site, op_factor in operator.sites_op.items()
                )
                prefactor = operator.prefactor
                terms = [factors[0] + factors[1], factors[0] - factors[1]]
                weights = [0.25 * prefactor, -0.25 * prefactor]
                return QuadraticFormOperator(terms, weights, system, None, False)
            if num_factors == 1:
                site, local_op = next(iter(sites_op))
                operator = LocalOperator(
                    site, operator.prefactor * local_op, system)
                return QuadraticFormOperator.build_from_operator(
                    operator, system, False
                )
            raise ValueError(
                "Input operator is not a quadratic form of local operators"
            )

        if isinstance(operator, (LocalOperator, OneBodyOperator)):
            return QuadraticFormOperator(
                [operator + 1, operator - 1], [0.25, -0.25], system, None, False
            )

        if isinstance(operator, SumOperator):
            terms = []
            weights = []
            for term in operator.terms:
                term_qf = QuadraticFormOperator.build_from_operator(
                    term, system, simplify
                )
                terms.extend(term_qf.terms)
                weights.extend(term_qf.weights)
            return QuadraticFormOperator(terms, weights, system, None, False)

        raise TypeError(
            "argument is not a quadratic form on local operators", type(operator))

    def __bool__(self):
        return len(self.weights) > 0 and any(self.weights) and any(self.terms)

    def __add__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return QuadraticFormOperator(
                self.terms + [1], self.weights + [operand], self.system, False
            )
        if isinstance(operand, Operator):
            system = self.system or operand.system
            try:
                convert_operand = self.build_from_operator(operand, system)
            except TypeError:
                return SumOperator([self, operand], self.system)

            offset = self.offset
            offset_2 = convert_operand.offset
            if offset is None:
                offset = offset_2
            elif offset_2 is not None:
                offset = offset + offset_2

            return QuadraticFormOperator(
                self.terms + convert_operand.terms,
                self.weights + convert_operand.weights,
                system=system,
                offset=offset,
                check_and_simplify=False,
            )
        raise ValueError("operand is not an operator")

    def __mul__(self, operand):
        if not bool(operand):
            return QuadraticFormOperator([], [], self.system)

        if isinstance(operand, LocalOperator) and isinstance(
            operand.operator, (int, float, complex)
        ):
            operand = operand.operator
        elif isinstance(operand, ProductOperator) and len(operand.site_ops) == 0:
            operand = operand.prefactor

        if isinstance(operand, (int, float, complex)):
            return QuadraticFormOperator(
                self.terms, [
                    w * operand for w in self.weights], self.system, False
            )
        return SumOperator(
            [w * term * term * operand for w,
                term in zip(self.terms, self.weights)],
            self.system,
        )

    def __neg__(self):
        return QuadraticFormOperator(
            self.terms, [-w for w in self.weights], self.system, False
        )

    def __rmul__(self, operand):
        if not bool(operand):
            return QuadraticFormOperator([], [], self.system)

        if isinstance(operand, LocalOperator) and isinstance(
            operand.operator, (int, float, complex)
        ):
            operand = operand.operator
        elif isinstance(operand, ProductOperator) and len(operand.site_ops) == 0:
            operand = operand.prefactor

        if isinstance(operand, (int, float, complex)):
            return QuadraticFormOperator(
                self.terms, [
                    w * operand for w in self.weights], self.system, False
            )

        return SumOperator(
            [w * operand * term * term for w,
                term in zip(self.terms, self.weights)],
            self.system,
        )

    def partial_trace(self, sites):
        return SumOperator(
            [
                w * (op_term * op_term).partial_trace(sites)
                for w, op_term in zip(self.weights, self.terms)
            ]
        )

    def to_qutip(self):
        return sum(
            w * (op_term.to_qutip()) ** 2
            for w, op_term in zip(self.weights, self.terms)
        )


def hs_scalar_product(o_1, o_2):
    """HS scalar product"""
    return (o_1.dag() * o_2).tr()


def matrix_change_to_orthogonal_basis(
    basis: list, scalar_product=hs_scalar_product, threeshold=1.0e-10
):
    """
    Build the coefficient matrix of the base change to an orthogonal base.
    """

    gram = np.array([[scalar_product(o_1, o_2)
                    for o_1 in basis] for o_2 in basis])

    u, s_diag, v_h = svd(gram, hermitian=True, full_matrices=False)
    kappa = len([sv for sv in s_diag if sv > threeshold])
    v_h = v_h[:kappa]
    return v_h.conj()


def simplify_quadratic_form(
    operator: QuadraticFormOperator, hermitic=True, scalar_product=hs_scalar_product
):
    """
    Takes a 2-body operator and returns lists weights, ops
    such that the original operator is sum(w * op.dag()*op for w,op in zip(weights,ops))

    """
    local_ops = operator.terms
    coeffs = operator.weights
    system = operator.system
    offset = operator.offset

    # Orthogonalize the basis
    u_transp = matrix_change_to_orthogonal_basis(
        local_ops, scalar_product=scalar_product
    )
    u_dag = u_transp.conj()
    # reduced_basis = [ sum(c*old_op  for c, old_op in zip(row,local_ops) )
    #                   for row in u_transp]
    # Build the coefficients of the quadratic form
    coeff_matrix = (u_dag * coeffs).dot(u_transp.transpose())
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
    new_basis = [
        OneBodyOperator(
            [c * old_op for c, old_op in zip(row, local_ops)], system)
        for row in rows
    ]

    # Until here, we assumed that
    if not hermitic:
        antihermitic_part = 1j * simplify_quadratic_form(-1j * operator, True)
        weights = weights + \
            [1j * weight for weight in antihermitic_part.weights]
        new_basis = new_basis + antihermitic_part.terms
    return QuadraticFormOperator(
        new_basis, weights, system, offset=offset, check_and_simplify=False
    )


def selfconsistent_meanfield_from_quadratic_form(quadratic_form: QuadraticFormOperator, max_it):
    """Build a self-consistent mean field approximation to the gibbs state associated to 
    the quadratic form.
    """
    quadratic_form = simplify_quadratic_form(quadratic_form)
    system = quadratic_form.system
    weights_and_basis = [(w, b) for w, b in zip(quadratic_form.weights,
                                                quadratic_form.terms) if w < 0]

    phi = [c[0] * (2.*random()-1.) for c in weights_and_basis]
    for it in range(max_it):
        k_exp = OneBodyOperator([phi_i*wb_i[1]
                                 for phi_i, wb_i in zip(phi, weights_and_basis)],
                                system)
        rho = GibbsProductDensityOperator(k_exp, 1., system)
        new_phi = [2 * w_and_b[0] *
                   rho.expect(w_and_b[1]) for w_and_b in weights_and_basis]
        change = sum(abs(old_phi_i-new_phi_i)
                     for old_phi_i, new_phi_i in zip(new_phi, phi))
        if change < 1.e-10:
            break
        phi = new_phi

    return rho
