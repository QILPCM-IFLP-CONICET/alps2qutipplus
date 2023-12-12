"""
Define SystemDescriptors and different kind of operators
"""

from numbers import Number

# from numbers import Number
from time import time
from typing import Union

import numpy as np
from numpy.linalg import eigh, svd
from numpy.random import random

from alpsqutip.model import SystemDescriptor
from alpsqutip.operator_functions import (
    hermitian_and_antihermitian_parts,
    simplify_sum_operator,
)
from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.scalarprod import gram_matrix
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

    def __init__(self, terms, weights, system=None, offset=None):
        # If the system is not given, infer it from the terms
        assert isinstance(terms, tuple)
        assert isinstance(weights, tuple)
        if system is None:
            for term in terms:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

        # If check_and_simplify, ensure that all the terms are
        # one-body operators and try to use the simplified forms
        # of the operators.

        self.weights = weights
        self.terms = terms
        self.system = system
        self.offset = offset

    def __bool__(self):
        return len(self.weights) > 0 and any(self.weights) and any(self.terms)

    def __noadd__(self, operand):
        if not bool(operand):
            return self
        if not bool(self):
            return operand

        if isinstance(operand, (int, float, complex)):
            return QuadraticFormOperator(
                self.terms + [1], self.weights + [operand], self.system, False
            )
        if isinstance(operand, QuadraticFormOperator):
            system = self.system or operand.system
            offset = operand.offset
            if operand.offset:
                if offset:
                    offset = offset + operand.offset
                else:
                    offset = operand.offset

            return QuadraticFormOperator(
                self.terms + operand.terms,
                self.weights + operand.weights,
                system,
                offset,
            )

        if isinstance(operand, Operator):
            system = self.system or operand.system
            try:
                convert_operand = build_quadratic_form_from_operator(
                    operand, system, False, False
                )
            except ValueError:
                return SumOperator(
                    (
                        self,
                        operand,
                    ),
                    self.system,
                )

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
            )
        raise ValueError("operand is not an operator")

    def __nomul__(self, operand):
        if not bool(operand):
            return QuadraticFormOperator([], [], self.system)

        if isinstance(operand, LocalOperator) and isinstance(
            operand.operator, (int, float, complex)
        ):
            operand = operand.operator
        elif (
            isinstance(operand, ProductOperator) and len(operand.sites_op) == 0
        ):
            operand = operand.prefactor

        if isinstance(operand, (int, float, complex)):
            return QuadraticFormOperator(
                self.terms,
                [w * operand for w in self.weights],
                self.system,
                None,
            )
        return SumOperator(
            tuple(
                w * term * term * operand
                for w, term in zip(self.terms, self.weights)
            ),
            self.system,
        )

    def __neg__(self):
        return QuadraticFormOperator(
            self.terms, tuple(-w for w in self.weights), self.system, None
        )

    def __normul__(self, operand):
        if not bool(operand):
            return QuadraticFormOperator([], [], self.system)

        if isinstance(operand, LocalOperator) and isinstance(
            operand.operator, (int, float, complex)
        ):
            operand = operand.operator
        elif (
            isinstance(operand, ProductOperator) and len(operand.site_ops) == 0
        ):
            operand = operand.prefactor

        if isinstance(operand, (int, float, complex)):
            return QuadraticFormOperator(
                self.terms,
                [w * operand for w in self.weights],
                self.system,
                None,
            )

        return SumOperator(
            tuple(
                weight * term * term
                for weight, term in zip(self.weights, self.terms)
            ),
            self.system,
        )

    def act_over(self):
        result = set()
        for term in self.terms:
            term_act_over = term.act_over()
            if term_act_over is None:
                return None
            result = result.union(term_act_over)
        return result

    @property
    def isherm(self):
        return all(isinstance(weight, (int, float)) for weight in self.weights)

    def partial_trace(self, sites):
        return SumOperator(
            tuple(
                w * (op_term * op_term).partial_trace(sites)
                for w, op_term in zip(self.weights, self.terms)
            )
        )

    def to_qutip(self):
        return sum(
            (w * op_term.dag() * op_term).to_qutip()
            for w, op_term in zip(self.weights, self.terms)
        )

    def to_sum_operator(self, symplify: bool = True) -> SumOperator:
        """Convert to a linear combination of quadratic operators"""
        result = sum(
            (w * op_term.dag() * op_term)
            for w, op_term in zip(self.weights, self.terms)
        )
        if symplify:
            return result.simplify()
        return result


def hs_scalar_product(o_1, o_2):
    """HS scalar product"""
    return (o_1.dag() * o_2).tr()


def matrix_change_to_orthogonal_basis(
    basis: list, scalar_product=hs_scalar_product, threeshold=1.0e-10
):
    """
    Build the coefficient matrix of the base change to an orthogonal base.
    """
    gram = gram_matrix(basis, scalar_product)
    # pylint: disable=unused-variable
    # TODO: Check if there is a way to obtain just the left singular vectors.
    left_sv, s_diag, v_h = svd(gram, hermitian=True, full_matrices=False)
    kappa = len([sv for sv in s_diag if sv > threeshold])
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
        print("   new phi:", new_phi)
        if isinstance(logdict, dict):
            evolution.append(new_phi)
            timestamps.append(time())

        change = sum(
            abs(old_phi_i - new_phi_i)
            for old_phi_i, new_phi_i in zip(new_phi, phi)
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
            [],
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
                    LocalOperator(
                        factor2[0], factor2[1] * (-prefactor), system
                    ),
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
            q_term = build_quadratic_form_from_operator(
                term, system, False, True
            )
            if q_term is None:
                return None
            terms.extend(q_term.terms)
            weights.extend(q_term.weights)
            if q_term.offset:
                offset.extend(q_term.offset)
        return QuadraticFormOperator(
            tuple(terms), tuple(weights), system, offset
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
            return build_quadratic_form_from_operator(
                real_part, system, simplify, True
            )

        if not bool(real_part):
            return (
                build_quadratic_form_from_operator(
                    imag_part, system, simplify, True
                )
                * 1j
            )

        result_re = build_quadratic_form_from_operator(
            real_part, system, simplify, True
        )

        result_im = build_quadratic_form_from_operator(
            imag_part, system, simplify, True
        )

        terms = result_re.terms + result_im.terms
        weights = result_re.weights + tuple(
            weight * 1j for weight in result_im.weights
        )
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
    return lookup_table_methods.get(
        type(operator), subclass_to_quadratic_form
    )(operator)


# #####################
#
#  Arithmetic
#
# #######################

# Sum Quadratic with Quadratic


@Operator.register_add_handler(
    (
        QuadraticFormOperator,
        QuadraticFormOperator,
    )
)
def _(x_op: QuadraticFormOperator, y_op: QuadraticFormOperator):
    system = x_op.system or y_op.system
    return QuadraticFormOperator(
        x_op.terms + y_op.terms, x_op.weights + y_op.weights, system, None
    )


@Operator.register_mul_handler(
    (
        QuadraticFormOperator,
        QuadraticFormOperator,
    )
)
def _(x_op: QuadraticFormOperator, y_op: QuadraticFormOperator):
    return x_op.to_sum_operator() * y_op.to_sum_operator()


# Number and QuadraticFormOperator


@Operator.register_add_handler(
    (
        QuadraticFormOperator,
        Number,
    )
)
def _(qf_op: QuadraticFormOperator, y_val: ScalarOperator):
    system = qf_op.system
    weights, terms = qf_op.weights, qf_op.terms

    return QuadraticFormOperator(
        weights=weights + (y_val,),
        terms=terms + (ScalarOperator(1, system),),
        system=system,
    )


@Operator.register_mul_handler(
    (
        QuadraticFormOperator,
        Number,
    )
)
def _(x_op: QuadraticFormOperator, y_val: Number):
    system = x_op.system
    return QuadraticFormOperator(
        x_op.terms,
        tuple(weight * y_val for weight in x_op.weights),
        system,
        None,
    )


@Operator.register_mul_handler(
    (
        Number,
        QuadraticFormOperator,
    )
)
def _(y_val: Number, x_op: QuadraticFormOperator):
    system = x_op.system
    return QuadraticFormOperator(
        x_op.terms,
        tuple(weight * y_val for weight in x_op.weights),
        system,
        None,
    )


# QuadraticOperator form and ScalarOperator


@Operator.register_add_handler(
    (
        QuadraticFormOperator,
        ScalarOperator,
    )
)
def _(qf_op: QuadraticFormOperator, sf_op: ScalarOperator):
    system = qf_op.system or sf_op.system
    weights, terms = qf_op.weights, qf_op.terms

    return QuadraticFormOperator(
        weights=weights + (sf_op.prefactor,),
        terms=terms + (ScalarOperator(1, system),),
        system=system,
    )


@Operator.register_mul_handler(
    (
        QuadraticFormOperator,
        ScalarOperator,
    )
)
def _(x_op: QuadraticFormOperator, y_op: ScalarOperator):
    system = x_op.system or y_op.system
    y_val = y_op.prefactor
    return QuadraticFormOperator(
        x_op.terms,
        tuple(weight * y_val for weight in x_op.weights),
        system,
        None,
    )


@Operator.register_mul_handler(
    (
        ScalarOperator,
        QuadraticFormOperator,
    )
)
def _(y_op: ScalarOperator, x_op: QuadraticFormOperator):
    system = x_op.system or y_op.system
    y_val = y_op.prefactor
    return QuadraticFormOperator(
        x_op.terms,
        tuple(weight * y_val for weight in x_op.weights),
        system,
        None,
    )


# Quadratic form and Local / Product operators


@Operator.register_add_handler(
    (
        QuadraticFormOperator,
        LocalOperator,
    )
)
@Operator.register_add_handler(
    (
        QuadraticFormOperator,
        ProductOperator,
    )
)
def _(
    qf_op: QuadraticFormOperator, y_op: Union[LocalOperator, ProductOperator]
):
    system = qf_op.system or y_op.system
    try:
        term = build_quadratic_form_from_operator(y_op, system, True, None)
    except ValueError:
        term = None

    if term:
        return qf_op + term

    return SumOperator(
        (
            qf_op,
            y_op,
        ),
        system,
    )


@Operator.register_mul_handler(
    (
        QuadraticFormOperator,
        LocalOperator,
    )
)
@Operator.register_mul_handler(
    (
        QuadraticFormOperator,
        ProductOperator,
    )
)
def _(
    qf_op: QuadraticFormOperator, y_op: Union[LocalOperator, ProductOperator]
):
    return qf_op.to_sum_operator() * y_op


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
def _(
    y_op: Union[LocalOperator, ProductOperator], qf_op: QuadraticFormOperator
):
    return y_op * qf_op.to_sum_operator()


# QuadraticForm and QutipOperator


@Operator.register_add_handler(
    (
        QuadraticFormOperator,
        QutipOperator,
    )
)
def _(qf_op: QuadraticFormOperator, y_op: QutipOperator):
    return qf_op.to_qutip() + y_op


@Operator.register_mul_handler(
    (
        QuadraticFormOperator,
        QutipOperator,
    )
)
def _(qf_op: QuadraticFormOperator, y_op: QutipOperator):
    return qf_op.to_qutip_operator() * y_op


@Operator.register_mul_handler(
    (
        QutipOperator,
        QuadraticFormOperator,
    )
)
def _(y_op: QutipOperator, qf_op: QuadraticFormOperator):
    return y_op * qf_op.to_qutip_operator()
