r"""
QuadraticForm Operators

Quadratic Form Operators provides a representation for quantum operators
of the form

Q= L + \sum_a w_a M_a ^2 + \delta Q

with L and M_a one-body operators, w_a certain weights and
\delta Q a *remainder* as a sum of n-body terms.



"""

from numbers import Number

# from numbers import Number
from time import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.random import random

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ScalarOperator,
)
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

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
    offset: Optional[Operator]

    def __init__(self, basis, weights, system=None, linear_term=None, offset=None):
        # If the system is not given, infer it from the terms
        if offset:
            offset = offset.simplify()
        if linear_term:
            linear_term = linear_term.simplify()
            assert (
                isinstance(linear_term, OneBodyOperator)
                or len(linear_term.acts_over()) < 2
            )
        self._isherm = None
        assert isinstance(basis, tuple)
        assert isinstance(weights, tuple)
        assert all(gen.isherm for gen in basis)  # TODO: REMOVE ME
        assert (
            isinstance(linear_term, (OneBodyOperator, LocalOperator, ScalarOperator))
            or linear_term is None
        ), f"{type(offset)} should be a LocalOperator or a OneBodyOperator"
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
        self.linear_term = linear_term
        self._simplified = False

    def __bool__(self):
        for term in (self.linear_term, self.offset):
            if term is not None:
                if not term.is_zero:
                    return True
        return len(self.weights) > 0 and any(self.weights) and any(self.basis)

    def __add__(self, other):

        # TODO: remove me and fix the sums
        if not bool(other):
            return self
        if isinstance(other, Number):
            other = ScalarOperator(other, system=self.system)

        assert isinstance(other, Operator), "other must be an operator."
        system = self.system or other.system
        if isinstance(other, QuadraticFormOperator):
            basis = self.basis + other.basis
            weights = self.weights + other.weights
            offset = self.offset
            linear_term = self.linear_term
            if offset is None:
                offset = other.offset
            else:
                if other.offset is not None:
                    offset = offset + other.offset

            if linear_term is None:
                offset = other.linear_term
            else:
                if other.linear_term is not None:
                    linear_term = linear_term + other.linear_term
            return QuadraticFormOperator(basis, weights, system, linear_term, offset)
        if isinstance(
            other,
            (
                ScalarOperator,
                LocalOperator,
                OneBodyOperator,
            ),
        ):
            linear_term = self.linear_term
            linear_term = (
                other if linear_term is None else (linear_term + other).simplify()
            )
            basis = self.basis
            weights = self.weights
            return QuadraticFormOperator(
                basis, weights, system, linear_term, offset=None
            )
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
            linear_term = self.linear_term
            if linear_term is not None:
                linear_term = (linear_term * other).simplify()

            return QuadraticFormOperator(
                self.basis,
                tuple(w * other for w in self.weights),
                system,
                linear_term=linear_term,
                offset=offset,
            )
        standard_repr = self.as_sum_of_products(False).simplify()
        return standard_repr * other

    def __neg__(self):
        offset = self.offset
        if offset is not None:
            offset = -offset
        linear_term = self.linear_term
        if linear_term is not None:
            linear_term = -linear_term
        return QuadraticFormOperator(
            self.basis,
            tuple(-w for w in self.weights),
            self.system,
            linear_term,
            offset,
        )

    def _set_system_(self, system=None):
        self.system = system
        for basis_elem in self.basis:
            basis_elem._set_system_(system)

        offset = self.offset
        linear_term = self.linear_term
        if offset is not None:
            offset._set_system_(system)
        if linear_term is not None:
            linear_term._set_system_(system)
        return self

    def acts_over(self):
        """
        Set of sites over the state acts.
        """
        result = frozenset()
        for term in self.basis:
            try:
                result = result.union(term.acts_over())
            except TypeError:
                return None

        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            try:
                result = result.union(term.acts_over())
            except TypeError:
                return None
        return result

    def as_sum_of_products(self, simplify: bool = True) -> SumOperator:
        """Convert to a linear combination of two-body operators"""
        isherm = self._isherm
        isdiag = self.isdiagonal
        if all(b_op.isherm for b_op in self.basis):
            terms = tuple(
                (
                    ((op_term.dag() * op_term) * w)
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

        for term in (self.offset, self.linear_term):
            if term is not None:
                terms = terms + (term,)
        if len(terms) == 0:
            return ScalarOperator(0, self.system)
        if len(terms) == 1:
            return terms[0]
        result = SumOperator(terms, self.system, isherm, isdiag)
        if simplify:
            return result.simplify()
        return result

    def dag(self):
        linear_term = self.linear_term
        linear_term = None if linear_term is None else linear_term.dag()
        offset = self.offset
        offset = None if offset is None else offset.dag()
        result = QuadraticFormOperator(
            self.basis,
            tuple((np.conj(w) for w in self.weights)),
            self.system,
            linear_term,
            offset,
        )
        result._simplified = self._simplified
        return result

    def flat(self):
        return self.as_sum_of_products().flat()

    @property
    def isdiagonal(self):
        """True if the operator is diagonal in the product basis."""
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            isdiagonal = term.isdiagonal
            if not isdiagonal:
                return isdiagonal

        if all(term.isdiagonal for term in self.basis):
            return True
        return False

    @property
    def isherm(self):
        isherm = self._isherm
        if isherm is not None:
            return isherm

        # We start assumig that the operator is hermitician
        isherm = True
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            isherm = (isherm and term.isherm) or False

        # Now, let's check the weights
        weights = self.weights
        if len(weights) == 0:
            self._isherm = isherm
            return isherm
        if isherm:
            isherm = all(
                abs(np.imag(weight)) < ALPSQUTIP_TOLERANCE for weight in weights
            )
            if isherm is not None:
                if isherm or len(weights) == 1:
                    self._isherm = isherm
                    return isherm
        # A more drastic approach: convert it to a sum of products
        isherm = self.as_sum_of_products().simplify().isherm or False
        self._isherm = isherm
        return isherm

    def partial_trace(self, sites: Union[tuple, SystemDescriptor]):

        if not isinstance(sites, SystemDescriptor):
            sites = self.system.subsystem(sites)

        result = None
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            if result:
                tpt = term.partial_trace(sites)
                assert isinstance(tpt, Operator)
                result = result + tpt
            else:
                result = term.partial_trace(sites)
                assert isinstance(
                    result, Operator
                ), f"partial trace of {type(term)} returns {type(result)}"

        if len(self.basis) == 0 and result is None:
            return ScalarOperator(0, sites)

        # TODO: Implement me to return a quadratic operator
        #
        #  (Sum_a  w_a(sum_i L_ai)^2).ptrace = Sum_a w_a ((sum_i L_ai)^2).ptrace
        #  (Sum_i L_ai)^2 = Sum_i (La_i L_aj).ptrace= (La_i1)^2*Tr[1_2] + I Tr(La_i2)^2+...
        #
        terms = tuple(
            w * (op_term * op_term).partial_trace(sites)
            for w, op_term in zip(self.weights, self.basis)
        )
        if result is not None:
            terms = terms + (result,)
        terms = tuple(terms)
        return SumOperator(
            terms,
            sites,
        ).simplify()

    def simplify(self):
        """
        Simplify the operator.
        Build a new representation with a smaller basis.
        """
        if self._simplified:
            return self

        result = simplify_quadratic_form(self, hermitic=False)
        result._simplified = True
        return result

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        """
        return a qutip object acting over the sites listed in
        `block`.
        By default (`block=None`) returns a qutip object
        acting over all the sites, in lexicographical order.
        """
        sites = self.system.sites
        if block is None:
            block = tuple(sorted(sites))
        else:
            block = block + tuple(
                (site for site in self.acts_over() if site not in block)
            )

        result = sum(
            (op_term.dag() * op_term * w).to_qutip(block)
            for w, op_term in zip(self.weights, self.basis)
        )
        for term in (self.offset, self.linear_term):
            if term is not None:
                result += term.to_qutip(block)
        return result


def quadratic_form_expect(sq_op, state):
    """
    Compute the expectation value of op, taking advantage
    of its structure.
    """
    sq_op = sq_op.as_sum_of_products(False)
    return state.expect(sq_op)


def selfconsistent_meanfield_from_quadratic_form(
    quadratic_form: QuadraticFormOperator, max_it, logdict=None
):
    """
    Build a self-consistent mean field approximation
    to the gibbs state associated to the quadratic form.
    """
    from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator

    #    quadratic_form = simplify_quadratic_form(quadratic_form)
    system = quadratic_form.system
    terms = quadratic_form.terms
    weights = quadratic_form.weights

    operators = [2 * w * b for w, b in zip(weights, terms)]
    basis = [b for w, b in zip(weights, terms)]

    phi = [2.0 * random() - 1.0]

    evolution: list = []
    timestamps: list = []

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


def one_body_operator_hermitician_hs_sp(x_op: OneBodyOperator, y_op: OneBodyOperator):
    """
    Hilbert Schmidt scalar product optimized for OneBodyOperators
    """
    result = 0
    terms_x: Tuple[LocalOperator] = (
        x_op.terms if isinstance(x_op, OneBodyOperator) else (x_op,)
    )
    terms_y: Tuple[LocalOperator] = (
        y_op.terms if isinstance(y_op, OneBodyOperator) else (y_op,)
    )

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
    sum(w * op**2 for w,op in zip(weights,ops))
    """
    from .build import build_quadratic_form_from_operator

    changed = False
    system = operator.system
    if not operator.isherm and hermitic:
        changed = True

    def simplify_other_terms(term):
        nonlocal changed
        if term is None:
            return term
        new_term = term
        if hermitic and not term.isherm:
            new_term = (new_term + new_term.dag()) * 0.5
        new_term = new_term.simplify()
        if term is not new_term:
            changed = True
        return new_term

    # First, rebuild the quadratic form.
    qf_op = QuadraticFormOperator(operator.basis, operator.weights, system)
    new_qf_op = build_quadratic_form_from_operator(
        qf_op.as_sum_of_products(), True, hermitic
    )
    # If the new basis is larger and the hermitician character haven´t changed, keep the older.
    if changed or len(new_qf_op.basis) < len(qf_op.basis):
        qf_op = new_qf_op
        changed = True

    # Now, work on the offset and the linear term

    linear_term = simplify_other_terms(operator.linear_term)
    offset = simplify_other_terms(operator.offset)

    if not changed:
        return operator

    if qf_op.linear_term:
        linear_term = (
            (linear_term + qf_op.linear_term).simplify()
            if linear_term
            else qf_op.linear_term
        )

    if qf_op.offset:
        offset = (
            (offset + qf_op.offset).simplify() if offset is not None else qf_op.offset
        )

    return QuadraticFormOperator(
        qf_op.basis, qf_op.weights, system, linear_term, offset
    )
