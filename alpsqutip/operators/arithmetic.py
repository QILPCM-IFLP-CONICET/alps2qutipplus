# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Classes and functions for operator arithmetic.
"""

import logging
from typing import List, Optional, Union

from numbers import Number

import numpy as np
import qutip


from alpsqutip.model import Operator, SystemDescriptor
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.basic import (
    ScalarOperator,
    LocalOperator,
    ProductOperator,
)

COUNTER_ISHERMITICIAN = 0


class SumOperator(Operator):
    """
    Represents a linear combination of operators
    """

    terms: List[Operator]
    system: Optional[SystemDescriptor]

    def __init__(
        self, term_list: tuple, system=None, isherm: Optional[bool] = None
    ):
        assert isinstance(term_list, tuple)
        self.terms = tuple(term_list)
        assert system is not None
        if system is None and term_list:
            for term in term_list:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

        self.system = system
        self._isherm = isherm

    def __add__(self, operand):
        system = self.system
        if not bool(operand):
            return self

        if isinstance(operand, (int, float, complex)):
            operand = ScalarOperator(operand, system)

        isherm = self._isherm and operand.isherm

        if isinstance(operand, ProductOperator):
            if operand.prefactor == 0:
                return self
            new_terms = self.terms + (operand,)
        elif isinstance(operand, LocalOperator):
            new_terms = self.terms + (operand,)
        elif isinstance(operand, SumOperator):
            if len(operand.terms) == len(self.terms) == 1:
                return self.terms[0] + operand.terms[0]
            new_terms = self.terms + operand.terms
        elif isinstance(operand, QutipOperator):
            return self.to_qutip_operator() + operand
        elif isinstance(operand, Operator):
            new_terms = self.terms + (operand,)
        else:
            raise ValueError(type(self), type(operand))

        new_terms = tuple(t for t in new_terms if t)
        return SumOperator(new_terms, system, isherm)

    def __bool__(self):
        if len(self.terms) == 0:
            return False

        if any(bool(t) for t in self.terms):
            return True
        return False

    def __pow__(self, exp):
        isherm = self._isherm
        if isinstance(exp, int):
            if exp == 0:
                return 1
            if exp == 1:
                return self
            if exp > 1:
                result = self
                exp -= 1
                while exp:
                    exp -= 1
                    result = result * self
                if isherm:
                    result = SumOperator(result.terms, self.system, True)
                return result

            raise TypeError("SumOperator does not support negative powers")
        raise TypeError(
            (
                f"unsupported operand type(s) for ** or pow(): "
                f"'SumOperator' and '{type(exp).__name__}'"
            )
        )

    def __no_mul__(self, operand):
        system = self.system
        if not bool(operand):
            return ScalarOperator(0.0, system)
        if isinstance(operand, QutipOperator):
            return self.to_qutip_operator() * operand

        isherm = self._isherm

        if isinstance(operand, (int, float)):
            new_terms = [
                operand * operand1 for operand1 in self.terms if operand1
            ]
        elif isinstance(operand, complex):
            isherm = isherm and (operand.imag == 0.0)
            new_terms = [
                operand * operand1 for operand1 in self.terms if operand1
            ]
        elif isinstance(operand, (ProductOperator, LocalOperator)):
            isherm = None
            if operand.prefactor:
                new_terms = [
                    operand1 * operand for operand1 in self.terms if operand1
                ]
            else:
                new_terms = []
        elif isinstance(operand, SumOperator):
            isherm = None
            new_terms = [
                op_1 * op_2 for op_1 in self.terms for op_2 in operand.terms
            ]
        elif isinstance(operand, Operator):
            return self.to_qutip_operator() * operand
        else:
            raise TypeError(type(operand))

        new_terms = [term for term in new_terms if term]
        if len(new_terms) == 0:
            return ScalarOperator(0.0, system)
        if len(new_terms) == 1:
            return new_terms[0]
        return SumOperator(new_terms, system, isherm)

    def __neg__(self):
        return SumOperator(
            tuple(-t for t in self.terms), self.system, self._isherm
        )

    def __repr__(self):
        return "(\n" + "\n  +".join(repr(t) for t in self.terms) + "\n)"

    def __no_rmul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return self * operand
        return NotImplementedError

    def act_over(self):
        result = set()
        for term in self.terms:
            term_act_over = term.act_over()
            if term_act_over is None:
                return None
            result = result.union(term_act_over)
        return result

    def dag(self):
        """return the adjoint operator"""
        if self._isherm:
            return self
        return SumOperator(
            tuple(term.dag() for term in self.terms), self.system
        )

    @property
    def isherm(self) -> bool:
        isherm = self._isherm

        def aggresive_hermitician_test():
            global COUNTER_ISHERMITICIAN
            # pylint: disable=import-outside-toplevel
            from alpsqutip.operator_functions import (
                simplify_sum_operator,
                hermitian_and_antihermitian_parts,
            )

            COUNTER_ISHERMITICIAN += 1
            assert COUNTER_ISHERMITICIAN < 200
            logging.warning("calling aggressive determination of hermiticity")
            self._isherm = False
            real_part, imag_part = hermitian_and_antihermitian_parts(self)
            real_part = simplify_sum_operator(real_part)
            imag_part = simplify_sum_operator(imag_part)
            if not bool(imag_part):
                self._isherm = True
                if isinstance(real_part, SumOperator):
                    self.terms = real_part.terms
                else:
                    self.terms = [real_part]
                return True
            self._isherm = False
            return False

        if isherm is None:
            # First, try with the less aggressive test:
            if all(term.isherm for term in self.terms):
                self._isherm = True
                return True
            return aggresive_hermitician_test()
        return self._isherm

    def partial_trace(self, sites: list):
        return sum(
            term.prefactor * term.partial_trace(sites) for term in self.terms
        )

    def simplify(self):
        system = self.system
        general_terms = []
        isherm = self._isherm
        # First, shallow the list of terms:
        for term in (t.simplify() for t in self.terms):
            if isinstance(term, SumOperator):
                general_terms.extend(term.terms)
            else:
                general_terms.append(term)

        terms = general_terms
        # Now, collect and sum LocalOperator and QutipOperator terms
        general_terms = []
        site_terms = {}
        qutip_terms = []
        for term in terms:
            if isinstance(term, LocalOperator):
                site_terms.setdefault(term.site, []).append(term.operator)
                continue
            if isinstance(term, QutipOperator):
                qutip_terms.append(term)
            else:
                general_terms.append(term)

        loc_ops_lst = [
            LocalOperator(site, sum(l_ops), system)
            for site, l_ops in site_terms.items()
        ]

        qutip_term = sum(qutip_terms)
        qutip_terms = qutip_term if qutip_terms else []
        terms = general_terms + loc_ops_lst + qutip_terms
        return SumOperator(tuple(terms), system, isherm)

    def to_qutip(self):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip()
        return sum(t.to_qutip() for t in self.terms)

    def tr(self):
        return sum(t.tr() for t in self.terms)


NBodyOperator = SumOperator


class OneBodyOperator(SumOperator):
    """A linear combination of local operators"""

    def __init__(self, terms, system=None, check_and_convert=True):
        """
        if check_and_convert is True,
        """
        assert isinstance(terms, tuple)
        check_and_convert = True

        def collect_systems(terms, system):
            for term in terms:
                term_system = term.system
                if term_system is None:
                    continue
                if system is None:
                    system = term.system
                else:
                    system = system.union(term_system)
            return system

        if check_and_convert:
            system = collect_systems(terms, system)
            terms, system = self._simplify_terms(terms, system)

        isherm = None  # all(term.isherm for term in terms)
        super().__init__(terms, system, isherm)

    def __add__(self, operand):
        system = self.system or operand.system
        if isinstance(operand, OneBodyOperator):
            my_terms = tuple(term for term in self.terms if term)
            other_terms = tuple(term for term in operand.terms if term)
            return OneBodyOperator(my_terms + other_terms, system)
        if isinstance(operand, Number):
            if operand:
                return OneBodyOperator(
                    self.terms + (ScalarOperator(operand, system),), system
                )
            return self
        if isinstance(operand, LocalOperator):
            return OneBodyOperator(self.terms + tuple((operand,)), system)
        return super().__add__(operand)

    def __repr__(self):
        return "  " + "\n  +".join(
            "(" + repr(term) + ")" for term in self.terms
        )

    def __rmul__(self, operand):
        system = self.system or operand.system
        if isinstance(operand, OneBodyOperator):
            my_terms = self.terms
            other_terms = operand.terms
            return SumOperator(
                [
                    other_term * my_term
                    for my_term in my_terms
                    for other_term in other_terms
                ],
                system,
            )
        if isinstance(operand, (int, float, complex)):
            if operand:
                return OneBodyOperator(
                    tuple(operand * term for term in self.terms if term),
                    system,
                )
            return ProductOperator({}, 0.0, system)
        return super().__mul__(operand)

    def __neg__(self):
        return OneBodyOperator(
            tuple(-term for term in self.terms), self.system
        )

    def dag(self):
        return OneBodyOperator(
            tuple(term.dag() for term in self.terms),
            system=self.system,
            check_and_convert=False,
        )

    def expm(self):
        sites_op = {}
        for term in self.terms:
            if not bool(term):
                continue
            operator = term.operator
            if hasattr(operator, "expm"):
                sites_op[term.site] = operator.expm()
            else:
                sites_op[term.site] = np.exp(operator)
        return ProductOperator(sites_op, system=self.system)

    @staticmethod
    def _simplify_terms(terms, system):
        """Group terms by subsystem and process scalar terms"""
        terms_by_subsystem = {}
        scalar_term = 0

        def process_term(term):
            """Process each term recursively"""
            nonlocal scalar_term
            nonlocal terms_by_subsystem

            if isinstance(term, Number):
                print("add ", type(term), "and ", scalar_term)
                scalar_term = scalar_term + term
                return
            term = term.simplify()
            if isinstance(term, SumOperator):
                for subterm in term.terms:
                    process_term(subterm)
                return

            subsystem = term.act_over()
            if subsystem is None:
                raise ValueError(
                    f"   {term} acting over the whole system "
                    "is not a one body term."
                )
            if len(subsystem) == 0:
                scalar_term = term + scalar_term
                return
            if len(subsystem) != 1:
                raise ValueError(
                    f"   {term} acting over {subsystem} "
                    "is not a one body term."
                )
            terms_by_subsystem.setdefault(tuple(subsystem), []).append(term)

        for term in terms:
            process_term(term)

        if scalar_term:
            # If the scalar term is not trivial,
            # add it to the first term of the first subsystem.
            # If the list is empty, just store it as the only term in
            # the sum.

            if not terms_by_subsystem:
                if isinstance(scalar_term, Number):
                    scalar_term = ScalarOperator(scalar_term, system)
                return (scalar_term,), system

            first_key, terms_list = next(iter(terms_by_subsystem.items()))
            print(first_key, "has a first term of type ", terms_list[0], "\n")
            first_term_plus_scalar = terms_list[0] + scalar_term
            print(
                " by adding the scalar we get",
                type(first_term_plus_scalar),
                "\n",
            )
            terms_list[0] = first_term_plus_scalar

        terms = tuple(
            LocalOperator(
                key[0],
                sum(term.operator for term in terms_subsystem),
                system,
            )
            for key, terms_subsystem in terms_by_subsystem.items()
        )

        return terms, system

    def simplify(self):
        terms, system = self._simplify_terms(self.terms, self.system)
        self.terms = terms
        self.system = system
        if len(terms) == 0:
            return ScalarOperator(0, system)
        if len(terms) == 1:
            return terms[0]
        return self


# Rules to compute products of operators and numbers

__mul__dispatch__ = Operator.__mul__dispatch__

# Number times operators
__mul__dispatch__[
    (
        ScalarOperator,
        ScalarOperator,
    )
] = lambda x, y: ScalarOperator(
    x.prefactor * y.prefactor, x.system or y.system
)


def mul_commute_ops(mult_func):
    """Produce a function with the parameters reversed"""

    def commuted_func(first, second):
        """Function called with the parameters in reversed order"""
        return mult_func(second, first)

    return commuted_func


def mul_local_times_local_operator(x: LocalOperator, y: LocalOperator):
    """Multiply two local operators"""
    site_x = x.site
    site_y = y.site
    system = x.system or y.system
    if site_x == site_y:
        return LocalOperator(site_x, x.operator * y.operator, x.system)
    return ProductOperator(
        {site_x: x.operator, site_y: y.operator}, 1.0, system
    )


__mul__dispatch__[
    (LocalOperator, LocalOperator)
] = mul_local_times_local_operator


def mul_local_times_product_operator(x: LocalOperator, y: ProductOperator):
    """Multiply two local operators"""
    sites_op = y.sites_op.copy()
    site_x = x.site
    prefactor = y.prefactor
    system = x.system or y.system
    op_right = sites_op.get(site_x, None)
    if op_right is None:
        sites_op[site_x] = x.operator * prefactor
    else:
        sites_op[site_x] = x.operator * op_right * prefactor

    if len(sites_op) == 1:
        return LocalOperator(site_x, sites_op[site_x], system)
    return ProductOperator(sites_op, 1, system)


__mul__dispatch__[
    (LocalOperator, ProductOperator)
] = mul_local_times_product_operator


def mul_product_times_local_operator(y: ProductOperator, x: LocalOperator):
    """Multiply two local operators"""
    sites_op = y.sites_op.copy()
    site_x = x.site
    prefactor = y.prefactor
    system = x.system or y.system
    op_left = sites_op.get(site_x, None)
    if op_left is None:
        sites_op[site_x] = x.operator * prefactor
    else:
        sites_op[site_x] = op_left * x.operator * prefactor

    if len(sites_op) == 1:
        return LocalOperator(site_x, sites_op[site_x], system)
    return ProductOperator(sites_op, 1, system)


__mul__dispatch__[
    (ProductOperator, LocalOperator)
] = mul_product_times_local_operator


def mul_product_times_product_operator(x: ProductOperator, y: ProductOperator):
    """Multiply two product operators"""
    sites_op = x.sites_op.copy()
    prefactor = x.prefactor * y.prefactor
    system = x.system or y.system
    for site_right, right_op in y.sites_op.items():
        left_op = sites_op.get(site_right, None)
        if left_op is None:
            sites_op[site_right] = right_op
        else:
            sites_op[site_right] = left_op * right_op

    if len(sites_op) == 1:
        site, op_local = next(iter(sites_op.items()))
        return LocalOperator(site, op_local * prefactor, system)
    return ProductOperator(sites_op, prefactor, system)


__mul__dispatch__[
    (ProductOperator, ProductOperator)
] = mul_product_times_product_operator


def mul_number_times_scalar(x: Number, y: ScalarOperator):
    """Product of a number and a scalar operator"""
    return ScalarOperator(x * y.prefactor, y.system)


def mul_number_times_local_operator(x: Number, y: LocalOperator):
    """Product of a number and a Local operator"""
    return LocalOperator(y.site, x * y.operator, y.system)


def mul_number_times_product_operator(x: Number, y: ProductOperator):
    """Product of a number and a product operator"""
    return ProductOperator(y.sites_op, x * y.prefactor, y.system)


def mul_number_times_qutip_operator(x: Number, y: QutipOperator):
    """Product of a number and a qutip operator"""
    return QutipOperator(y.operator, y.system, y.prefactor * x)


# Qutip times operator


def mul_qutip_and_operator(x: qutip.Qobj, y: Operator):
    """Multiply a Qobj by an operator"""
    return QutipOperator(x, system=y.system, prefactor=1) * y


def mul_operator_and_qutip(x: Operator, y: qutip.Qobj):
    """Multiply a Qobj by an operator"""
    return x * QutipOperator(y, system=x.system, prefactor=1)


# Sum operator


def mul_any_times_sum_operator(x: Union[Operator, Number], y: SumOperator):
    """Product of a number and a product operator"""
    return SumOperator(tuple(x * term for term in y.terms), y.system)


def mul_sum_times_any_operator(y: SumOperator, x: Union[Operator, Number]):
    """Product of a number and a product operator"""
    return SumOperator(tuple(term * x for term in y.terms), y.system)


def mul_sum_times_sum(factor_left: SumOperator, factor_right: SumOperator):
    """Produc of two sums of operators"""
    system = factor_left.system or factor_right.system
    return SumOperator(
        tuple(
            term_x * term_y
            for term_x in factor_left.terms
            for term_y in factor_right.terms
        ),
        system,
    )


def mul_scalar_times_onebody_operator(
    x: Union[ScalarOperator, Number], y: OneBodyOperator
):
    """Product of a number and a product operator"""
    return OneBodyOperator(tuple(term * x for term in y.terms), y.system)


_mul_scalar_to_operator_type = {
    ScalarOperator: mul_number_times_scalar,
    LocalOperator: mul_number_times_local_operator,
    ProductOperator: mul_number_times_product_operator,
    QutipOperator: mul_number_times_qutip_operator,
    SumOperator: mul_any_times_sum_operator,
    OneBodyOperator: mul_scalar_times_onebody_operator,
}


for type_op, func in _mul_scalar_to_operator_type.items():
    __mul__dispatch__[
        (
            qutip.Qobj,
            type_op,
        )
    ] = mul_qutip_and_operator
    __mul__dispatch__[
        (
            type_op,
            qutip.Qobj,
        )
    ] = mul_operator_and_qutip
    for type_number in (int, float, complex):
        __mul__dispatch__[
            (
                type_number,
                type_op,
            )
        ] = func
        __mul__dispatch__[
            (
                type_op,
                type_number,
            )
        ] = mul_commute_ops(func)

__mul__dispatch__[
    (
        SumOperator,
        SumOperator,
    )
] = mul_sum_times_sum
__mul__dispatch__[
    (
        SumOperator,
        Operator,
    )
] = mul_sum_times_any_operator
__mul__dispatch__[
    (
        Operator,
        SumOperator,
    )
] = mul_any_times_sum_operator
