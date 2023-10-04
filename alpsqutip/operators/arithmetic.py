# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Classes and functions for operator arithmetic.
"""

from numbers import Number

# import logging
from typing import List, Optional, Union

import numpy as np
from qutip import Qobj

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.qutip import QutipOperator


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
        assert system is not None
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

    def __neg__(self):
        return SumOperator(
            tuple(-t for t in self.terms), self.system, self._isherm
        )

    def __repr__(self):
        return "(\n" + "\n  +".join(repr(t) for t in self.terms) + "\n)"

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
            # pylint: disable=import-outside-toplevel
            from alpsqutip.operator_functions import (
                hermitian_and_antihermitian_parts,
                simplify_sum_operator,
            )

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
            term.partial_trace(sites) * term.prefactor for term in self.terms
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
        assert system is not None
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

    def __repr__(self):
        return "  " + "\n  +".join(
            "(" + repr(term) + ")" for term in self.terms
        )

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

            terms_list = next(iter(terms_by_subsystem.values()))
            first_term_plus_scalar = terms_list[0] + scalar_term
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


# #####################################
# Arithmetic operations
# ####################################


# #######################################################
#               Sum operators
# #######################################################


# Sum with numbers


@Operator.register_add_handler(
    (
        SumOperator,
        Number,
    )
)
def _(x_op: SumOperator, y_value: Number):
    return x_op + ScalarOperator(y_value, x_op.system)


@Operator.register_mul_handler(
    (
        SumOperator,
        Number,
    )
)
def _(x_op: SumOperator, y_value: Number):
    if y_value == 0:
        return ScalarOperator(0, x_op.system)

    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (
        not isinstance(y_value, complex) or y_value.imag == 0
    )
    return SumOperator(terms, x_op.system, isherm).simplify()


@Operator.register_mul_handler(
    (
        Number,
        SumOperator,
    )
)
def _(y_value: Number, x_op: SumOperator):
    if y_value == 0:
        return ScalarOperator(0, x_op.system)

    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (
        not isinstance(y_value, complex) or y_value.imag == 0
    )
    return SumOperator(terms, x_op.system, isherm).simplify()


# Sum with ScalarOperator


@Operator.register_mul_handler(
    (
        SumOperator,
        ScalarOperator,
    )
)
def _(x_op: SumOperator, y_op: ScalarOperator):
    system = x_op.system or y_op.system()
    y_value = y_op.prefactor
    if y_value == 0:
        return ScalarOperator(0, system)

    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (
        not isinstance(y_value, complex) or y_value.imag == 0
    )
    return SumOperator(terms, system, isherm)


@Operator.register_mul_handler(
    (
        ScalarOperator,
        SumOperator,
    )
)
def _(y_op: ScalarOperator, x_op: SumOperator):
    system = x_op.system or y_op.system()
    y_value = y_op.prefactor
    if y_value == 0:
        return ScalarOperator(0, system)

    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (
        not isinstance(y_value, complex) or y_value.imag == 0
    )
    return SumOperator(terms, system, isherm)


# Sum with LocalOperator


@Operator.register_mul_handler(
    (
        LocalOperator,
        SumOperator,
    )
)
def _(y_op: LocalOperator, x_op: SumOperator):
    system = x_op.system or y_op.system()

    terms_it = (y_op * term for term in x_op.terms)
    terms = tuple(term for term in terms_it if bool(term))
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and y_op.isherm
    return SumOperator(terms, system, isherm)


@Operator.register_mul_handler(
    (
        SumOperator,
        LocalOperator,
    )
)
def _(x_op: SumOperator, y_op: LocalOperator):
    system = x_op.system or y_op.system()

    terms_it = (term * y_op for term in x_op.terms)
    terms = tuple(term for term in terms_it if bool(term))
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and y_op.isherm
    return SumOperator(terms, system, isherm)


# SumOperator and any Operator


@Operator.register_add_handler(
    (
        SumOperator,
        Operator,
    )
)
def _(x_op: SumOperator, y_op: Operator):
    system = x_op.system or y_op.system
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and y_op.isherm
    return SumOperator(terms, system, isherm)


# SumOperator plus SumOperator


@Operator.register_add_handler(
    (
        SumOperator,
        SumOperator,
    )
)
def _(x_op: SumOperator, y_op: SumOperator):
    system = x_op.system or y_op.system
    terms = x_op.terms + y_op.terms
    isherm = x_op._isherm and y_op._isherm
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(terms, system, isherm)


@Operator.register_mul_handler(
    (
        SumOperator,
        SumOperator,
    )
)
def _(x_op: SumOperator, y_op: SumOperator):
    system = x_op.system or y_op.system
    terms = tuple(
        factor_x * factor_y
        for factor_x in x_op.terms
        for factor_y in y_op.terms
    )
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]

    if all(
        act_over and len(act_over) < 2
        for act_over in (term.act_over() for term in terms)
    ):
        return OneBodyOperator(terms, system, False)
    return SumOperator(terms, system)


@Operator.register_mul_handler(
    (
        SumOperator,
        Operator,
    )
)
@Operator.register_mul_handler(
    (
        SumOperator,
        Qobj,
    )
)
def _(x_op: SumOperator, y_op: Union[Operator, Qobj]):
    system = x_op.system or y_op.system
    terms = tuple(factor_x * y_op for factor_x in x_op.terms)
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(terms, system)


@Operator.register_mul_handler(
    (
        Operator,
        SumOperator,
    )
)
@Operator.register_mul_handler(
    (
        Qobj,
        SumOperator,
    )
)
def _(y_op: Union[Operator, Qobj], x_op: SumOperator):
    system = x_op.system or y_op.system
    terms = tuple(y_op * factor_x for factor_x in x_op.terms)
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(terms, system)


# ######################
#
#   OneBodyOperator
#
# ######################


@Operator.register_add_handler(
    (
        OneBodyOperator,
        OneBodyOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: OneBodyOperator):
    system = x_op.system or y_op.system
    terms = x_op.terms + y_op.terms
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_add_handler(
    (
        OneBodyOperator,
        Number,
    )
)
def _(x_op: OneBodyOperator, y_value: Number):
    system = x_op.system
    y_op = ScalarOperator(y_value, system)
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_add_handler(
    (
        OneBodyOperator,
        ScalarOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: ScalarOperator):
    system = x_op.system or y_op.system
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_add_handler(
    (
        OneBodyOperator,
        LocalOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: LocalOperator):
    system = x_op.system or y_op.system
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler(
    (
        OneBodyOperator,
        Number,
    )
)
@Operator.register_mul_handler(
    (
        OneBodyOperator,
        int,
    )
)
@Operator.register_mul_handler(
    (
        OneBodyOperator,
        float,
    )
)
@Operator.register_mul_handler(
    (
        OneBodyOperator,
        complex,
    )
)
def _(x_op: OneBodyOperator, y_value: Number):
    system = x_op.system
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler(
    (
        Number,
        OneBodyOperator,
    )
)
@Operator.register_mul_handler(
    (
        int,
        OneBodyOperator,
    )
)
@Operator.register_mul_handler(
    (
        float,
        OneBodyOperator,
    )
)
@Operator.register_mul_handler(
    (
        complex,
        OneBodyOperator,
    )
)
def _(y_value: Number, x_op: OneBodyOperator):
    system = x_op.system
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler(
    (
        OneBodyOperator,
        ScalarOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: ScalarOperator):
    system = x_op.system
    y_value = y_op.prefactor
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler(
    (
        ScalarOperator,
        OneBodyOperator,
    )
)
def _(y_op: ScalarOperator, x_op: OneBodyOperator):
    system = x_op.system
    y_value = y_op.prefactor
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


# ######################
#
#   LocalOperator
#
# ######################


@Operator.register_add_handler(
    (
        LocalOperator,
        LocalOperator,
    )
)
def _(x_op: LocalOperator, y_op: LocalOperator):
    system = x_op.system or y_op.system
    site1 = x_op.site
    site2 = y_op.site
    if site1 == site2:
        return LocalOperator(site1, x_op.operator + y_op.operator, system)
    return OneBodyOperator(
        (
            x_op,
            y_op,
        ),
        system,
        False,
    )


# ######################
#
#   ProductOperator
#
# ######################


@Operator.register_add_handler(
    (
        ProductOperator,
        Number,
    )
)
def _(x_op: ProductOperator, y_value: Number):
    site_op = x_op.sites_op.copy()
    prefactor = x_op.prefactor
    system = x_op.system
    if len(site_op) == 0:
        return ScalarOperator(prefactor + y_value, system)
    if len(site_op) == 1:
        first_site, first_loc_op = next(iter(site_op.items()))
        return LocalOperator(
            first_site, first_loc_op * prefactor + y_value, system
        )
    y_op = ScalarOperator(y_value, system)
    return SumOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )


@Operator.register_add_handler(
    (
        ProductOperator,
        ScalarOperator,
    )
)
def _(x_op: ProductOperator, y_op: ScalarOperator):
    site_op = x_op.sites_op.copy()
    prefactor = x_op.prefactor
    system = x_op.system or y_op.system
    if len(site_op) == 0:
        return ScalarOperator(prefactor + y_op.prefactor, system)
    if len(site_op) == 1:
        first_site, first_loc_op = next(iter(site_op.items()))
        return LocalOperator(
            first_site, first_loc_op * prefactor + y_op.prefactor, system
        )

    return SumOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )


@Operator.register_add_handler(
    (
        ProductOperator,
        ProductOperator,
    )
)
def _(x_op: ProductOperator, y_op: ProductOperator):
    system = x_op.system or y_op.system
    site_op_x = x_op.sites_op
    site_op_y = y_op.sites_op
    if len(site_op_x) > 1 or len(site_op_y) > 1:
        return SumOperator(
            (
                x_op,
                y_op,
            ),
            system,
        )
    return x_op.simplify() + y_op.simplify()


@Operator.register_add_handler(
    (
        ProductOperator,
        LocalOperator,
    )
)
def _(x_op: ProductOperator, y_op: LocalOperator):
    system = x_op.system or y_op.system
    site_op_x = x_op.sites_op
    if len(site_op_x) > 1:
        return SumOperator(
            (
                x_op,
                y_op,
            ),
            system,
        )
    return x_op.simplify() + y_op.simplify()


# #######################################
#
#  QutipOperator
#
# #######################################


@Operator.register_add_handler(
    (
        QutipOperator,
        Operator,
    )
)
def _(x_op: QutipOperator, y_op: Operator):
    system = x_op.system or y_op.system
    return SumOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )


@Operator.register_mul_handler(
    (
        QutipOperator,
        ScalarOperator,
    )
)
def _(x_op: QutipOperator, y_op: ScalarOperator):
    system = x_op.system or y_op.system
    return QutipOperator(
        x_op.operator, system, x_op.site_names, x_op.prefactor * y_op.prefactor
    )


@Operator.register_mul_handler(
    (
        ScalarOperator,
        QutipOperator,
    )
)
def _(y_op: ScalarOperator, x_op: QutipOperator):
    system = x_op.system or y_op.system
    return QutipOperator(
        x_op.operator, system, x_op.site_names, x_op.prefactor * y_op.prefactor
    )


@Operator.register_mul_handler(
    (
        QutipOperator,
        LocalOperator,
    )
)
@Operator.register_mul_handler(
    (
        QutipOperator,
        ProductOperator,
    )
)
@Operator.register_mul_handler(
    (
        QutipOperator,
        SumOperator,
    )
)
def _(x_op: QutipOperator, y_op: Operator):
    return x_op * y_op.to_qutip()


@Operator.register_mul_handler(
    (
        LocalOperator,
        QutipOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductOperator,
        QutipOperator,
    )
)
@Operator.register_mul_handler(
    (
        SumOperator,
        QutipOperator,
    )
)
def _(y_op: Operator, x_op: QutipOperator):
    return y_op.to_qutip() * x_op
