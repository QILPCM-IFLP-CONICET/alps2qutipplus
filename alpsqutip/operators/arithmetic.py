# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Classes and functions for operator arithmetic.
"""

import logging
from numbers import Number
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
        self,
        term_tuple: tuple,
        system=None,
        isherm: Optional[bool] = None,
        isdiag: Optional[bool] = None,
    ):
        assert system is not None
        assert isinstance(term_tuple, tuple)
        self.terms = term_tuple
        if system is None and term_tuple:
            for term in term_tuple:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

        # sites=tuple(system.dimensions.keys())
        # assert all(sites==tuple(t.system.dimensions.keys()) for t in term_tuple if t.system), f"{system.dimensions.keys()} and {tuple((tuple(t.system.dimensions.keys()) for t in term_tuple if t.system))}"
        self.system = system
        self._isherm = isherm
        self._isdiagonal = isdiag

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
        return SumOperator(tuple(-t for t in self.terms), self.system, self._isherm)

    def __repr__(self):
        return "(\n" + "\n  +".join(repr(t) for t in self.terms) + "\n)"

    def acts_over(self):
        result = set()
        for term in self.terms:
            term_acts_over = term.acts_over()
            if term_acts_over is None:
                return None
            result = result.union(term_acts_over)
        return result

    def dag(self):
        """return the adjoint operator"""
        if self._isherm:
            return self
        return SumOperator(tuple(term.dag() for term in self.terms), self.system)

    def flat(self):
        terms = []
        changed = False
        for term in self.terms:
            if isinstance(term, SumOperator):
                term_flat = term.flat()
                if hasattr(term_flat, "terms"):
                    terms.extend(term_flat.terms)
                else:
                    terms.append(term_flat)
                changed = True
            else:
                new_term = term.flat()
                terms.append(new_term)
                if term is not new_term:
                    changed = True
        if changed:
            return SumOperator(tuple(terms), self.system)
        return self

    @property
    def isherm(self) -> bool:
        isherm = self._isherm

        def aggresive_hermitician_test():
            # pylint: disable=import-outside-toplevel
            from alpsqutip.operators.functions import (
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

    @property
    def isdiagonal(self) -> bool:
        if self._isdiagonal is None:
            simplified = self.simplify()
            self._isdiagonal = all(term.isdiagonal for term in simplified.terms)
        return self._isdiagonal

    def partial_trace(self, sites: Union[list, SystemDescriptor]):
        if not isinstance(sites, SystemDescriptor):
            sites = self.system.subsystem(sites)
        new_terms = (term.partial_trace(sites) for term in self.terms)
        return sum(new_terms)

    def simplify(self):
        """Simplify the operator"""
        system = self.system
        general_terms = []
        isherm = self._isherm
        # First, shallow the list of terms:
        for term in (t.flat() for t in self.terms):
            if isinstance(term, SumOperator):
                general_terms.extend(term.terms)
            else:
                general_terms.append(term)

        terms = general_terms
        # Now, collect and sum LocalOperator and QutipOperator terms
        general_terms = []
        site_terms = {}
        qutip_terms = []
        scalar_term = 0
        last_scalar_term = None
        for term in terms:
            if isinstance(term, ScalarOperator):
                last_scalar_term = term
                scalar_term += term.prefactor
            elif isinstance(term, LocalOperator):
                if term:
                    site_terms.setdefault(term.site, []).append(term)
            elif isinstance(term, QutipOperator):
                if term:
                    qutip_terms.append(term)
            else:
                if term:
                    general_terms.append(term)

        loc_ops_lst = [
            (
                LocalOperator(site, sum(lop.operator for lop in l_ops), system)
                if len(l_ops) > 1
                else l_ops[0]
            )
            for site, l_ops in site_terms.items()
        ]

        if len(qutip_terms) > 1:
            qutip_terms = [sum(qutip_terms)]

        is_one_body = len(general_terms) == 0 and len(qutip_terms) == 0
        terms = general_terms + loc_ops_lst + qutip_terms

        if scalar_term:
            # If we found just one non-trivial scalar term, do not create a new one
            if last_scalar_term is None or last_scalar_term.prefactor != scalar_term:
                last_scalar_term = ScalarOperator(scalar_term, system)
            terms += [last_scalar_term]

        num_terms = len(terms)
        if num_terms == 0:
            return ScalarOperator(0, system)
        if num_terms == 1:
            return terms[0]
        if num_terms == len(self.terms) and all(
            old_t is new_t for old_t, new_t in zip(self.terms, terms)
        ):
            return self
        # If all the terms are LocalOperators or ScalarOperators, return a OneBodyOperator
        if is_one_body:
            return OneBodyOperator(tuple(terms), system, isherm)
        return SumOperator(tuple(terms), system, isherm)

    def to_qutip(self):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip()

        return sum(t.to_qutip() for t in self.terms)

    def tr(self):
        return sum(t.tr() for t in self.terms)

    def tidyup(self, atol=None):
        """Removes small elements from the quantum object."""
        tidy_terms = [term.tidyup(atol) for term in self.terms]
        tidy_terms = tuple((term for term in tidy_terms if term))
        isherm = all(term.isherm for term in tidy_terms) or None
        return SumOperator(tidy_terms, self.system, isherm=isherm)


NBodyOperator = SumOperator


class OneBodyOperator(SumOperator):
    """A linear combination of local operators"""

    def __init__(
        self,
        terms,
        system=None,
        check_and_convert=True,
        isherm: Optional[bool] = None,
        isdiag: Optional[bool] = None,
    ):
        """
        if check_and_convert is True,
        """
        assert isinstance(terms, tuple)
        assert system is not None

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

        super().__init__(terms, system, isherm, isdiag)

    def __repr__(self):
        return "  " + "\n  +".join("(" + repr(term) + ")" for term in self.terms)

    def __neg__(self):
        return OneBodyOperator(tuple(-term for term in self.terms), self.system)

    def dag(self):
        return OneBodyOperator(
            tuple(term.dag() for term in self.terms),
            system=self.system,
            check_and_convert=False,
        )

    def expm(self):
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators.functions import eigenvalues

        sites_op = {}
        ln_prefactor = 0
        for term in self.simplify().terms:
            if not bool(term):
                assert False, "No empty terms should reach here"
                continue
            if isinstance(term, ScalarOperator):
                ln_prefactor += term.prefactor
                continue
            operator_qt = term.operator
            try:
                k_0 = max(
                    np.real(
                        eigenvalues(operator_qt, sparse=True, sort="high", eigvals=3)
                    )
                )
            except ValueError:
                k_0 = max(np.real(eigenvalues(operator_qt, sort="high")))

            operator_qt = operator_qt - k_0
            ln_prefactor += k_0
            if hasattr(operator_qt, "expm"):
                sites_op[term.site] = operator_qt.expm()
            else:
                logging.warning(f"{type(operator_qt)} evaluated as a number")
                sites_op[term.site] = np.exp(operator_qt)

        prefactor = np.exp(ln_prefactor)
        return ProductOperator(sites_op, prefactor=prefactor, system=self.system)

    @staticmethod
    def _simplify_terms(terms, system):
        """Group terms by subsystem and process scalar terms"""
        simply_terms = [term.simplify() for term in terms]
        terms = []
        terms_by_subsystem = {}
        scalar_term_value = 0
        scalar_term = None

        for term in simply_terms:
            if isinstance(term, SumOperator):
                terms.extend(term.terms)
            elif isinstance(term, (ScalarOperator, LocalOperator)):
                terms.append(term)
            else:
                raise ValueError(
                    f"A OneBodyOperator can not have {type(term)} as a term."
                )
        # Now terms are just scalars and local operators.

        for term in terms:
            if isinstance(term, ScalarOperator):
                scalar_term = term
                scalar_term_value += term.prefactor
            elif isinstance(term, LocalOperator):
                terms_by_subsystem.setdefault(term.site, []).append(term)

        if scalar_term_value == 0:
            scalar_term = None
            terms = []
        elif scalar_term_value == scalar_term.prefactor:
            terms = [scalar_term]
        else:
            terms = [ScalarOperator(scalar_term_value, system)]

        # Reduce the local terms
        for site, local_terms in terms_by_subsystem.items():
            if len(local_terms) > 1:
                terms.append(sum(local_terms))
            else:
                terms.extend(local_terms)

        return tuple(terms), system

    def tidyup(self, atol=None):
        """Removes small elements from the quantum object."""
        tidy_terms = [term.tidyup(atol) for term in self.terms]
        tidy_terms = tuple((term for term in tidy_terms if term))
        isherm = all(term.isherm for term in tidy_terms) or None
        isdiag = all(term.isdiag for term in tidy_terms) or None
        return OneBodyOperator(tidy_terms, self.system, isherm=isherm, isdiag=isdiag)


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
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return SumOperator(terms, x_op.system, isherm)


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
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
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
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
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
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return SumOperator(terms, system, isherm)


# Sum with LocalOperator


@Operator.register_mul_handler(
    (
        LocalOperator,
        SumOperator,
    )
)
def _(y_op: LocalOperator, x_op: SumOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system

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
    system = x_op.system * y_op.system if x_op.system else y_op.system

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
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = tuple(
        factor_x * factor_y for factor_x in x_op.terms for factor_y in y_op.terms
    )
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]

    if all(
        acts_over and len(acts_over) < 2
        for acts_over in (term.acts_over() for term in terms)
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
    system = x_op.system * y_op.system if x_op.system else y_op.system
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
    system = x_op.system * y_op.system if x_op.system else y_op.system
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
    result = OneBodyOperator(terms, system)
    return result


@Operator.register_add_handler(
    (
        OneBodyOperator,
        LocalOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: LocalOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
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


@Operator.register_add_handler(
    (
        ScalarOperator,
        LocalOperator,
    )
)
def _(x_op: ScalarOperator, y_op: LocalOperator):
    if x_op.prefactor == 0:
        return y_op

    system = y_op.system or x_op.system
    site = y_op.site
    return LocalOperator(site, y_op.operator + x_op.prefactor, system)


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
        return LocalOperator(first_site, first_loc_op * prefactor + y_value, system)
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
    system = x_op.system * y_op.system if x_op.system else y_op.system
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
    system = x_op.system * y_op.system if x_op.system else y_op.system
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
