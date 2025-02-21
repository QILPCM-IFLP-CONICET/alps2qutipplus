# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Classes and functions for operator arithmetic.
"""

import logging
from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
from qutip import Qobj  # type: ignore[import-untyped]

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

    terms: Tuple[Operator]

    def __init__(
        self,
        term_tuple: tuple,
        system=None,
        isherm: Optional[bool] = None,
        isdiag: Optional[bool] = None,
        simplified: Optional[bool] = False,
    ):
        assert system is not None
        assert isinstance(term_tuple, tuple)
        assert self not in term_tuple, "cannot be a term of myself."
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
        self._simplified = simplified

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

    def _repr_latex_(self):
        """LaTeX Representation"""
        terms = self.terms
        if len(terms) > 6:
            result = " + ".join(term._repr_latex_()[1:-1] for term in terms[:3])
            result += " + ... + "
            result = " + ".join(term._repr_latex_()[1:-1] for term in terms[-3:])
        else:
            result = " + ".join(term._repr_latex_()[1:-1] for term in terms)
        return f"${result}$"

    def acts_over(self):
        result = set()
        for term in self.terms:
            term_acts_over = term.acts_over()
            result = result.union(term_acts_over)
        return result

    def dag(self):
        """return the adjoint operator"""
        if self._isherm:
            return self
        return SumOperator(tuple(term.dag() for term in self.terms), self.system)

    def flat(self):
        """
        Use the associativity to write the sum of sums
        as a sum of non sum terms.
        """
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

        def aggresive_hermitician_test(non_hermitian_tuple: Tuple[Operator]):
            """Determine if the antihermitician part is zero"""
            # Here we assume that after simplify, the operator is a single term
            # (not a SumOperator), a OneBodyOperator, or a sum of a one-body operator
            # and terms acting over an specific block.
            nh_sum = SumOperator(non_hermitian_tuple, self.system).simplify()
            if not hasattr(nh_sum, "terms"):
                self._isherm = nh_sum.isherm
                return self._isherm

            # Hermitician until the opposite is shown:
            for term in nh_sum.terms:
                if isinstance(term, OneBodyOperator) or not isinstance(
                    term, SumOperator
                ):
                    if not term.isherm:
                        self._isherm = False
                        return False
                # If the term is a sum of many-body terms acting on a site,
                # check if the HS norm of the antihermitician part is not zero.
                ah_part = term - term.dag()
                if abs((ah_part * ah_part).tr()) > 1e-10:
                    self._isherm = False
                    return False
            self._isherm = True
            return True

        if isherm is None:
            # First, try with the less aggressive test:
            non_hermitian = tuple((term for term in self.terms if not term.isherm))
            if non_hermitian:
                return aggresive_hermitician_test(non_hermitian)

            self._isherm = True
            return True

        return bool(self._isherm)

    @property
    def isdiagonal(self) -> bool:
        if self._isdiagonal is None:
            simplified = self if self._simplified else self.simplify()
            try:
                self._isdiagonal = all(term.isdiagonal for term in simplified.terms)
            except AttributeError:
                self._isdiagonal = simplified.isdiagonal
        return self._isdiagonal

    @property
    def is_zero(self) -> bool:
        simplify_self = self if self._simplified else self.simplify()
        if hasattr(simplify_self, "terms"):
            return all(term.is_zero for term in simplify_self.terms)
        return simplify_self.is_zero

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        if not isinstance(sites, SystemDescriptor):
            sites = self.system.subsystem(sites)
        new_terms = (term.partial_trace(sites) for term in self.terms)
        return sum(new_terms)

    def simplify(self):
        """Simplify the operator"""
        from alpsqutip.operators.simplify import group_terms_by_blocks

        if self._simplified:
            return self

        return group_terms_by_blocks(self.flat().tidyup())

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        """Produce a qutip compatible object"""
        terms = self.terms
        system = self.system
        assert all(t.system is system for t in terms)
        if block is None:
            block = tuple(sorted(self.acts_over() if system is None else system.sites))
        else:
            block = block + tuple(
                sorted(site for site in self.acts_over() if site not in block)
            )
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip(block)

        qutip_terms = (t.to_qutip(block) for t in terms)
        result = sum(qutip_terms)
        return result

    def tr(self):
        return sum(t.tr() for t in self.terms)

    def tidyup(self, atol=None):
        """Removes small elements from the quantum object."""
        tidy_terms = [term.tidyup(atol) for term in self.terms]
        tidy_terms = tuple((term for term in tidy_terms if term))
        if len(tidy_terms) == 0:
            return ScalarOperator(0, self.system)
        if len(tidy_terms) == 1:
            return tidy_terms[0]
        isherm = all(term.isherm for term in tidy_terms) or None
        isdiag = all(term.isdiagonal for term in tidy_terms) or None
        return SumOperator(tidy_terms, self.system, isherm=isherm, isdiag=isdiag)


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
        simplified: Optional[bool] = False,
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
            simplified = True

        super().__init__(
            terms, system=system, isherm=isherm, isdiag=isdiag, simplified=simplified
        )

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

    def simplify(self):
        if self._simplified:
            return self
        terms, system = self._simplify_terms(self.terms, self.system)
        num_terms = len(terms)
        if num_terms == 0:
            return ScalarOperator(0, system)
        if num_terms == 1:
            return terms[0]
        return OneBodyOperator(
            terms, system, isherm=self._isherm, isdiag=self._isdiagonal, simplified=True
        )

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
            elif isinstance(term, QutipOperator):
                terms.append(
                    LocalOperator(
                        tuple(term.acts_over())[0], term.operator, system=term.system
                    )
                )
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
        isdiag = all(term.isdiagonal for term in tidy_terms) or None
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
@Operator.register_mul_handler(
    (
        int,
        SumOperator,
    )
)
@Operator.register_mul_handler(
    (
        float,
        SumOperator,
    )
)
@Operator.register_mul_handler(
    (
        complex,
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
    system = x_op.system or y_op.system
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
    system = x_op.system or y_op.system
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



