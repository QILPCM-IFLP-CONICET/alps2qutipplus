# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Classes and functions for operator arithmetic.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.settings import ALPSQUTIP_TOLERANCE


class SumOperator(Operator):
    """Represents a linear combination of operators"""

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
        assert len(term_tuple) > 0
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
            result += f" + \\ldots ({len(terms)-6} terms) \\ldots + "
            result += " + ".join(term._repr_latex_()[1:-1] for term in terms[-3:])
        else:
            result = " + ".join(term._repr_latex_()[1:-1] for term in terms)
        return f"${result}$"

    def acts_over(self):
        """ """
        result = set()
        for term in self.terms:
            term_acts_over = term.acts_over()
            result = result.union(term_acts_over)
        return frozenset(result)

    def dag(self):
        """ """
        if self._isherm:
            return self
        return SumOperator(tuple(term.dag() for term in self.terms), self.system)

    def flat(self):
        """Use the associativity to write the sum of sums
        as a sum of non sum terms.

        Returns
        -------
        The object in flat representation, it. if the object is a sum,
        the terms of objects that are not sums.
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
                assert isinstance(
                    new_term, Operator
                ), f"{type(term)} produces type({new_term})"
                terms.append(new_term)
                if term is not new_term:
                    changed = True
        if changed:
            return SumOperator(tuple(terms), self.system)
        return self

    @property
    def isherm(self) -> bool:
        """ """
        isherm = self._isherm

        def aggresive_hermitician_test(non_hermitian_tuple: Tuple[Operator]):
            """Determine if the antihermitician part is zero

            Parameters
            ----------
            non_hermitian_tuple: Tuple[Operator] :


            Returns
            -------

            """
            # Here we assume that after simplify, the operator is a single term
            # (not a SumOperator), a OneBodyOperator, or a sum of a one-body operator
            # and terms acting over an specific block.
            nh_sum = SumOperator(non_hermitian_tuple, self.system).simplify()
            if not hasattr(nh_sum, "terms"):
                self._isherm = nh_sum.isherm
                return self._isherm

            # Hermitician until the opposite is shown:
            isherm = True
            for term in nh_sum.terms:
                term_isherm = term.isherm
                # if term_isherm could not determine by itself if the
                # term is hermitician, try harder looking at the frobenious norm
                # of its anti-hermitician part. This step can be very costly...
                if term_isherm is None:
                    # Last resource:
                    ah_part = term - term.dag()
                    term_isherm = abs((ah_part * ah_part).tr()) < ALPSQUTIP_TOLERANCE
                if not term_isherm:
                    isherm = False
                    break
            self._isherm = isherm
            return isherm

        if isherm is None:
            # First, collect the non-hermitician terms
            non_hermitian = tuple((term for term in self.terms if not term.isherm))
            # If there are non-hermitician terms, try the more aggressive strategy
            # over these terms.
            if non_hermitian:
                return aggresive_hermitician_test(non_hermitian)

            self._isherm = True
            return True

        return bool(self._isherm)

    @property
    def isdiagonal(self) -> bool:
        """ """
        if self._isdiagonal is None:
            simplified = self if self._simplified else self.simplify()
            try:
                self._isdiagonal = all(term.isdiagonal for term in simplified.terms)
            except AttributeError:
                self._isdiagonal = simplified.isdiagonal
        return self._isdiagonal

    @property
    def is_zero(self) -> bool:
        """ """
        simplify_self = self if self._simplified else self.simplify()
        if hasattr(simplify_self, "terms"):
            result = all(term.is_zero for term in simplify_self.terms)
        else:
            result = simplify_self.is_zero
        if result:
            self._isherm = True
        return result

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace

        Parameters
        ----------
        sites: Union[frozenset :

        SystemDescriptor] :


        Returns
        -------

        """
        if not isinstance(sites, SystemDescriptor):
            sites = self.system.subsystem(sites)
        new_terms = tuple((term.partial_trace(sites) for term in self.terms))
        subsystem = new_terms[0].system
        new_terms = tuple((term for term in new_terms if term))
        if len(new_terms) == 0:
            return ScalarOperator(0, subsystem)
        if len(new_terms) == 1:
            return new_terms[0]
        return SumOperator(new_terms, subsystem)

    def simplify(self):
        """Simplify the operator"""
        from alpsqutip.operators.simplify import group_terms_by_blocks

        if self._simplified:
            return self
        if len(self.terms) == 1:
            return self.terms[0].simplify()

        return group_terms_by_blocks(self.flat().tidyup())

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        """Produce a qutip compatible object

        Parameters
        ----------
        block: Optional[Tuple[str]] :
             (Default value = None)

        Returns
        -------
        Return a Qutip Qobj representation of the object, acting
        on the smallest subsystem containing the block.
        """
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
        """ """
        return sum(t.tr() for t in self.terms)

    def tidyup(self, atol=None):
        """Removes small elements from the quantum object.

        Parameters
        ----------
        atol :
             the tolerance in the truncation (Default value = None)

        Returns
        -------
           A new Operator in whose representation
           all the numbers with modules smaller that `atol`
           are replaced by `0`
        """
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
            """

            Parameters
            ----------
            terms :

            system :


            Returns
            -------

            """
            for term in terms:
                if not hasattr(term, "system"):
                    continue
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
            # Ensure that all the terms are operators.
            terms = [
                term if isinstance(term, Operator) else ScalarOperator(term, system)
                for term in terms
            ]
            terms, system = self._simplify_terms(terms, system)
            simplified = True
            if len(terms) == 0:
                terms = tuple((ScalarOperator(0.0, system),))

        super().__init__(
            terms, system=system, isherm=isherm, isdiag=isdiag, simplified=simplified
        )

    def __repr__(self):
        return "  " + "\n  +".join("(" + repr(term) + ")" for term in self.terms)

    def __neg__(self):
        return OneBodyOperator(tuple(-term for term in self.terms), self.system)

    def dag(self):
        """ """
        return OneBodyOperator(
            tuple(term.dag() for term in self.terms),
            system=self.system,
            check_and_convert=False,
        )

    def expm(self):
        """ """
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
        """ """
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
                        tuple(term.acts_over())[0],
                        term.operator * term.prefactor,
                        system=term.system,
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

        if scalar_term is None:
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
        """Removes small elements from the quantum object.

        Parameters
        ----------
        atol :
             the tolerance upto the value of a number
        is considered zero(Default value = None)

        Returns
        -------
        Return a new Operator whose matrix elements with absolute
        values smaller than tolerance are replaced by zero.
        """
        tidy_terms = [term.tidyup(atol) for term in self.terms]
        tidy_terms = tuple((term for term in tidy_terms if term))
        isherm = all(term.isherm for term in tidy_terms) or None
        isdiag = all(term.isdiagonal for term in tidy_terms) or None
        return OneBodyOperator(tidy_terms, self.system, isherm=isherm, isdiag=isdiag)
