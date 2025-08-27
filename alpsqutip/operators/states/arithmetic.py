"""
Arithmetic operations with states.

Essentially, arithmetic operations with states involves just mixing of operators,
implemented though the class MixtureDensityOperator.

"""

import logging
import pickle
from numbers import Number
from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import SumOperator
from alpsqutip.operators.basic import (
    Operator,
    ScalarOperator,
)
from alpsqutip.operators.states.basic import (
    DensityOperatorMixin,
    ProductDensityOperator,
)


class MixtureDensityOperator(DensityOperatorMixin, SumOperator):
    """
    A mixture of density operators
    """

    terms: Tuple[Operator]

    def __init__(self, terms: tuple, system: Optional[SystemDescriptor] = None):
        super().__init__(terms, system, True)

    def __add__(self, rho: Operator):
        terms: tuple[Operator] = self.terms
        system = self.system

        if isinstance(rho, MixtureDensityOperator):
            terms = cast(Tuple[Operator], terms + rho.terms)
        elif isinstance(rho, DensityOperatorMixin):
            terms = cast(Tuple[Operator], terms + (rho,))
        elif isinstance(rho, (int, float)) and rho >= 0:
            terms = cast(
                Tuple[Operator],
                terms + (ProductDensityOperator({}, rho, system, False),),
            )
        else:
            # return super().__add__(rho)
            return (
                SumOperator(
                    tuple((-(-term) * term.prefactor for term in terms)), system
                )
                + rho
            )
        return MixtureDensityOperator(terms, system)

    def __mul__(self, a):
        if isinstance(a, float) and a >= 0:
            return MixtureDensityOperator(
                tuple(term * a for term in self.terms), self.system
            )
        if isinstance(a, MixtureDensityOperator):
            return SumOperator(
                tuple(
                    (term * term_a) * (term.prefactor * term_a.prefactor)
                    for term in self.terms
                    for term_a in a.terms
                ),
                self.system,
            )
        if isinstance(a, SumOperator):
            return SumOperator(
                tuple(
                    (term * term_a) * term.prefactor
                    for term in self.terms
                    for term_a in a.terms
                ),
                self.system,
            )
        return SumOperator(
            tuple((-term * a) * (-term.prefactor) for term in self.terms), self.system
        )

    def __neg__(self):
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        new_terms = tuple(((-t) * (t.prefactor) for t in self.terms))
        return SumOperator(new_terms, self.system, isherm=True)

    def __radd__(self, rho: Operator):
        terms = self.terms
        system = self.system

        if isinstance(rho, MixtureDensityOperator):
            terms = cast(Tuple[Operator], rho.terms + terms)
        elif isinstance(rho, DensityOperatorMixin):
            terms = cast(Tuple[Operator], (rho,) + terms)
        elif isinstance(rho, (int, float)) and rho >= 0:
            terms = cast(
                Tuple[Operator],
                (ProductDensityOperator({}, rho, system, False),) + terms,
            )
        else:
            # return super().__add__(rho)
            return rho + SumOperator(terms, system)
        return MixtureDensityOperator(terms, system)

    def __rmul__(self, a):
        if isinstance(a, float) and a >= 0:
            return MixtureDensityOperator(
                tuple(term * a for term in self.terms), self.system
            )
        if isinstance(a, SumOperator):
            return SumOperator(
                tuple(
                    (
                        term_a * term * term.prefactor
                        for term in self.terms
                        for term_a in a.terms
                    )
                ),
                self.system,
            )
        return SumOperator(
            tuple((-a * term) * (-term.prefactor) for term in self.terms), self.system
        )

    def acts_over(self) -> Optional[frozenset]:
        """
        Return a set with the name of the
        sites where the operator nontrivially acts
        """
        sites: set = set()
        for term in self.terms:
            acts_over = cast(Operator, term).acts_over()
            if acts_over is None:
                return None
            sites.update(acts_over)
        return frozenset(sites)

    def expect(self, obs: Union[Operator, Iterable]) -> Union[np.ndarray, dict, Number]:

        def compute_results(curr_obs, sub_averages, prefactors):
            if isinstance(curr_obs, dict):
                result = {}
                for key in curr_obs:
                    content = curr_obs[key]
                    result[key] = compute_results(
                        content,
                        tuple(contrib[key] for contrib in sub_averages),
                        prefactors,
                    )
                return result
            # Operator, list or tuple, just return the linear combination, because exp_eval
            # is a tuple of Operator or ndarray objects.
            return sum(
                exp_val * p_refactor
                for exp_val, p_refactor in zip(sub_averages, prefactors)
            )

        averages = tuple(
            cast(DensityOperatorMixin, term).expect(obs) for term in self.terms
        )
        prefactors = tuple(term.prefactor for term in self.terms)
        return compute_results(obs, averages, prefactors)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        new_terms = tuple(cast(Operator, t).partial_trace(sites) for t in self.terms)
        subsystem = new_terms[0].system
        return MixtureDensityOperator(new_terms, subsystem)

    def simplify(self):
        # DensityOperator's are considered "simplified".
        return self

    def __setstate__(self, state):
        state = pickle.loads(state)
        self.__dict__.update(state)
        self._set_system_(self.system)

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip()

        acts_over = self.acts_over()
        if block is None or acts_over is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(acts_over) if site not in block)
            )

        # TODO: find a more efficient way to avoid element-wise
        # multiplications
        terms = (
            (
                cast(Operator, term).to_qutip(block),
                term.prefactor,
            )
            for term in self.terms
        )
        return sum(term[0] * term[1] for term in terms)
