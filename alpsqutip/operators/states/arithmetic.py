"""
Arithmetic operations with states.

Essentially, arithmetic operations with states involves just mixing of operators,
implemented though the class MixtureDensityOperator.

"""

from numbers import Number
from typing import Iterable, Optional, Tuple, Union

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

    terms: Tuple[DensityOperatorMixin]

    def __init__(self, terms: tuple, system: SystemDescriptor = None):
        super().__init__(terms, system, True)

    def __add__(self, rho: Operator):
        terms = self.terms
        system = self.system

        if isinstance(rho, MixtureDensityOperator):
            terms = terms + rho.terms
        elif isinstance(rho, DensityOperatorMixin):
            terms = terms + (rho,)
        elif isinstance(rho, (int, float)) and rho >= 0:
            terms = terms + (ProductDensityOperator({}, rho, system, False),)
        else:
            # return super().__add__(rho)
            return SumOperator(terms, system) + rho
        return MixtureDensityOperator(terms, system)

    def __mul__(self, a):
        if isinstance(a, float) and a >= 0:
            return MixtureDensityOperator(tuple(term * a for term in self.terms))
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
            tuple((term * a) * term.prefactor for term in self.terms), self.system
        )

    def __radd__(self, rho: Operator):
        terms = self.terms
        system = self.system

        if isinstance(rho, MixtureDensityOperator):
            terms = rho.terms + terms
        elif isinstance(rho, DensityOperatorMixin):
            terms = (rho,) + terms
        elif isinstance(rho, (int, float)) and rho >= 0:
            terms = (ProductDensityOperator({}, rho, system, False),) + terms
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
            tuple(a * term * term.prefactor for term in self.terms), self.system
        )

    def acts_over(self) -> set:
        """
        Return a set with the name of the
        sites where the operator nontrivially acts
        """
        sites: set = set()
        for term in self.terms:
            sites.update(term.acts_over())
        return sites

    def expect(self, obs: Union[Operator, Iterable]) -> Union[np.ndarray, dict, Number]:
        strip = False
        if isinstance(obs, Operator):
            strip = True
            obs = [obs]

        av_terms = tuple((term.expect(obs), term.prefactor) for term in self.terms)

        if isinstance(obs, dict):
            return {
                op_name: sum(term[0][op_name] * term[1] for term in av_terms)
                for op_name in obs
            }
        if strip:
            return sum(np.array(term[0]) * term[1] for term in av_terms)[0]
        return sum(np.array(term[0]) * term[1] for term in av_terms)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        new_terms = tuple(t.partial_trace(sites) for t in self.terms)
        subsystem = new_terms[0].system
        return MixtureDensityOperator(new_terms, subsystem)

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip()

        if block is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(self.acts_over()) if site not in block)
            )

        # TODO: find a more efficient way to avoid element-wise
        # multiplications
        return sum(term.to_qutip(block) * term.prefactor for term in self.terms)
