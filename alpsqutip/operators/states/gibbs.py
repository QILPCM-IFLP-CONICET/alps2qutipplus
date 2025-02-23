"""
Classes to represent density operators as Gibbs states $rho=e^{-k}$.

"""

from numbers import Number
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator
from alpsqutip.operators.basic import LocalOperator, Operator, is_diagonal_op
from alpsqutip.operators.states.basic import (
    DensityOperatorMixin,
    ProductDensityOperator,
)
from alpsqutip.operators.states.utils import (
    k_by_site_from_operator,
    safe_exp_and_normalize,
)


class GibbsDensityOperator(DensityOperatorMixin, Operator):
    """
    Stores an operator of the form rho= prefactor * exp(-K) / Tr(exp(-K)).

    """

    free_energy: float
    normalized: bool
    k: Operator

    def __init__(
        self,
        k: Operator,
        system: Optional[SystemDescriptor] = None,
        prefactor=1.0,
        normalized=False,
    ):
        assert prefactor > 0
        self.k = k
        self.f_global = 0.0
        self.free_energy = 0.0
        self.prefactor = prefactor
        self.normalized = normalized
        self.system = k.system.union(system)

    def __mul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return GibbsDensityOperator(
                self.k,
                self.system,
                self.prefactor * operand,
                normalized=self.normalized,
            )
        return self.to_qutip_operator() * operand

    def __neg__(self):
        return -self.to_qutip_operator()

    def __rmul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return GibbsDensityOperator(
                self.k,
                self.system,
                self.prefactor * operand,
                normalized=self.normalized,
            )
        return operand * self.to_qutip_operator()

    def __truediv__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return GibbsDensityOperator(
                self.k,
                self.system,
                self.prefactor / operand,
                normalized=self.normalized,
            )
        if isinstance(operand, Operator):
            return self * operand.inv()
        raise ValueError("Division of an operator by ", type(operand), " not defined.")

    def acts_over(self) -> set:
        """
        Return a set with the name of the
        sites where the operator nontrivially acts
        """
        return self.k.acts_over()

    def expect(self, obs: Union[Operator, Iterable]) -> Union[np.ndarray, dict, Number]:
        return self.to_qutip_operator().expect(obs)

    def logm(self):
        self.normalize()
        k = self.k
        return -k

    def normalize(self) -> Operator:
        """Normalize the operator in a way that exp(-K).tr()==1"""
        if not self.normalized:
            rho, log_prefactor = safe_exp_and_normalize(
                -self.k
            )  # pylint: disable=unused-variable
            self.k = (self.k + log_prefactor).simplify()
            self.free_energy = -log_prefactor
            self.normalized = True
        return self

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        return self.to_qutip_operator().partial_trace(sites)

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        if block is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(self.acts_over()) if site not in block)
            )

        if not self.normalized:
            rho, log_prefactor = safe_exp_and_normalize(-self.k)
            self.k = self.k + log_prefactor
            self.free_energy = -log_prefactor
            self.normalized = True
            return rho.to_qutip(block)
        result = (-self.k).to_qutip(block).expm()
        return result


class GibbsProductDensityOperator(DensityOperatorMixin, Operator):
    """
    Stores an operator of the form
    rho = prefactor * \\otimes_i exp(-K_i)/Tr(exp(-K_i)).

    """

    k_by_site: Dict[str, Operator]
    prefactor: float
    free_energies: Dict[str, float]
    isherm: bool = True

    def __init__(
        self,
        k: Union[Operator, dict],
        prefactor: float = 1,
        system: Optional[SystemDescriptor] = None,
        normalized: bool = False,
    ):
        assert prefactor > 0.0

        self.prefactor = prefactor
        if isinstance(k, dict):
            self.system = system
            k_by_site = k
        else:
            try:
                k = k.simplify()
                if system:
                    system = k.system.union(system)
                else:
                    system = k.system
                self.system = system
                k_by_site = k_by_site_from_operator(k)
            except AttributeError:
                raise ValueError(
                    f"k_by_site must be a dictionary or an Operator. Got {type(k)}"
                )

        assert system is not None
        if normalized:
            if system:
                self.free_energies = {
                    site: 0 if site in k_by_site else np.log(dimension)
                    for site, dimension in system.dimensions.items()
                }
            else:
                self.free_energies = {site: 0 for site in k_by_site}
        else:
            f_locals = {
                site: -np.log((-l_op).expm().tr()) for site, l_op in k_by_site.items()
            }

            if system:
                self.free_energies = {
                    site: f_locals.get(site, np.log(dimension))
                    for site, dimension in system.dimensions.items()
                }
            else:
                self.free_energies = f_locals

            k_by_site = {
                site: local_k - f_locals[site] for site, local_k in k_by_site.items()
            }

        self.k_by_site = k_by_site

    def __mul__(self, operand):
        if isinstance(operand, (int, float)):
            if operand > 0:
                return GibbsProductDensityOperator(
                    self.k_by_site, self.prefactor * operand, self.system, True
                )
        return self.to_product_state() * operand

    def __neg__(self):
        return -self.to_product_state()

    def __rmul__(self, operand):
        if isinstance(operand, (int, float)):
            if operand > 0:
                return GibbsProductDensityOperator(
                    self.k_by_site, self.prefactor * operand, self.system, True
                )
        return operand * self.to_product_state()

    def acts_over(self) -> set:
        """
        Return a set with the names of the sites where
        the operator non-trivially acts over.
        """
        return set(site for site in self.k_by_site)

    def expect(self, obs: Union[Operator, Iterable]) -> Union[np.ndarray, dict, Number]:
        # TODO: write a better implementation
        if isinstance(obs, Operator):
            return (self.to_product_state()).expect(obs)
        return super().expect(obs)

    @property
    def isdiagonal(self) -> bool:
        for operator in self.k_by_site.values():
            if not is_diagonal_op(operator):
                return False
        return True

    def logm(self):
        terms = tuple(
            LocalOperator(site, -loc_op, self.system)
            for site, loc_op in self.k_by_site.items()
        )
        return OneBodyOperator(terms, self.system, False)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):

        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(
                (site for site in subsystem.sites if site in self.system.dimensions)
            )
        else:
            subsystem = self.system.subsystem(sites)

        k_by_site = self.k_by_site
        return GibbsProductDensityOperator(
            OneBodyOperator(
                tuple(
                    LocalOperator(site, k_by_site[site], subsystem)
                    for site in sites
                    if site in k_by_site
                ),
                subsystem,
            ),
            self.prefactor,
            subsystem,
            True,
        )

    def to_product_state(self):
        """Convert the operator in a productstate"""
        local_states = {
            site: (-local_k).expm() for site, local_k in self.k_by_site.items()
        }
        return ProductDensityOperator(
            local_states,
            self.prefactor,
            system=self.system,
            normalize=False,
        )

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        return self.to_product_state().to_qutip(block)
