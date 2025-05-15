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
from alpsqutip.operators.states.utils import k_by_site_from_operator
from alpsqutip.qutip_tools.tools import safe_exp_and_normalize


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

    def expect(
        self, obs_objs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, Number]:
        return self.to_qutip_operator().expect(obs_objs)

    def logm(self):
        self.normalize()
        k = self.k
        return -k

    def normalize(self) -> Operator:
        """Normalize the operator in a way that exp(-K).tr()==1"""
        if not self.normalized:
            self.to_qutip(tuple())
        return self

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        return self.to_qutip_operator().partial_trace(sites)

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        system = self.system
        all_sites = tuple(system.sites)
        if block is None:
            block = tuple(sorted(all_sites))
        elif len(block) < len(all_sites):
            block = block + tuple(
                sorted((site for site in all_sites if site not in block))
            )

        if not self.normalized:
            rho_qutip, log_prefactor = safe_exp_and_normalize(-self.k.to_qutip())
            self.k = self.k + log_prefactor
            self.free_energy = -log_prefactor
            self.normalized = True
            if block == all_sites:
                return rho_qutip
            return rho_qutip.permute((all_sites.index(site) for site in block))

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
        system: Optional[SystemDescriptor] = None,
        prefactor: float = 1,
        normalized: bool = False,
    ):
        assert prefactor > 0.0

        self.prefactor = prefactor
        if isinstance(k, dict):
            assert system is not None
            self.system = system
            k_by_site = k
        else:
            k = k.simplify()
            if system:
                system = k.system.union(system)
            else:
                system = k.system
            self.system = system
            k_by_site = k_by_site_from_operator(k)

        if normalized:
            f_locals = {site: 0 for site in k_by_site}
        else:
            f_locals = {
                site: -np.log((-l_op).expm().tr()) for site, l_op in k_by_site.items()
            }
            for site in k_by_site:
                k_by_site[site] = k_by_site[site] - f_locals[site]

        # Add missing terms
        for site in system.sites:
            if site in k_by_site:
                continue
            f_local = np.log(system.dimensions[site])
            f_locals[site] = -f_local
            k_by_site[site] = system.site_identity(site) * f_local

        self.free_energies = f_locals
        self.k_by_site = k_by_site

    def __mul__(self, operand):
        if isinstance(operand, (int, float)):
            if operand > 0:
                return GibbsProductDensityOperator(
                    self.k_by_site, self.system, self.prefactor * operand,  True
                )
        return self.to_product_state() * operand

    def __neg__(self):
        return -self.to_product_state()

    def __rmul__(self, operand):
        if isinstance(operand, (int, float)):
            if operand > 0:
                return GibbsProductDensityOperator(
                    self.k_by_site, self.system, self.prefactor * operand,  True
                )
        return operand * self.to_product_state()

    def acts_over(self) -> set:
        """
        Return a set with the names of the sites where
        the operator non-trivially acts over.
        """
        return set(site for site in self.k_by_site)

    def expect(
        self, obs_objs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, Number]:
        return (self.to_product_state()).expect(obs_objs)

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

        k_by_site = {
            site: localstate
            for site, localstate in self.k_by_site.items()
            if site in sites
        }
        return GibbsProductDensityOperator(
            k_by_site,
            subsystem,
            self.prefactor,
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
            normalize=True,
        )

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        return self.to_product_state().to_qutip(block)
