"""
Classes to represent density operators as Gibbs states $rho=e^{-k}$.

"""

from numbers import Number
from typing import Dict, Iterable, Optional, Tuple, Union, cast

import numpy as np

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ScalarOperator,
    is_diagonal_op,
)
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

    _free_energy: float
    normalized: bool
    k: Operator

    def __init__(
        self,
        k: Operator,
        system: Optional[SystemDescriptor] = None,
        prefactor=1.0,
        normalized=False,
        meanfield=None,
    ):
        if prefactor == 0:
            self.k = ScalarOperator(0, k.system)
            self.f_global = 0.0
            self._free_energy = 0.0
            self.normalized = normalized
            self.prefactor = 0
            self.normalized = normalized
            self.system = system if system is not None else k.system
            self._meanfield = ProductDensityOperator({}, system=self.system, weight=0)
            return

        assert prefactor > 0
        self.k = k
        self.f_global = 0.0
        self._free_energy = 0.0
        self.prefactor = prefactor
        self.normalized = normalized
        self.system = system if system is not None else k.system
        self._meanfield = meanfield

    def __mul__(self, operand):
        if isinstance(operand, (int, float, np.float64)) and operand >= 0:
            return GibbsDensityOperator(
                self.k,
                self.system,
                self.prefactor * operand,
                normalized=self.normalized,
            )
        return self.to_qutip_operator() * operand

    def __neg__(self):
        return -(self.to_qutip_operator())

    def __rmul__(self, operand):
        if isinstance(operand, (int, float, np.float64)) and operand >= 0.0:
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

    def acts_over(self) -> Optional[frozenset]:
        """
        Return a set with the name of the
        sites where the operator nontrivially acts
        """
        return self.k.acts_over()

    # def expect(
    #    self, obs_objs: Union[Operator, Iterable]
    # ) -> Union[np.ndarray, dict, Number]:
    #    return self.to_qutip_operator().expect(obs_objs)

    @property
    def free_energy(self):
        """compute the free energy"""
        if not self.normalized:
            self.normalize()
        return self._free_energy

    @free_energy.setter
    def free_energy(self, value):
        """set the free energy"""
        self._free_energy = value
        return self._free_energy

    def logm(self):
        self.normalize()
        k = self.k
        return -k

    def normalize(self) -> Operator:
        """Normalize the operator in a way that exp(-K).tr()==1"""
        if not self.normalized:
            self.to_qutip(cast(Tuple[str], tuple()))

        return self

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        from alpsqutip.operators.states.meanfield.gibbs_partial_trace import (
            gibbs_meanfield_partial_trace,
        )

        if isinstance(sites, SystemDescriptor):
            sites = frozenset(sites.sites)
        return gibbs_meanfield_partial_trace(self, sites)

    def to_qutip_operator(self):
        from alpsqutip.operators.states import QutipDensityOperator

        block = tuple(sorted(self.system.sites))
        names = {name: pos for pos, name in enumerate(block)}
        rho_qutip = self.to_qutip(block)
        prefactor = getattr(self, "prefactor", 1.0)
        return QutipDensityOperator(
            rho_qutip, names=names, system=self.system, prefactor=prefactor
        )

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        system = self.system
        all_sites = tuple(system.sites)
        if block is None:
            block = tuple(sorted(all_sites))
        elif len(block) < len(all_sites):
            assert all(
                site in all_sites for site in block
            ), "sites must be in the (sub)system"
            block = block + tuple(
                sorted((site for site in all_sites if site not in block))
            )

        if self.normalized:
            result = (-self.k).to_qutip(block).expm()
        else:
            result, log_prefactor = safe_exp_and_normalize(-self.k.to_qutip(block))
            self.k = self.k + log_prefactor
            self._free_energy = -log_prefactor
            self.normalized = True
            # result = result.permute(tuple((all_sites.index(site) for site in block)))

        result = result.ptrace(tuple(range(len(block))))

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
        self_system: SystemDescriptor
        k_by_site: Dict[str, Operator]
        f_locals: Dict[str, float]

        assert prefactor > 0.0

        self.prefactor = prefactor
        if isinstance(k, dict):
            assert system is not None
            self_system = self.system = cast(SystemDescriptor, system)
            k_by_site = k
        else:
            k_operator: Operator = cast(Operator, k)
            k_operator = k_operator.simplify()
            if system:
                system = k_operator.system.union(system)
            else:
                system = k_operator.system
            self_system = self.system = cast(SystemDescriptor, system)
            k_by_site = k_by_site_from_operator(k_operator)

        if normalized:
            f_locals = {site: 0.0 for site in k_by_site}
        else:

            def safe_local_f(op_loc):
                spectrum = -(op_loc.eigenenergies())
                f0 = max(spectrum)
                spectrum = spectrum - f0
                return -np.log(sum(np.exp(spectrum))) - f0

            f_locals = {site: safe_local_f(l_op) for site, l_op in k_by_site.items()}

            for site in k_by_site:
                k_by_site[site] = k_by_site[site] - f_locals[site]

        # Add missing terms
        for site in self_system.sites:
            if site in k_by_site:
                continue
            f_local = np.log(self_system.dimensions[site])
            f_locals[site] = -f_local
            k_by_site[site] = self_system.site_identity(site) * f_local

        # for site, op_qutip in k_by_site.items():
        #     eig_vals = op_qutip.eigenenergies()
        #     probs = np.exp(-eig_vals)
        #     assert abs(sum(probs) - 1) < 1e-8, f"{probs} from {eig_vals}"
        #     assert all(p >= 0 for p in probs)

        self.free_energies = f_locals
        self.k_by_site = k_by_site

    def __mul__(self, operand):
        if isinstance(operand, (int, float)):
            if operand > 0:
                return GibbsProductDensityOperator(
                    self.k_by_site, self.system, self.prefactor * operand, True
                )
        return self.to_product_state() * operand

    def __neg__(self):
        return -(self.to_product_state())

    def __rmul__(self, operand):
        if isinstance(operand, (int, float)):
            if operand > 0:
                return GibbsProductDensityOperator(
                    self.k_by_site, self.system, self.prefactor * operand, True
                )
        return operand * self.to_product_state()

    def acts_over(self) -> Optional[frozenset]:
        """
        Return a set with the names of the sites where
        the operator non-trivially acts over.
        """
        return frozenset(site for site in self.k_by_site)

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
