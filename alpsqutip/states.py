"""
Density operator classes.
"""
from functools import reduce
from numbers import Number
from typing import Dict, Iterable, Optional, Union

import numpy as np
from qutip import Qobj, qeye as qutip_qeye, tensor as qutip_tensor

from alpsqutip.model import Operator, SystemDescriptor
from alpsqutip.operator_functions import eigenvalues
from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
    SumOperator,
)


def safe_exp_and_normalize(operator):
    """Compute `expm(operator)/Z` and `log(Z)`.
    `Z=expm(operator).tr()` in a safe way.
    """
    k_0 = max(abs(eigenvalues(operator, sparse=True, sort="high", eigvals=3)))
    op_exp = (operator - k_0).expm()
    op_exp_tr = op_exp.tr()
    op_exp = op_exp * (1.0 / op_exp_tr)
    return op_exp, np.log(op_exp_tr) + k_0


class DensityOperatorMixin:
    """
    DensityOperatorMixin is a Mixing class that
    contributes operator subclasses with the method
    `expect`.
    """

    def __add__(self, operand):
        if isinstance(operand, DensityOperatorMixin):
            return MixtureDensityOperator((self, operand,), self.system)
        return super().__add__(operand)

    def expect(
        self, obs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, Number]:
        """Compute the expectation value of an observable"""
        if isinstance(obs, dict):
            return {
                name: self.expect(operator) for name, operator in obs.items()
            }

        if isinstance(obs, (tuple, list)):
            return np.array([self.expect(operator) for operator in obs])

        return (self * obs).tr()

    @property
    def isherm(self) -> bool:
        return True


class QutipDensityOperator(QutipOperator, DensityOperatorMixin):
    """
    Qutip representation of a density operator
    """

    def __init__(
        self,
        qoperator: Qobj,
        system: Optional[SystemDescriptor] = None,
        names=None,
        prefactor=1,
    ):
        prefactor = prefactor * qoperator.tr()
        assert prefactor >= 0 and qoperator.isherm
        qoperator = qoperator / prefactor

        super().__init__(qoperator, system, names, prefactor)

    def __mul__(self, operand) -> Operator:
        if isinstance(operand, (int, float)):
            assert operand >= 0
            return QutipDensityOperator(
                self.operator,
                self.system,
                self.site_names,
                self.prefactor * operand,
            )

        return super().__mul__(operand)

    def __rmul__(self, operand) -> Operator:
        if isinstance(operand, (int, float)):
            assert operand >= 0
            return QutipDensityOperator(
                self.operator,
                self.system,
                self.site_names,
                self.prefactor * operand,
            )

        return super().__mul__(operand)


class ProductDensityOperator(ProductOperator, DensityOperatorMixin):
    """An uncorrelated density operator."""

    def __init__(
        self,
        local_states: dict,
        weight: float = 1.0,
        system: Optional[SystemDescriptor] = None,
        normalize: bool = True,
    ):
        assert weight >= 0
        sites = tuple(system.sites.keys() if system else local_states.keys())
        dimensions = system.dimensions
        local_zs = {
            site: (
                local_states[site].tr()
                if site in local_states
                else dimensions[site]
            )
            for site in sites
        }

        if normalize:
            assert (z > 0 for z in local_zs.values())
            local_states = {
                site: sigma / local_zs[site]
                for site, sigma in local_states.items()
            }
        # TODO: remove me
        self.tagged_scalar = True
        super().__init__(local_states, prefactor=weight, system=system)
        self.local_fs = {site: -np.log(z) for site, z in local_zs.items()}

    def expect(
        self, obs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, Number]:
        if isinstance(obs, LocalOperator):
            operator = obs.operator
            site = obs.site
            local_states = self.sites_op
            if site in local_states:
                return (local_states[site] * operator).tr()
            return operator.tr() / reduce(lambda x, y: x * y, operator.dims[0])
        if isinstance(obs, SumOperator):
            return sum(self.expect(term) for term in obs.terms)

        if isinstance(obs, ProductOperator):
            sites_obs = obs.sites_op
            local_states = self.sites_op
            result = obs.prefactor

            for site, obs_op in sites_obs.items():
                if result == 0:
                    break
                if site in local_states:
                    result *= (local_states[site] * obs_op).tr()
                else:
                    result *= obs_op.tr() / reduce(
                        (lambda x, y: x * y), obs_op.dims[0]
                    )
            return result
        return super().expect(obs)

    def partial_trace(self, sites: list):
        sites_op = self.sites_op
        sites_in = [site for site in sites if site in sites_op]
        local_states = {site: sites_op[site] for site in sites_in}
        subsystem = self.system.subsystem(sites_in)
        return ProductDensityOperator(
            local_states, self.prefactor, subsystem, normalize=False
        )

    def __add__(self, rho: Operator):
        system = self.system
        if isinstance(rho, float):
            sites_op = self.sites_op
            n_factors = len(sites_op)
            if n_factors == 0:
                return ProductDensityOperator(
                    {}, self.prefactor + rho, self.system, False
                )
            if n_factors == 1:
                site, sigma = next(sites_op.items())
                local_dim = (
                    1 / system.dimensions.get(site, 1) if system else 1.0
                )
                prefactor = self.prefactor
                new_prefactor = prefactor + rho / local_dim
                return ProductDensityOperator(
                    {
                        site: prefactor * sigma + rho / local_dim,
                    },
                    new_prefactor,
                    system,
                )
            return MixtureDensityOperator(
                [ProductDensityOperator({}, rho, self.system, False)]
            )

        if isinstance(rho, ProductDensityOperator):
            sites_op = self.sites_op
            n_factors = len(sites_op)
            other_sites_op = rho.sites_op
            other_n_factors = len(other_sites_op)
            if len(sites_op) == 0:
                return rho + self.prefactor
            if len(other_sites_op) == 0:
                return self + rho.prefactor

            if 1 == n_factors == other_n_factors:
                site, sigma = next(sites_op.items())
                other_sigma = other_sites_op.get(site, None)
                if other_sigma is not None:
                    a_float, b_float = self.prefactor, rho.prefactor
                    new_prefactor = a_float + b_float
                    a_float, b_float = (
                        a_float / new_prefactor,
                        b_float / new_prefactor,
                    )
                    new_sigma = a_float * sigma + b_float * other_sigma
                    return ProductDensityOperator(
                        {
                            site: new_sigma,
                        },
                        new_prefactor,
                        self.system,
                        False,
                    )
            return MixtureDensityOperator([self, rho])
        if isinstance(rho, MixtureDensityOperator):
            return MixtureDensityOperator(rho.terms + [self])
        return super().__add__(rho)

    def __mul__(self, a):
        if isinstance(a, float):
            if a > 0:
                return ProductDensityOperator(
                    self.sites_op, self.prefactor * a, self.system, False
                )

            if a == 0.0:
                return ProductDensityOperator({}, a, self.system, False)
        return super().__mul__(a)

    def __rmul__(self, a):
        if isinstance(a, float):
            if a > 0:
                return ProductDensityOperator(
                    self.sites_op, self.prefactor * a, self.system, False
                )

            if a == 0.0:
                return ProductDensityOperator({}, a, self.system, False)
        return super().__rmul__(a)

    def to_qutip(self):
        prefactor = self.prefactor
        if prefactor == 0 or len(self.system.dimensions) == 0:
            return prefactor
        ops = self.sites_op
        return prefactor * qutip_tensor(
            [
                ops[site] if site in ops else qutip_qeye(dim) / dim
                for site, dim in self.system.dimensions.items()
            ]
        )


class MixtureDensityOperator(SumOperator, DensityOperatorMixin):
    """
    A mixture of density operators
    """

    def __init__(self, terms: tuple, system: SystemDescriptor = None):
        assert (isinstance(t, ProductDensityOperator) for t in terms)
        super().__init__(terms, system)

    def expect(
        self, obs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, Number]:
        strip = False
        if isinstance(obs, Operator):
            strip = True
            obs = [obs]

        av_terms = tuple(term.expect(obs) for term in self.terms)
        if isinstance(obs, dict):
            return {
                op_name: sum(term[op_name] for term in av_terms)
                for op_name in obs
            }
        if strip:
            return sum(term for term in av_terms)[0]
        return sum(term for term in av_terms)

    def partial_trace(self, sites: list):
        return MixtureDensityOperator(
            tuple(t.partial_trace(sites) for t in self.terms), self.system
        )

    def __add__(self, rho: Operator):
        terms = self.terms
        system = self.system

        if isinstance(rho, MixtureDensityOperator):
            terms = terms + rho.terms
        elif isinstance(rho, DensityOperatorMixin):
            terms = terms + (rho,)
        elif isinstance(rho, (int, float)):
            terms = terms + (ProductDensityOperator({}, rho, system, False),)
        else:
            return super().__add__(rho)
        return MixtureDensityOperator(terms, system)

    def __mul__(self, a):
        if isinstance(a, float):
            return MixtureDensityOperator(
                tuple(
                    ProductDensityOperator(
                        t.sites_op, t.prefactor * a, t.system, False
                    )
                    for t in self.terms
                )
            )
        return super().__mul__(a)

    def to_qutip(self):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ProductOperator({}, 0, self.system).to_qutip()
        return sum(term.to_qutip() for term in self.terms)

    def tr(self) -> float:
        return sum(term.tr() for term in self.terms)


class GibbsDensityOperator(Operator, DensityOperatorMixin):
    """
    Stores an operator of the form rho= prefactor * exp(-K) / Tr(exp(-K)).

    """

    free_energy: float
    normalized: bool
    k: Operator

    def __init__(
        self,
        k: Operator,
        system: SystemDescriptor = None,
        prefactor=1.0,
        normalized=False,
    ):
        assert prefactor > 0
        self.k = k
        self.f_global = 0.0
        self.free_energy = 0.0
        self.prefactor = prefactor
        self.normalized = normalized
        self.system = system or k.system

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
        raise ValueError(
            "Division of an operator by ", type(operand), " not defined."
        )

    def expect(
        self, obs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, Number]:
        return self.to_qutip_operator().expect(obs)

    def normalize(self) -> Operator:
        """Normalize the operator in a way that exp(-K).tr()==1"""
        if not self.normalized:
            rho, log_prefactor = safe_exp_and_normalize(
                -self.k
            )  # pylint: disable=unused-variable
            self.k = self.k + log_prefactor
            self.free_energy = log_prefactor
            self.normalized = True

    def partial_trace(self, sites):
        return self.to_qutip_operator().partial_trace(sites)

    def to_qutip(self):
        if not self.normalized:
            rho, log_prefactor = safe_exp_and_normalize(-self.k)
            self.k = self.k + log_prefactor
            self.free_energy = log_prefactor
            self.normalized = True
            return self.prefactor * rho.to_qutip()
        result = (-self.k).to_qutip().expm() * self.prefactor
        return result

    def to_qutip_operator(self):
        rho_qutip = self.to_qutip()
        return QutipDensityOperator(rho_qutip, self.system, prefactor=1)


class GibbsProductDensityOperator(Operator, DensityOperatorMixin):
    """
    Stores an operator of the form
    rho = prefactor * \\otimes_i exp(-K_i)/Tr(exp(-K_i)).

    """

    k_by_site: list
    prefactor: float
    free_energies: Dict[str, float]

    isherm: bool = True

    def __init__(
        self,
        k: Union[Operator, dict],
        prefactor: float = 1,
        system: SystemDescriptor = None,
        normalized: bool = False,
    ):
        assert prefactor > 0.0
        self.prefactor = prefactor
        if isinstance(k, LocalOperator):
            self.system = system or k.system
            k_by_site = {k.site: k.operator}
        elif isinstance(k, OneBodyOperator):
            self.system = system or k.system
            k_by_site = {k_local.site: k_local.operator for k_local in k.terms}
        elif isinstance(k, dict):
            self.system = system
            k_by_site = k
        elif isinstance(k, Number):
            self.system = system
            site, dim = next(iter(system.dimensions.items()))
            operator = k * qutip_qeye(dim)
            k_by_site = {site: operator}
        else:
            raise ValueError(
                "ProductGibbsOperator cannot be initialized from a ", type(k)
            )

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
                site: np.log((-l_op).expm().tr())
                for site, l_op in k_by_site.items()
            }

            if system:
                self.free_energies = {
                    site: f_locals.get(site, np.log(dimension))
                    for site, dimension in system.dimensions.items()
                }
            else:
                self.free_energies = f_locals

            k_by_site = {
                site: local_k + f_locals[site]
                for site, local_k in k_by_site.items()
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

    def expect(
        self, obs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, Number]:
        # TODO: write a better implementation
        if isinstance(obs, Operator):
            return (self.to_product_state()).expect(obs) * self.prefactor
        return super().expect(obs)

    def partial_trace(self, sites):
        sites = [site for site in sites if site in self.system.dimensions]
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

    def to_qutip(self):
        return self.to_product_state().to_qutip()
