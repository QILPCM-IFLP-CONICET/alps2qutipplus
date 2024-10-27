"""
Density operator classes.
"""

from functools import reduce
from numbers import Number
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from qutip import Qobj, qeye as qutip_qeye, tensor as qutip_tensor

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
    check_multiplication,
    is_diagonal_op,
)
from alpsqutip.operators.functions import eigenvalues
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.utils import operator_to_wolfram


def safe_exp_and_normalize(operator):
    """Compute `expm(operator)/Z` and `log(Z)`.
    `Z=expm(operator).tr()` in a safe way.
    """
    k_0 = max(np.real(eigenvalues(operator, sparse=True, sort="high", eigvals=3)))
    op_exp = (operator - k_0).expm()
    op_exp_tr = op_exp.tr()
    op_exp = op_exp * (1.0 / op_exp_tr)
    return op_exp, np.log(op_exp_tr) + k_0


class DensityOperatorMixin:
    """
    DensityOperatorMixin is a Mixing class that
    contributes operator subclasses with the method
    `expect`.

    Notice that the `prefactor` attribute of these classes
    is only taken into account when density operators are combined
    into  a mixture by adding them, and when we do operations with
    positive numbers.

    In other operations, like mutiplication with other operators,
    density operators are handled as positive operators of trace 1.

    So, for example,
    ```
    rho = .3 * ProductDensityOperator({"site1": qutip.qeye(2) + qutip.sigmax(), "site2": qutip.qeye(2)})
    ```
    acts under operations with other operators like
    ```
    rho = ProductOperator({"site1": .5*(qutip.qeye(2) + qutip.sigmax()), "site2": .5*qutip.qeye(2)})
    ```

    If now we introduce a `sigma` operator
    ```
    sigma = .7 ProductDensityOperator({"site1": qutip.qeye(2), "site2": qutip.qeye(2) + qutip.sigmax()})
    ```
    the mixture
    ```
    mix = rho + sigma
    ```

    and another operator `A`
    ```
    A=ProductOperator({"site1": qutip.sigmax(), "site2": qutip.sigmax()})
    ```

    we obtain the equality
    ```
    (mix * A).tr() == .3 * (A * rho).tr() + .7 * (A* sigma)
    ```

    Notice that algebraic operations does not check if the prefactors of all the terms adds to 1.
    To be sure about the normalization, use the method `expect`:

    ```
    mix.expect(A)== (mix * A).tr()/sum([t.prefactor for t in A.terms])
    ```



    """

    def expect(self, obs: Union[Operator, Iterable]) -> Union[np.ndarray, dict, Number]:
        """Compute the expectation value of an observable"""
        if isinstance(obs, dict):
            return {name: self.expect(operator) for name, operator in obs.items()}

        if isinstance(obs, (tuple, list)):
            return np.array([self.expect(operator) for operator in obs])

        return (self * obs).tr() / self.prefactor

    @property
    def isherm(self):
        return True

    def to_qutip_operator(self):
        rho_qutip = self.to_qutip()
        return QutipDensityOperator(rho_qutip, self.system, prefactor=1)

    def tr(self):
        return 1


class QutipDensityOperator(DensityOperatorMixin, QutipOperator):
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
        super().__init__(qoperator, system, names, prefactor)

    def __add__(self, operand) -> Operator:
        if isinstance(operand, (int, float)):
            assert operand >= 0
            return QutipDensityOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )

        op_qo = operand.to_qutip()
        if isinstance(operand, DensityOperatorMixin):
            op_qo = op_qo * self.prefactor
            return QutipDensityOperator(op_qo, self.system or op_qo.system)
        return QutipOperator(op_qo, self.system or op_qo.system)

    def __mul__(self, operand) -> Operator:
        if isinstance(operand, (int, float)):
            assert operand >= 0
            return QutipDensityOperator(
                self.operator,
                self.system,
                self.site_names,
                self.prefactor * operand,
            )
        op_qo = operand.to_qutip()
        return QutipOperator(self.operator * op_qo, self.system or op_qo.system)

    def __radd__(self, operand) -> Operator:
        if isinstance(operand, (int, float)):
            assert operand >= 0
            return QutipDensityOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )

        op_qo = operand.to_qutip()
        if isinstance(operand, DensityOperatorMixin):
            op_qo = op_qo * self.prefactor
            return QutipDensityOperator(op_qo, self.system or op_qo.system)
        return QutipOperator(op_qo, self.system or op_qo.system)

    def __rmul__(self, operand) -> Operator:
        if isinstance(operand, (int, float)):
            assert operand >= 0
            return QutipDensityOperator(
                self.operator,
                self.system,
                self.site_names,
                self.prefactor * operand,
            )
        op_qo = operand.to_qutip()
        return QutipOperator(op_qo * self.operator, self.system or op_qo.system)

    def logm(self):
        operator = self.operator
        evals, evecs = operator.eigenstates()
        evals[abs(evals) < 1.0e-30] = 1.0e-30
        log_op = sum(
            np.log(e_val) * e_vec * e_vec.dag() for e_val, e_vec in zip(evals, evecs)
        )
        return QutipOperator(log_op, self.system, self.site_names)


class ProductDensityOperator(DensityOperatorMixin, ProductOperator):
    """An uncorrelated density operator."""

    def __init__(
        self,
        local_states: dict,
        weight: float = 1.0,
        system: Optional[SystemDescriptor] = None,
        normalize: bool = True,
    ):
        assert weight >= 0

        # Build the local partition functions and normalize
        # if required
        local_zs = {site: state.tr() for site, state in local_states.items()}
        if normalize:
            assert (z > 0 for z in local_zs.values())
            local_states = {
                site: sigma / local_zs[site] for site, sigma in local_states.items()
            }

        # Complete the scalar factors using the system
        if system is None:
            sites = tuple(local_states.keys())
            dimensions = {
                site: operator.data.shape[0] for site, operator in local_states.items()
            }
            # TODO: build a system
        else:
            sites = tuple(system.sites.keys())
            dimensions = system.dimensions
            local_identities = {}
            for site, dimension in dimensions.items():
                if site not in local_states:
                    local_id = local_identities.get(dimension, None)
                    local_zs[site] = dimension
                    if local_id is None:
                        local_id = qutip_qeye(dimension) / dimension
                        local_identities[dimension] = local_id
                    local_states[site] = local_id

        # TODO: remove me
        self.tagged_scalar = True
        super().__init__(local_states, prefactor=weight, system=system)
        self.local_fs = {site: -np.log(z) for site, z in local_zs.items()}

    def __add__(self, operand):
        if isinstance(operand, MixtureDensityOperator):
            return operand + self

        if isinstance(operand, DensityOperatorMixin):
            return MixtureDensityOperator(
                (
                    self,
                    operand,
                ),
                self.system,
            )
        return super().__add__(operand)

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

    def expect(self, obs: Union[Operator, Iterable]) -> Union[np.ndarray, dict, Number]:
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
                    result *= obs_op.tr() / reduce((lambda x, y: x * y), obs_op.dims[0])
            return result
        return super().expect(obs)

    def logm(self):
        def log_qutip(loc_op):
            evals, evecs = loc_op.eigenstates()
            evals[abs(evals) < 1.0e-30] = 1.0e-30
            return sum(
                np.log(e_val) * e_vec * e_vec.dag()
                for e_val, e_vec in zip(evals, evecs)
            )

        system = self.system
        sites_op = self.sites_op
        terms = tuple(
            LocalOperator(site, log_qutip(loc_op), system)
            for site, loc_op in sites_op.items()
        )
        if system:
            norm = -sum(
                np.log(dim)
                for site, dim in system.dimensions.items()
                if site not in self.sites_op
            )
            return OneBodyOperator(terms, system, False) + ScalarOperator(norm, system)
        return OneBodyOperator(terms, system, False)

    def partial_trace(self, sites: Union[list, SystemDescriptor]):
        sites_op = self.sites_op
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = tuple(sites.sites.keys())
        else:
            subsystem = self.system.subsystem(sites)

        sites_in = [site for site in sites if site in sites_op]
        local_states = {site: sites_op[site] for site in sites}

        return ProductDensityOperator(
            local_states, self.prefactor, subsystem, normalize=False
        )

    def to_qutip(self):
        prefactor = self.prefactor
        if prefactor == 0 or len(self.system.dimensions) == 0:
            return np.exp(-sum(np.log(dim) for dim in self.system.dimensions.values()))
        ops = self.sites_op
        return qutip_tensor(
            [
                ops[site] if site in ops else qutip_qeye(dim) / dim
                for site, dim in self.system.dimensions.items()
            ]
        )


class MixtureDensityOperator(DensityOperatorMixin, SumOperator):
    """
    A mixture of density operators
    """

    def __init__(self, terms: tuple, system: SystemDescriptor = None):
        super().__init__(terms, system, True)

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
            return MixtureDensityOperator(tuple(term * a for term in self.terms))
        return super().__mul__(a)

    def __rmul__(self, a):
        if isinstance(a, float) and a >= 0:
            return MixtureDensityOperator(
                tuple(term * a for term in self.terms), self.system
            )
        return super().__rmul__(a)

    def acts_over(self) -> set:
        """
        Return a set with the name of the
        sites where the operator nontrivially acts
        """
        sites = set()
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

    def partial_trace(self, sites: Union[list, SystemDescriptor]):
        new_terms = tuple(t.partial_trace(sites) for t in self.terms)
        subsystem = new_terms[0].system
        return MixtureDensityOperator(new_terms, subsystem)

    def to_qutip(self):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip()

        # TODO: find a more efficient way to avoid element-wise
        # multiplications
        return sum(term.to_qutip() * term.prefactor for term in self.terms)


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
            self.k = self.k + log_prefactor
            self.free_energy = -log_prefactor
            self.normalized = True

    def partial_trace(self, sites: Union[List, SystemDescriptor]):
        return self.to_qutip_operator().partial_trace(sites)

    def to_qutip(self):
        if not self.normalized:
            rho, log_prefactor = safe_exp_and_normalize(-self.k)
            self.k = self.k + log_prefactor
            self.free_energy = -log_prefactor
            self.normalized = True
            return rho.to_qutip()
        result = (-self.k).to_qutip().expm()
        return result


class GibbsProductDensityOperator(DensityOperatorMixin, Operator):
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
            k_by_site = {
                k_local.site: k_local.operator
                for k_local in k.terms
                if isinstance(k_local, LocalOperator)
            }
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


# ####################################
#  Arithmetic
# ####################################


##### Sums #############


@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        MixtureDensityOperator,
    )
)
def sum_two_mixture_operators(
    x_op: MixtureDensityOperator, y_op: MixtureDensityOperator
):
    terms = x_op.terms + y_op.terms
    # If there is just one term, return it:
    if len(terms) == 1:
        return terms[0]

    # For empty terms, return 0
    system = x_op.system or y_op.system
    if len(terms) == 0:
        return ScalarOperator(0.0, system)
    # General case
    return MixtureDensityOperator(terms, system)


@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        DensityOperatorMixin,
    )
)
@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        GibbsDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        GibbsProductDensityOperator,
    )
)
def sum_mixture_and_density_operators(
    x_op: MixtureDensityOperator, y_op: DensityOperatorMixin
):
    terms = x_op.terms + (y_op,)
    # If there is just one term, return it:
    if len(terms) == 1:
        return terms[0]

    # For empty terms, return 0
    system = x_op.system or y_op.system
    if len(terms) == 0:
        return ScalarOperator(0.0, system)
    # General case
    return MixtureDensityOperator(terms, system)


@Operator.register_add_handler(
    (
        DensityOperatorMixin,
        DensityOperatorMixin,
    )
)
@Operator.register_add_handler(
    (
        DensityOperatorMixin,
        GibbsDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        DensityOperatorMixin,
        GibbsProductDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        GibbsDensityOperator,
        GibbsDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        GibbsDensityOperator,
        GibbsProductDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        GibbsProductDensityOperator,
        GibbsProductDensityOperator,
    )
)
def _(x_op, y_op):
    return MixtureDensityOperator(
        (
            x_op,
            y_op,
        ),
        x_op.system or y_op.system,
    )


@Operator.register_add_handler(
    (
        GibbsProductDensityOperator,
        ProductOperator,
    )
)
def _(x_op, y_op):
    return SumOperator(
        (
            x_op,
            y_op,
        ),
        x_op.system or y_op.system,
    )


##### Products #############


# ProductDensityOperator times ProductDensityOperator
@Operator.register_mul_handler((ProductDensityOperator, ProductDensityOperator))
def _(x_op: ProductDensityOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    sites_op = x_op.sites_op.copy()
    for site, factor in y_op.sites_op.items():
        if site in sites_op:
            sites_op[site] *= factor
        else:
            sites_op[site] = factor
    return ProductOperator(sites_op, 1, system)


# ProductDensityOperator times Operators

####   ScalarOperator


@Operator.register_mul_handler((ScalarOperator, ProductDensityOperator))
def _(x_op: ScalarOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    result = ProductOperator(y_op.sites_op, x_op.prefactor, system)
    return result


@Operator.register_mul_handler((ProductDensityOperator, ScalarOperator))
def _(x_op: ProductDensityOperator, y_op: ScalarOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    prefactor = y_op.prefactor
    # prefactor = prefactor * np.exp(sum(x_op.local_fs.values()))
    return ProductOperator(x_op.sites_op, y_op.prefactor, system)


####   LocalOperator


@Operator.register_mul_handler((LocalOperator, ProductDensityOperator))
def _(x_op: LocalOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    prefactor = x_op.prefactor
    sites_op = y_op.sites_op.copy()
    site = x_op.site
    if site in sites_op:
        sites_op[site] = x_op.operator * sites_op[site]
    else:
        sites_op[site] = x_op.operator

    return ProductOperator(sites_op, prefactor, system)


@Operator.register_mul_handler((ProductDensityOperator, LocalOperator))
def _(x_op: ProductDensityOperator, y_op: LocalOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    sites_op = x_op.sites_op.copy()
    site = y_op.site
    if site in sites_op:
        sites_op[site] *= y_op.operator
    else:
        sites_op[site] = y_op.operator
    return ProductOperator(sites_op, prefactor=1, system=system)


# ProductOperator


@Operator.register_mul_handler((ProductOperator, ProductDensityOperator))
def _(x_op: ProductOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    prefactor = x_op.prefactor
    sites_op = x_op.sites_op.copy()
    for site, factor in y_op.sites_op.items():
        if site in sites_op:
            sites_op[site] *= factor
        else:
            sites_op[site] = factor

    return ProductOperator(sites_op, prefactor, system)


@Operator.register_mul_handler((ProductDensityOperator, ProductOperator))
def _(x_op: ProductDensityOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    sites_op = x_op.sites_op.copy()
    for site, factor in y_op.sites_op.items():
        if site in sites_op:
            sites_op[site] *= factor
        else:
            sites_op[site] = factor

    return ProductOperator(sites_op, y_op.prefactor, system)


# SumOperators


@Operator.register_mul_handler(
    (
        ProductDensityOperator,
        SumOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductDensityOperator,
        OneBodyOperator,
    )
)
def _(x_op: ProductDensityOperator, y_op: SumOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return SumOperator(
        tuple(x_op * term for term in y_op.terms),
        system,
    )


@Operator.register_mul_handler(
    (
        OneBodyOperator,
        ProductDensityOperator,
    )
)
@Operator.register_mul_handler(
    (
        SumOperator,
        ProductDensityOperator,
    )
)
def _(x_op: SumOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = tuple(term * y_op for term in x_op.terms)
    return SumOperator(
        terms,
        system,
    )


#############    Mixtures  ##########################


# Anything that is not a SumOperator


@Operator.register_mul_handler(
    (
        Operator,
        MixtureDensityOperator,
    )
)
@Operator.register_mul_handler(
    (
        ScalarOperator,
        MixtureDensityOperator,
    )
)
@Operator.register_mul_handler(
    (
        LocalOperator,
        MixtureDensityOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductOperator,
        MixtureDensityOperator,
    )
)
@Operator.register_mul_handler((ProductDensityOperator, MixtureDensityOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, MixtureDensityOperator))
@Operator.register_mul_handler((QutipOperator, MixtureDensityOperator))
def _(x_op: Operator, y_op: MixtureDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = tuple((x_op * term) * term.prefactor for term in y_op.terms)
    result = SumOperator(terms, system)
    return result


@Operator.register_mul_handler((MixtureDensityOperator, Operator))
@Operator.register_mul_handler((MixtureDensityOperator, ScalarOperator))
@Operator.register_mul_handler((MixtureDensityOperator, LocalOperator))
@Operator.register_mul_handler((MixtureDensityOperator, ProductOperator))
@Operator.register_mul_handler((MixtureDensityOperator, ProductDensityOperator))
@Operator.register_mul_handler((MixtureDensityOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((MixtureDensityOperator, QutipOperator))
def _(
    x_op: MixtureDensityOperator,
    y_op: Operator,
):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = []
    for term in x_op.terms:
        prefactor = term.prefactor
        if prefactor == 0:
            continue
        new_term = (term * y_op) * prefactor
        terms.append(new_term)

    return SumOperator(tuple(terms), system)


# MixtureDensityOperator times SumOperators
# and its derivatives


@Operator.register_mul_handler((MixtureDensityOperator, MixtureDensityOperator))
@Operator.register_mul_handler((MixtureDensityOperator, OneBodyOperator))
@Operator.register_mul_handler((MixtureDensityOperator, SumOperator))
def _(
    x_op: MixtureDensityOperator,
    y_op: Operator,
):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = []
    for term in x_op.terms:
        prefactor = term.prefactor
        if prefactor == 0:
            continue
        new_term = term * y_op
        new_term = new_term * prefactor
        terms.append(new_term)

    result = SumOperator(tuple(terms), system)
    return result


@Operator.register_mul_handler((OneBodyOperator, MixtureDensityOperator))
@Operator.register_mul_handler((SumOperator, MixtureDensityOperator))
def _(
    x_op: SumOperator,
    y_op: MixtureDensityOperator,
):

    terms = []
    for term in y_op.terms:
        prefactor = term.prefactor
        if prefactor == 0:
            continue
        new_term = x_op * term
        new_term = new_term * prefactor
        terms.append(new_term)

    result = SumOperator(tuple(terms), x_op.system or y_op.system)
    return result


#########  GibbsDensityOperators


@Operator.register_mul_handler((ScalarOperator, GibbsDensityOperator))
def _(x_op: ScalarOperator, y_op: GibbsDensityOperator):

    y_qutip = y_op.to_qutip()
    result = QutipOperator(x_op.prefactor * y_qutip, x_op.system or y_op.system)
    return result


@Operator.register_mul_handler((GibbsDensityOperator, ScalarOperator))
def _(x_op: GibbsDensityOperator, y_op: ScalarOperator):
    x_qutip = x_op.to_qutip()
    result = QutipOperator(y_op.prefactor * y_qutip, x_op.system or y_op.system)
    return result


# GibbsDensityOperators times any other operator


@Operator.register_mul_handler((GibbsDensityOperator, GibbsDensityOperator))
@Operator.register_mul_handler((LocalOperator, GibbsDensityOperator))
@Operator.register_mul_handler((ProductOperator, GibbsDensityOperator))
@Operator.register_mul_handler((OneBodyOperator, GibbsDensityOperator))
@Operator.register_mul_handler((ProductDensityOperator, GibbsDensityOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, GibbsDensityOperator))
@Operator.register_mul_handler((SumOperator, GibbsDensityOperator))
@Operator.register_mul_handler((OneBodyOperator, GibbsDensityOperator))
@Operator.register_mul_handler((MixtureDensityOperator, GibbsDensityOperator))
### and backward
@Operator.register_mul_handler((GibbsDensityOperator, LocalOperator))
@Operator.register_mul_handler((GibbsDensityOperator, ProductOperator))
@Operator.register_mul_handler((GibbsDensityOperator, OneBodyOperator))
@Operator.register_mul_handler((GibbsDensityOperator, ProductDensityOperator))
@Operator.register_mul_handler((GibbsDensityOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((GibbsDensityOperator, SumOperator))
@Operator.register_mul_handler((GibbsDensityOperator, OneBodyOperator))
@Operator.register_mul_handler((GibbsDensityOperator, MixtureDensityOperator))
def _(
    x_op: Operator,
    y_op: Operator,
):
    return x_op.to_qutip_operator() * y_op.to_qutip_operator()


#############################
#  GibbsProductOperators
#############################


@Operator.register_mul_handler(
    (GibbsProductDensityOperator, GibbsProductDensityOperator)
)
def _(x_op: GibbsProductDensityOperator, y_op: GibbsProductDensityOperator):
    return x_op.to_product_state() * y_op.to_product_state()


# times ScalarOperator, LocalOperator, ProductOperator
@Operator.register_mul_handler((GibbsProductDensityOperator, ScalarOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, LocalOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, ProductOperator))
def _(x_op: GibbsProductDensityOperator, y_op: Operator):
    return x_op.to_product_state() * y_op


@Operator.register_mul_handler((ScalarOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((LocalOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((ProductOperator, GibbsProductDensityOperator))
def _(x_op: Operator, y_op: GibbsProductDensityOperator):
    y_prod = y_op.to_product_state()
    return x_op * y_prod
