"""
Density operator classes.
"""

from numbers import Number
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from qutip import (  # type: ignore[import-untyped]
    qeye as qutip_qeye,
    tensor as qutip_tensor,
)

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)


class DensityOperatorMixin:
    """
    DensityOperatorMixin is a Mixing class that
    contributes operator subclasses with the method
    `expect`.

    Notice that the `prefactor` attribute of these classes
    is only taken into account when density operators are combined
    into  a mixture by adding them, and when we do operations with
    positive numbers.

    In other operations, like multiplication with other operators,
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

    system: SystemDescriptor

    def eigenstates(self) -> list:
        if isinstance(self, Operator):
            return super().eigenstates()  # type:ignore[misc]
        raise NotImplementedError

    def expect(
        self, obs_objs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, Number]:
        """Compute the expectation value of an observable"""
        from alpsqutip.operators.quadratic import QuadraticFormOperator

        # TODO: expode that expectation values of operators just requires the
        # state where the operators acts.

        local_states = {None: self}

        def do_evaluate_expect(obs):
            """
            Inner function to evaluate expectation values. This method keeps
            track of the states of the subsystems required in the evaluation,
            which in typical cases is the most expensive part of the evaluation.
            """
            nonlocal local_states

            if isinstance(obs, dict):
                return {
                    name: do_evaluate_expect(operator) for name, operator in obs.items()
                }

            if isinstance(obs, (tuple, list)):
                return np.array([do_evaluate_expect(operator) for operator in obs])

            if isinstance(obs, QuadraticFormOperator):
                obs = obs.to_sum_operator()

            if isinstance(obs, SumOperator):
                return sum(do_evaluate_expect(term) for term in obs.terms)

            acts_over = obs.acts_over()
            if len(acts_over) == 0:
                if hasattr(obs, "prefactor"):
                    return obs.prefactor

            if acts_over not in local_states:
                local_states[acts_over] = self.partial_trace(acts_over)

            # if the argument matches with the argument of expect, it means that
            # we already try with the implementation of the subclasses. Then, let's rely
            # in the generic implementation: convert everything to qutip and evaluate
            # the trace:
            if obs_objs is obs:
                block = tuple(sorted(acts_over))
                return (
                    local_states[acts_over].to_qutip(block) * obs.to_qutip(block)
                ).tr()

            # If obs comes from an internal call, then try to use the specific method
            # of the subclass.
            return local_states[acts_over].expect(obs)

        return do_evaluate_expect(obs_objs)

    @property
    def isherm(self):
        return True

    def to_qutip_operator(self):
        from alpsqutip.operators.states import QutipDensityOperator

        block = tuple(sorted(self.system.sites))
        names = {name: pos for pos, name in enumerate(block)}
        rho_qutip = self.to_qutip(block)
        return QutipDensityOperator(
            rho_qutip, names=names, system=self.system, prefactor=1
        )

    def tr(self):
        return 1


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
        if weight == 0:
            local_states = {}
            local_zs = {}
        else:
            local_zs = {site: state.tr() for site, state in local_states.items()}
            if normalize:
                assert (z > 0 for z in local_zs.values())
                local_states = {
                    site: sigma / local_zs[site] for site, sigma in local_states.items()
                }

        # Complete the scalar factors using the system
        if system is None:
            dimensions = {
                site: operator.data.shape[0] for site, operator in local_states.items()
            }
            # TODO: build a system
        else:
            dimensions = system.dimensions
            local_identities: dict = {}
            for site, dimension in dimensions.items():
                if site not in local_states:
                    local_id = local_identities.get(dimension, None)
                    local_zs[site] = dimension
                    if local_id is None:
                        local_id = qutip_qeye(dimension) / dimension
                        local_identities[dimension] = local_id
                    local_states[site] = local_id

        super().__init__(local_states, prefactor=weight, system=system)
        self.local_fs = {site: -np.log(z) for site, z in local_zs.items()}

    def __mul__(self, a):
        if isinstance(a, float):
            if a >= 0:
                return ProductDensityOperator(
                    self.sites_op, self.prefactor * a, self.system, False
                )
        return super().__mul__(a)

    def __rmul__(self, a):
        if isinstance(a, float):
            if a >= 0:
                return ProductDensityOperator(
                    self.sites_op, self.prefactor * a, self.system, False
                )
        return super().__rmul__(a)

    def expect(self, obs: Union[Operator, Iterable]) -> Union[np.ndarray, dict, Number]:
        if isinstance(obs, LocalOperator):
            operator = obs.operator
            site = obs.site
            local_states = self.sites_op
            if site in local_states:
                return (local_states[site] * operator).tr()
            return operator.tr() / self.system.dimensions[site]

        if isinstance(obs, SumOperator):
            return sum(self.expect(term) for term in obs.terms)

        if isinstance(obs, ProductOperator):
            sites_obs = obs.sites_op
            local_states = self.sites_op
            dimensions = self.system.dimensions
            result = obs.prefactor

            for site, obs_op in sites_obs.items():
                if result == 0:
                    break
                if site in local_states:
                    result *= (local_states[site] * obs_op).tr()
                else:
                    result *= obs_op.tr() / dimensions[site]
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

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        sites_op = self.sites_op
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(sites.sites.keys())
        else:
            subsystem = self.system.subsystem(sites)

        local_states = {site: sites_op[site] for site in sites}

        return ProductDensityOperator(
            local_states, self.prefactor, subsystem, normalize=False
        )

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        prefactor = self.prefactor
        if prefactor == 0 or len(self.system.dimensions) == 0:
            return np.exp(-sum(np.log(dim) for dim in self.system.dimensions.values()))

        sites_op = self.sites_op
        dimensions = self.system.dimensions
        if block is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(sites_op) if site not in block)
            )

        return qutip_tensor(
            [
                (
                    sites_op[site]
                    if site in sites_op
                    else qutip_qeye(dimensions[site]) / dimensions[site]
                )
                for site in block
            ]
        )
