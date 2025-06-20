# -*- coding: utf-8 -*-
"""
Qutip representation of an operator.
"""

import logging
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

from numpy import imag, log as np_log
from qutip import Qobj, tensor  # type: ignore[import-untyped]

from alpsqutip.alpsmodels import qutip_model_from_dims
from alpsqutip.geometry import GraphDescriptor
from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
    empty_op,
    is_diagonal_op,
)
from alpsqutip.qutip_tools.tools import decompose_qutip_operator, scalar_value


class QutipOperator(Operator):
    """
    Represents a Qutip operator that acts over a block
    of sites of a system.

    If two QutipOperator are combined in an arithmetic
    operation, the result is QutipOperator acting on
    the union of both blocks.

    """

    system: SystemDescriptor
    operator: Qobj
    site_names: dict

    def __init__(
        self,
        qoperator: Qobj,
        system: Optional[SystemDescriptor] = None,
        names: Optional[Dict[str, int]] = None,
        prefactor=1,
    ):
        assert isinstance(
            qoperator, Qobj
        ), f"qoperator should be a Qutip Operator. Was {type(qoperator)}"
        if system is None:
            dims = qoperator.dims[0]
            model = qutip_model_from_dims(dims)
            if names is None:
                names = {f"qutip_{i}": i for i in range(len(dims))}
            sitebasis = model.site_basis
            sites = {s: sitebasis[f"qutip_{i}"] for i, s in enumerate(names)}

            graph = GraphDescriptor(
                "disconnected graph",
                {s: {"type": f"qutip_{i}"} for i, s in enumerate(sites)},
                {},
                {},
            )
            system = SystemDescriptor(graph, model, sites=sites)
        if names is None:
            names = {s: i for i, s in enumerate(system.sites)}

        self.system = system
        assert len(qoperator.dims[0]) == len(
            names
        ), f"{qoperator.dims[0]} and {names} have different lengths"
        assert all(pos < len(names) for pos in names.values())

        self.operator = qoperator
        self.site_names = names
        self.prefactor = prefactor

    def __neg__(self):
        return QutipOperator(
            self.operator,
            self.system,
            names=self.site_names,
            prefactor=-self.prefactor,
        )

    def __pow__(self, exponent):
        operator = self.operator
        if exponent < 0:
            operator = operator.inv()
            exponent = -exponent

        return QutipOperator(
            operator**exponent,
            system=self.system,
            names=self.site_names,
            prefactor=1 / self.prefactor**exponent,
        )

    def __repr__(self) -> str:
        return (
            f"qutip interface operator over sites {self.site_names} for {self.prefactor} x  \n"
            + repr(self.operator)
        )

    def acts_over(self) -> set:
        return frozenset(self.site_names.keys())

    def as_sum_of_products(self):
        """
        Decompose the operator as a
        sum of product operators
        """
        from alpsqutip.operators.arithmetic import SumOperator

        isherm = self.operator.isherm
        site_names = self.site_names
        sites = sorted(site_names, key=lambda x: site_names[x])
        decomposition = decompose_qutip_operator(self.operator.tidyup())
        prefactor = self.prefactor
        terms = tuple(
            (
                ProductOperator(
                    dict(zip(sites, term)),
                    prefactor=prefactor,
                    system=self.system,
                ).simplify()
                for term in decomposition
            )
        )
        if len(terms) == 0:
            terms = tuple((ScalarOperator(0, self.system),))
        return SumOperator(terms, self.system, isherm=isherm)

    def dag(self):
        prefactor = self.prefactor
        operator = self.operator
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        else:
            if operator.isherm:
                return self
        return QutipOperator(
            operator.dag(),
            system=self.system,
            names=self.site_names,
            prefactor=prefactor,
        )

    def eigenenergies(self):
        return self.operator.eigenenergies() * self.prefactor

    def eigenstates(self):
        evals, evecs = self.operator.eigenstates()
        return evals * self.prefactor, evecs

    def inv(self):
        """the inverse of the operator"""
        operator = self.operator
        return QutipOperator(
            operator.inv(),
            system=self.system,
            names=self.site_names,
            prefactor=1 / self.prefactor,
        )

    @property
    def isherm(self) -> bool:
        return self.operator.isherm and imag(self.prefactor) == 0.0

    @property
    def isdiagonal(self) -> bool:
        """Check if the operator is diagonal"""
        return is_diagonal_op(self.operator)

    @property
    def is_zero(self) -> bool:
        """Check if the matrix is zero"""
        return empty_op(self.operator)

    def logm(self):
        operator = self.operator
        evals, evecs = operator.eigenstates()
        evals = evals * self.prefactor
        evals[abs(evals) < 1.0e-50] = 1.0e-50
        if any(value < 0 for value in evals):
            evals = (1.0 + 0j) * evals
        log_op = sum(
            np_log(e_val) * e_vec * e_vec.dag() for e_val, e_vec in zip(evals, evecs)
        )
        return QutipOperator(log_op, self.system, self.site_names)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(site for site in subsystem.sites)
        else:
            subsystem = self.system.subsystem(sites)
            sites = frozenset(sites)

        if len(sites) == 0:
            return ScalarOperator(self.tr(), subsystem)

        prefactor = self.prefactor
        system = self.system
        dimensions = system.dimensions
        site_names = self.site_names
        partial_site_names = {
            site: pos for site, pos in site_names.items() if site in sites
        }
        keep = tuple(partial_site_names.values())
        if len(keep) == 0:
            # compute the trace of the block,
            # and multiply by the prefactor
            prefactor *= self.operator.tr()
            # Now, multiply by the dimensions not included in
            # sites or site_names
            dims_other = (
                dim
                for site, dim in dimensions.items()
                if site not in site_names and site not in sites
            )
            prefactor = reduce(lambda x, y: x * y, dims_other, prefactor)
            return ScalarOperator(prefactor, subsystem)

        new_qutip_op = self.operator.ptrace(keep)
        new_site_names = {
            site: i
            for i, site in enumerate(
                sorted(partial_site_names, key=lambda x: partial_site_names[x])
            )
        }
        other_dims = (
            dim
            for site, dim in dimensions.items()
            if (site not in sites and site not in site_names)
        )
        new_prefactor = reduce(lambda x, y: x * y, other_dims, self.prefactor)
        return QutipOperator(
            new_qutip_op,
            subsystem,
            names=new_site_names,
            prefactor=new_prefactor,
        )

    def simplify(self):
        """Simplify the operator"""
        names = self.site_names
        prefactor = self.prefactor
        qt_operator = self.operator
        system = self.system
        if prefactor == 0:
            return ScalarOperator(0.0, system)
        assert len(names) > 0

        # If is an empty op, return a ScalarOperator
        if empty_op(qt_operator):
            return ScalarOperator(0.0, self.system)

        if len(names) > 1:
            return self

        # The operator acts on a single site. Check if is an scalar
        s_val = scalar_value(qt_operator.data)
        if s_val is not None:
            return ScalarOperator(s_val * self.prefactor, self.system)
        # Otherwise, return a local operator:
        (site,) = names.keys()
        operator = self.operator.tidyup() * self.prefactor
        return LocalOperator(site, operator, self.system)

    def tidyup(self, atol=None):
        """Removes small elements from the quantum object."""
        return QutipOperator(
            self.operator.tidyup(atol),
            system=self.system,
            names=self.site_names,
            prefactor=self.prefactor,
        )

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        """
        Return a qutip operator representing the action over
        sites in block.
        By default (`block`=`None`), returns an operator
        acting over the full system, with sites sorted in
        lexicographical order.
        If `block`=`(,)` (the empty tuple), returns
        `self.operator`.
        """
        site_names_dict = self.site_names
        site_names = sorted(site_names_dict, key=lambda x: site_names_dict[x])
        system = self.system
        sites = system.sites
        operator_qutip: Qobj = self.operator * self.prefactor
        if block is None:
            if len(sites) > 8:
                logging.warning(
                    (
                        "to_qutip does not received a block. "
                        "Return an operator over the full system"
                    )
                )
            block = tuple(sorted(self.system.sites.keys()))

        if len(block) == 0 or list(block) == site_names:
            return operator_qutip

        # Look for sites in block that are not in site_names
        out_sites = tuple(
            (site for site in block if site not in site_names_dict and site in sites)
        )
        # Add identities and operators in block but not in site_names
        if out_sites:
            next_index: int = len(site_names)
            site_names_dict = site_names_dict.copy()
            site_names_dict.update(
                {site: next_index + i for i, site in enumerate(out_sites)}
            )
            extra_identities = (sites[site]["identity"] for site in out_sites)
            operator_qutip = tensor(operator_qutip, *extra_identities)

        # Add sites which are in site_names, but not in block
        block = block + tuple((site for site in site_names if site not in block))
        shuffle: List[int] = list(site_names_dict[site] for site in block)
        assert len(shuffle) == len(
            operator_qutip.dims[0]
        ), f"len({shuffle})!=len({operator_qutip.dims[0]})"
        if shuffle == sorted(shuffle):
            return operator_qutip
        return operator_qutip.permute(shuffle)

    def tr(self):
        prefactor = self.prefactor
        if prefactor == 0:
            return prefactor

        site_names: Dict[str, int] = self.site_names
        op_tr = self.operator.tr() if site_names else 0.0
        if op_tr == 0.0:
            return op_tr

        system: SystemDescriptor = self.system
        dimensions: Dict[str, int] = system.dimensions
        if len(site_names) < len(dimensions):
            names = set(site_names)
            dims_other = (dim for site, dim in dimensions.items() if site not in names)
            prefactor = reduce(lambda x, y: x * y, dims_other, self.prefactor)
        else:
            prefactor = self.prefactor
        result = op_tr * prefactor
        return result
