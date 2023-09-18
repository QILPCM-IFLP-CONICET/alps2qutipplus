# -*- coding: utf-8 -*-
"""
Qutip representation of an operator.
"""

from typing import Optional

from numbers import Number

from qutip import Qobj

from alpsqutip.alpsmodels import qutip_model_from_dims
from alpsqutip.geometry import GraphDescriptor
from alpsqutip.model import Operator, SystemDescriptor


class QutipOperator(Operator):
    """Represents a Qutip operator associated with a system"""

    def __init__(
        self,
        qoperator: Qobj,
        system: Optional[SystemDescriptor] = None,
        names=None,
        prefactor=1,
    ):
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
            )
            system = SystemDescriptor(graph, model, sites=sites)
        if names is None:
            names = {s: i for i, s in enumerate(system.sites)}

        self.system = system
        self.operator = qoperator
        self.site_names = names
        self.prefactor = prefactor

    def __add__(self, operand):
        if isinstance(operand, Operator):
            return QutipOperator(
                self.prefactor * self.operator + operand.to_qutip(),
                self.system,
                names=self.site_names,
            )
        if isinstance(operand, (int, float, complex, Qobj)):
            return QutipOperator(
                self.prefactor * self.operator + operand,
                self.system,
                names=self.site_names,
            )
        raise ValueError()

    def __mul__(self, operand):
        if isinstance(operand, Operator):
            operand = operand.to_qutip()

        if isinstance(operand, Qobj):
            return QutipOperator(
                self.operator * operand * self.prefactor,
                self.system,
                names=self.site_names,
            )
        if isinstance(operand, Number):
            return QutipOperator(
                self.operator,
                self.system,
                names=self.site_names,
                prefactor=self.prefactor * operand,
            )
        raise ValueError(
            f"type {type(operand)} cannot multiply a {type(self)}"
        )

    def __neg__(self):
        return QutipOperator(
            -self.operator, self.system, names=self.site_names
        )

    def __no_rmul__(self, operand):
        if isinstance(operand, Operator):
            operand = operand.to_qutip()

        if isinstance(operand, Qobj):
            return QutipOperator(
                operand * self.operator * self.prefactor,
                self.system,
                names=self.site_names,
            )
        if isinstance(operand, Number):
            return QutipOperator(
                self.operator,
                self.system,
                names=self.site_names,
                prefactor=self.prefactor * operand,
            )
        raise ValueError()

    def __pow__(self, exponent):
        operator = self.operator
        if exponent < 0:
            operator = operator.inv()
            exponent = -exponent

        return QutipOperator(
            operator**exponent,
            system=self.system,
            names=self.site_names,
            prefactor=1 / self.prefactor,
        )

    def dag(self):
        prefactor = self.prefactor
        operator = self.operator
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        else:
            if operator.isherm:
                return self
        return QutipOperator(
            operator.dag(), self.system, self.site_names, prefactor
        )

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
        return self.operator.isherm

    def partial_trace(self, sites: list):
        site_names = self.site_names
        sites = sorted(
            [s for s in self.site_names if s in sites],
            key=lambda s: site_names[s],
        )
        subsystem = self.system.subsystem(sites)
        site_indxs = [site_names[s] for s in sites]
        new_site_names = {s: i for i, s in enumerate(sites)}
        if site_indxs:
            op_ptrace = self.operator.ptrace(site_indxs)
        else:
            op_ptrace = self.operator.tr()

        return QutipOperator(
            op_ptrace,
            subsystem,
            names=new_site_names,
            prefactor=self.prefactor,
        )

    def to_qutip(self):
        return self.operator * self.prefactor

    def tr(self):
        return self.operator.tr() * self.prefactor
