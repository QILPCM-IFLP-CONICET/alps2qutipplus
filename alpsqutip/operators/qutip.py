# -*- coding: utf-8 -*-
"""
Qutip representation of an operator.
"""

from numbers import Number
from typing import Optional, Union

from numpy import log as np_log
from qutip import Qobj

from alpsqutip.alpsmodels import qutip_model_from_dims
from alpsqutip.geometry import GraphDescriptor
from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.basic import Operator, ScalarOperator, is_diagonal_op


class QutipOperator(Operator):
    """Represents a Qutip operator associated with a system"""

    def __init__(
        self,
        qoperator: Qobj,
        system: Optional[SystemDescriptor] = None,
        names=None,
        prefactor=1,
    ):
        assert isinstance(qoperator, Qobj), "qoperator should be a Qutip Operator"
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
        return "qutip interface operator for\n" + repr(self.operator)

    def dag(self):
        prefactor = self.prefactor
        operator = self.operator
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        else:
            if operator.isherm:
                return self
        return QutipOperator(operator.dag(), self.system, self.site_names, prefactor)

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

    @property
    def isdiagonal(self) -> bool:
        """Check if the operator is diagonal"""
        return is_diagonal_op(self.operator)

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

        if isinstance(op_ptrace, Qobj):
            return QutipOperator(
                op_ptrace,
                subsystem,
                names=new_site_names,
                prefactor=self.prefactor,
            )

        return ScalarOperator(self.prefactor * op_ptrace, subsystem)

    def tidyup(self, atol=None):
        """Removes small elements from the quantum object."""
        return QutipOperator(self.operator.tidyup(atol), self.system, self.prefactor)

    def to_qutip(self):
        return self.operator * self.prefactor

    def tr(self):
        return self.operator.tr() * self.prefactor


# #################################
# Arithmetic
# #################################


# Sum Qutip operators
@Operator.register_add_handler(
    (
        QutipOperator,
        QutipOperator,
    )
)
def sum_qutip_operator_plus_operator(x_op: QutipOperator, y_op: QutipOperator):
    """Sum two qutip operators"""
    names = x_op.site_names.copy()
    names.update(y_op.site_names)
    return QutipOperator(
        x_op.operator * x_op.prefactor + y_op.operator * y_op.prefactor,
        x_op.system or y_op.system,
        names=names,
        prefactor=1,
    )


@Operator.register_add_handler(
    (
        ScalarOperator,
        QutipOperator,
    )
)
def sum_scalarop_with_qutipop(x_op: ScalarOperator, y_op: QutipOperator):
    """Sum a Scalar operator to a Qutip Operator"""
    return QutipOperator(
        y_op.operator * y_op.prefactor + x_op.prefactor,
        names=y_op.site_names,
        prefactor=1,
    )


@Operator.register_add_handler(
    (
        QutipOperator,
        Number,
    )
)
@Operator.register_add_handler(
    (
        QutipOperator,
        Qobj,
    )
)
def sum_qutip_operator_plus_number(x_op: QutipOperator, y_val: Union[Number, Qobj]):
    """Sum an operator and a number  or a Qobj"""
    return QutipOperator(
        x_op.operator + y_val,
        x_op.system or y_val.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor,
    )


@Operator.register_mul_handler(
    (
        QutipOperator,
        QutipOperator,
    )
)
def mul_qutip_operator_qutip_operator(x_op: QutipOperator, y_op: QutipOperator):
    """Product of two qutip operators"""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    names = x_op.site_names.copy()
    names.update(y_op.site_names)
    return QutipOperator(
        x_op.operator * y_op.operator,
        system,
        names=names,
        prefactor=x_op.prefactor * y_op.prefactor,
    )


@Operator.register_mul_handler(
    (
        ScalarOperator,
        QutipOperator,
    )
)
def mul_scalarop_with_qutipop(x_op: ScalarOperator, y_op: QutipOperator):
    """Sum a Scalar operator to a Qutip Operator"""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return QutipOperator(
        y_op.operator,
        names=y_op.site_names,
        prefactor=x_op.prefactor * y_op.prefactor,
        system=system,
    )


@Operator.register_mul_handler(
    (
        QutipOperator,
        ScalarOperator,
    )
)
def mul_qutipop_with_scalarop(y_op: QutipOperator, x_op: ScalarOperator):
    """Sum a Scalar operator to a Qutip Operator"""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return QutipOperator(
        y_op.operator,
        names=y_op.site_names,
        prefactor=x_op.prefactor * y_op.prefactor,
        system=system,
    )


@Operator.register_mul_handler(
    (
        QutipOperator,
        Number,
    )
)
@Operator.register_mul_handler(
    (
        QutipOperator,
        float,
    )
)
@Operator.register_mul_handler(
    (
        QutipOperator,
        complex,
    )
)
def mul_qutip_operator_times_number(x_op: QutipOperator, y_val: Number):
    """product of a QutipOperator and a number."""
    return QutipOperator(
        x_op.operator,
        x_op.system or y_val.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor * y_val,
    )


@Operator.register_mul_handler(
    (
        Number,
        QutipOperator,
    )
)
@Operator.register_mul_handler(
    (
        float,
        QutipOperator,
    )
)
@Operator.register_mul_handler(
    (
        complex,
        QutipOperator,
    )
)
def mul_number_and_qutipoperator(y_val: Number, x_op: QutipOperator):
    """product of a number and a QutipOperator."""
    return QutipOperator(
        x_op.operator,
        x_op.system or y_val.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor * y_val,
    )


@Operator.register_mul_handler(
    (
        QutipOperator,
        Qobj,
    )
)
def mul_qutip_operator_times_qobj(x_op: QutipOperator, y_op: Qobj):
    """product of a QutipOperator and a Qobj."""
    return QutipOperator(
        x_op.operator * y_op,
        x_op.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor,
    )


@Operator.register_mul_handler(
    (
        Qobj,
        QutipOperator,
    )
)
def mul_qutip_obj_times_qutip_operator(y_op: Qobj, x_op: QutipOperator):
    """product of a Qobj and a QutipOperator."""
    system = x_op.system
    return QutipOperator(
        y_op * x_op.operator,
        system,
        names=x_op.site_names,
        prefactor=x_op.prefactor,
    )
