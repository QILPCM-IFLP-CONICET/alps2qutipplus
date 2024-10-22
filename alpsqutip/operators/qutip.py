# -*- coding: utf-8 -*-
"""
Qutip representation of an operator.
"""

from numbers import Number
from typing import List, Optional, Union, Tuple

from numpy import log as np_log, zeros as np_zeros
from scipy.linalg import svd
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


def reshape_qutip_data(data, dims):
    """
    reshape the data representing an operator with dimensions
    dims = [[dim1, dim2,...],[dim1, dim2,...]]
    as an array with shape
    dims' = [[dim1,dim1],[dim2,dim3,... dim2,dim3,...]]
    """
    data_type = data.data.dtype
    dim_1 = dims[0]
    dim_2 = int(data.shape[0]/dim_1)
    new_data = np_zeros((dim_1**2, dim_2**2,), dtype=data_type)
    # reshape the operator
    # TODO: see to exploit the sparse structure of data to build the matrix
    for i_idx in range(dim_1):
        for j_idx in range(dim_1):
            for k_idx in range(dim_2):
                for l_idx in range(dim_2):
                    alpha = dim_1*i_idx+j_idx
                    beta = dim_2*k_idx+l_idx
                    gamma = dim_2*i_idx+k_idx
                    delta = dim_2*j_idx+l_idx
                    new_data[alpha, beta] = data[gamma, delta]
    return new_data


def factorize_qutip_operator(operator: Qobj) -> List[Tuple]:
    """
    Decompose a qutip operator q123... into a sum
    of tensor products sum_{ka, kb, kc...} q1^{ka} q2^{kakb} q3^{kakbkc}...
    return a list of tuples, with each factor.
    """
    dims = operator.dims[0]
    if len(dims) < 2:
        return [(operator,)]
    data = operator.data
    dim_1 = dims[0]
    dim_2 = int(data.shape[0]/dim_1)
    dims_1 = [[dim_1], [dim_1]]
    shape_1 = [dim_1, dim_1]
    dims_2 = [dims[1:], dims[1:]]
    shape_2 = [dim_2, dim_2]
    u_mat, s_mat, vh_mat = svd(reshape_qutip_data(data, dims),
                               full_matrices=False,
                               overwrite_a=True)
    ops_1 = [Qobj(s*u_mat[:, i].reshape(dim_1, dim_1),
                  dims_1, shape_1, copy=False)
             for i, s in enumerate(s_mat) if s]
    ops_2 = [Qobj(vh_mat_row.reshape(dim_2, dim_2),
                  dims_2, shape_2, copy=False)
             for vh_mat_row, s in zip(vh_mat, s_mat) if s]
    if len(dims) < 3:
        return list(zip(ops_1, ops_2))
    ops_2_factors = [factorize_qutip_operator(op2) for op2 in ops_2]
    return [(op1,) + factors
            for op1, op21_factors in zip(ops_1, ops_2_factors)
            for factors in op21_factors]


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
    names = x_op.site_names.copy()
    names.update(y_op.site_names)
    return QutipOperator(
        x_op.operator * y_op.operator,
        x_op.system or y_op.system,
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
    return QutipOperator(
        y_op.operator, names=y_op.site_names, prefactor=x_op.prefactor * y_op.prefactor
    )


@Operator.register_mul_handler(
    (
        QutipOperator,
        ScalarOperator,
    )
)
def mul_qutipop_with_scalarop(y_op: QutipOperator, x_op: ScalarOperator):
    """Sum a Scalar operator to a Qutip Operator"""
    return QutipOperator(
        y_op.operator, names=y_op.site_names, prefactor=x_op.prefactor * y_op.prefactor
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
    return QutipOperator(
        y_op * x_op.operator,
        x_op.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor,
    )
