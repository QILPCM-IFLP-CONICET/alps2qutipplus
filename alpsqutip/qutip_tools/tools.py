"""
Functions for basic interface with qutip objects.
"""

from functools import reduce
from itertools import combinations
from typing import Iterator, List, Tuple

import numpy as np
from numpy import ndarray, zeros as np_zeros
from qutip import (  # type: ignore[import-untyped]
    Qobj,
    __version__ as qutip_version,
    qeye,
    tensor as qutip_tensor,
)
from scipy.linalg import svd  # type: ignore[import-untyped]

if int(qutip_version[0]) < 5:

    def data_element_iterator(data) -> Iterator:
        """
        Generator for the nontrivial elements.
        """
        i_idx, j_idx = data.nonzero()
        yield from zip(i_idx, j_idx, data.data)

    def data_get_coeff(data, i_idx, j_idx):
        """
        Access to a matrix entry
        """
        return data[i_idx, j_idx]

    def data_get_type(data) -> type:
        """
        Get the type of the elements in data
        """
        return data.dtype

    def data_is_diagonal(data) -> bool:
        """
        Check if data is diagonal
        """
        if data.nnz == 0 or all(a == b for a, b in zip(*data.nonzero())):
            return True
        return all(val == 0 for val, a, b in zip(data.data, *data.nonzero()) if a != b)

    def data_is_zero(data) -> bool:
        """
        check if the matrix is empty
        """
        if data.nnz == 0:
            return True
        return not any(data.data)

    def scalar_value(data):
        """
        If data is a scalar matrix, return any
        of its diagonal elements. Otherwise, return `None`
        """
        if data.nnz == 0:
            return 0.0
        if all(a == b for a, b in zip(*data.nonzero())):
            dim1, _ = data.shape
            elems = data.data
            if len(elems) < dim1:
                return None
            val = elems[0]
            return val if all(val == elem for elem in elems) else None

        if any(val for val, a, b in zip(data.data, *data.nonzero()) if a != b):
            return None
        vals = [val for val, a, b in zip(data.data, *data.nonzero()) if a == b]
        elem = vals[0]
        return elem if all(elem == val for val in vals) else None

else:

    def data_element_iterator(data) -> Iterator:
        """
        walk over data elements.
        """
        if hasattr(data, "num_diag"):
            data = data.as_scipy()
            for offset, diag_data in zip(data.offsets, data.data):
                if offset < 0:
                    for idx, value in enumerate(diag_data):
                        yield (
                            idx - offset,
                            idx,
                            value,
                        )
                else:
                    for idx, value in enumerate(diag_data):
                        yield (
                            idx,
                            idx + offset,
                            value,
                        )
        elif hasattr(data, "as_scipy"):
            data = data.as_scipy()
            i_ind, j_ind = data.nonzero()
            yield from zip(i_ind, j_ind, data.data)
        else:
            data = data.as_ndarray()
            dim_i, dim_j = data.shape
            for i_idx in range(dim_i):
                for j_idx in range(dim_j):
                    yield (
                        i_idx,
                        j_idx,
                        data[i_idx, j_idx],
                    )

    def data_get_coeff(data, i_idx, j_idx):
        """
        Access to a matrix entry
        """
        if hasattr(data, "num_diag"):
            data_sp = data.as_scipy()
            offset = j_idx - i_idx
            offsets = data_sp.offsets
            if offset not in offsets:
                return 0
            return data_sp.diagonal(offset)[j_idx]
        if hasattr(data, "as_scipy"):
            return data.as_scipy()[i_idx, j_idx]
        return data.as_ndarray()[i_idx, j_idx]

    def data_get_type(data) -> type:
        """
        Get the type of the elements in data
        """
        if hasattr(data, "as_scipy"):
            return data.as_scipy().dtype
        return data.as_ndarray().dtype

    def data_is_diagonal(data) -> bool:
        """
        Check if data is diagonal
        """
        if hasattr(data, "num_diag"):
            if data.num_diag == 0:
                return True
            if data.num_diag > 1:
                return False
            offsets = data.as_scipy().offsets
            return bool(offsets[0] == 0)
        if hasattr(data, "as_scipy"):
            data = data.as_scipy()
            if data.nnz == 0:
                return True
            return all(a == b for a, b in zip(*data.nonzero()))
        data = data.as_ndarray()
        dim_i, dim_j = data.shape
        return not any(
            data[i_idx, j_idx]
            for i_idx in range(dim_i)
            for j_idx in range(dim_j)
            if i_idx != j_idx
        )

    def data_is_zero(data) -> bool:
        """
        check if the matrix is empty
        """
        if hasattr(data, "num_diag"):
            return data.num_diag == 0
        if hasattr(data, "as_scipy"):
            return data.as_scipy().nnz == 0
        return not bool(data.as_ndarray().any())

    def scalar_value(data):
        """
        If data is a scalar matrix, return any
        of its diagonal elements. Otherwise, return `None`
        """

        dim1, _ = data.shape
        if hasattr(data, "num_diag"):
            if data.num_diag == 0:
                return 0.0
            if data.num_diag > 1:
                return None
            data = data.as_scipy()
            offsets = data.offsets
            if bool(offsets[0] != 0):
                return None
            diagonal = data.diagonal(0)
            scalar = diagonal[0]
            return (
                scalar
                if len(diagonal) == dim1 and all(elem == scalar for elem in diagonal)
                else None
            )

        if hasattr(data, "as_scipy"):
            data = data.as_scipy()
            if data.nnz == 0:
                return 0.0
            if not all(a == b for a, b in zip(*data.nonzero())):
                return None
            dim = data.shape[0]
            data = data.data
            scalar = data[0]
            return (
                scalar
                if len(data) == dim and all(value == scalar for value in data)
                else None
            )

        # Must be dense...
        data = data.as_ndarray()
        dim_i, dim_j = data.shape
        if any(
            data[i_idx, j_idx]
            for i_idx in range(dim_i)
            for j_idx in range(dim_j)
            if i_idx != j_idx
        ):
            return None
        scalar = data[0, 0]
        return (
            scalar if all(scalar == data[i, i] for i in range(data.shape[0])) else None
        )


def data_is_scalar(data) -> bool:
    """
    Check if data is a multiple of the identity matrix.
    """
    return scalar_value(data) is not None


def is_scalar_op(op: Qobj) -> bool:
    """Check if op is a multiple of the identity operator"""
    return data_is_scalar(op.data)


def reshape_qutip_data(data, dims, bs=1) -> ndarray:
    """
    reshape the data representing an operator with dimensions
    dims = [[dim1, dim2,...],[dim1, dim2,...]]
    as an array with shape
    dims' = [[dim1,dim1],[dim2,dim3,... dim2,dim3,...]]
    """

    data_type = data_get_type(data)

    dim_1 = reduce(lambda x, y: x * y, dims[:bs])
    dim_2 = int(data.shape[0] / dim_1)
    new_data: ndarray = np_zeros(
        (
            dim_1**2,
            dim_2**2,
        ),
        dtype=data_type,
    )
    # reshape the operator
    # TODO: see to exploit the sparse structure of data to build the matrix
    for alpha, beta, value in data_element_iterator(data):
        i_idx, k_idx = divmod(alpha, dim_2)
        j_idx, l_idx = divmod(beta, dim_2)
        gamma = dim_1 * i_idx + j_idx
        delta = dim_2 * k_idx + l_idx
        new_data[gamma, delta] = value

    return new_data


def schmidt_dec_first_rest_qutip_operator(
    operator: Qobj, tol: float = 1e-10
) -> List[Tuple]:
    """
    Decompose a qutip operator acting over H_1 (x) H_2 (x) H_3 (x)
    as a sum of terms of the form Q_{k} (x) Rest_{k}
    """
    dims = operator.dims[0]
    if len(dims) < 2:
        return [(operator,)]
    data = operator.data
    dim_1 = dims[0]
    dim_2 = int(data.shape[0] / dim_1)
    dims_1 = [[dim_1], [dim_1]]
    dims_2 = [dims[1:], dims[1:]]
    u_mat, s_mat, vh_mat = svd(
        reshape_qutip_data(data, dims, 1), full_matrices=False, overwrite_a=True
    )
    ops_1 = [
        Qobj(s * u_mat[:, i].reshape(dim_1, dim_1), dims=dims_1, copy=False)
        for i, s in enumerate(s_mat)
        if s > tol
    ]
    ops_2 = [
        Qobj(vh_mat_row.reshape(dim_2, dim_2), dims=dims_2, copy=False)
        for vh_mat_row, s in zip(vh_mat, s_mat)
        if s > tol
    ]
    return ops_1, ops_2


def schmidt_dec_firsts_last_qutip_operator(
    operator: Qobj, tol: float = 1e-10
) -> List[Tuple]:
    """
    Decompose a qutip operator acting over H_1 (x) H_2 (x) H_3 (x)
    as a sum of terms of the form Q_{k} (x) Rest_{k}
    """
    dims = operator.dims[0]
    if len(dims) < 2:
        return [(operator,)]
    data = operator.data
    dim_2 = dims[-1]
    dim_1 = int(data.shape[0] / dim_2)
    dims_1 = [dims[:-1], dims[:-1]]
    dims_2 = [[dim_2], [dim_2]]
    u_mat, s_mat, vh_mat = svd(
        reshape_qutip_data(data, dims, -1), full_matrices=False, overwrite_a=True
    )
    ops_1 = [
        Qobj(s * u_mat[:, i].reshape(dim_1, dim_1), dims=dims_1, copy=False)
        for i, s in enumerate(s_mat)
        if s > tol
    ]
    ops_2 = [
        Qobj(vh_mat_row.reshape(dim_2, dim_2), dims=dims_2, copy=False)
        for vh_mat_row, s in zip(vh_mat, s_mat)
        if s > tol
    ]
    return ops_1, ops_2


def decompose_qutip_operator(operator: Qobj, tol: float = 1e-10) -> List[Tuple]:
    """
    Decompose a qutip operator q123... into a sum
    of tensor products sum_{ka, kb, kc...} q1^{ka} q2^{kakb} q3^{kakbkc}...
    return a list of tuples, with each factor.
    """
    dims = operator.dims[0]
    ops_1, *ops_2 = schmidt_dec_first_rest_qutip_operator(operator, tol)
    if len(ops_2) == 0:
        return [(op_l,) for op_l in ops_1]
    ops_2 = ops_2[0]
    if len(dims) < 3:
        return list(zip(ops_1, ops_2))
    ops_2_factors = [decompose_qutip_operator(op2, tol) for op2 in ops_2]
    return [
        (op1,) + factors
        for op1, op21_factors in zip(ops_1, ops_2_factors)
        for factors in op21_factors
    ]


def project_qutip_to_m_body(op_qutip: Qobj, m_max=2, local_sigmas=None) -> Qobj:
    """
    Project a qutip operator onto a m_max - body operators sub-algebra
    relative to the local states `local_sigmas`.
    If `local_sigmas` is not given, maximally mixed states are assumed.
    """
    scalar_term = 0
    dimensions = op_qutip.dims[0]
    site_indx = list(range(len(dimensions)))
    idops = [qeye(dim) for dim in dimensions]
    result = qutip_tensor([0 * local_id for local_id in idops])
    # Decompose the operator
    decompose = decompose_qutip_operator(op_qutip)
    # Build the local states
    if local_sigmas is None:
        local_sigmas = [1 / dim for dim in dimensions]
    for term in decompose:
        term = [t.tidyup() for t in term]
        local_exp_vals = [
            (state * factor).tr() for state, factor in zip(local_sigmas, term)
        ]
        local_delta_op = [
            factor - exp_val for factor, exp_val in zip(term, local_exp_vals)
        ]
        scalar_term += reduce(lambda x, y: x * y, local_exp_vals)
        for m in range(m_max):
            for fluc_sites in combinations(site_indx, m + 1):
                prefactor = reduce(
                    lambda x, y: x * y,
                    [
                        e_val
                        for i, e_val in enumerate(local_exp_vals)
                        if i not in fluc_sites
                    ],
                    1,
                )
                if abs(prefactor) < 1e-10:
                    continue
                new_term = (
                    qutip_tensor(
                        [
                            local_delta_op[idx] if idx in fluc_sites else idops[idx]
                            for idx in site_indx
                        ]
                    )
                    * prefactor
                )
                result += new_term
    return result + scalar_term


def safe_exp_and_normalize(operator: Qobj) -> Tuple[Qobj, float]:
    """
    Compute the decomposition of exp(operator) as rho*exp(f)
    with f = Tr[exp(operator)], for operator a Qutip operator.

    operator: Qobj

    result: Tuple[Qobj, float]
         (exp(operator)/f , f)

    """
    num_eigvals = min(3, operator.shape[0])
    k_0 = max(
        np.real(operator.eigenenergies(sparse=True, sort="high", eigvals=num_eigvals))
    )
    op_exp = (operator - k_0).expm()
    op_exp_tr = op_exp.tr()
    op_exp = op_exp * (1.0 / op_exp_tr)
    k_0 = np.log(op_exp_tr) + k_0
    return op_exp, k_0
