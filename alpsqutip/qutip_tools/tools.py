"""
Functions for basic interface with qutip objects.
"""

import logging
from functools import reduce
from itertools import combinations
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
from numpy import ndarray, zeros as np_zeros
from numpy.linalg import eigh
from qutip import (  # type: ignore[import-untyped]
    Qobj,
    __version__ as qutip_version,
    qeye,
    tensor as qutip_tensor,
)
from scipy.linalg import norm as scipy_norm, svd  # type: ignore[import-untyped]
from scipy.sparse.linalg import ArpackNoConvergence

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

        def do_dense():
            arr = data.as_ndarray() if hasattr(data, "as_ndarray") else data.to_array()
            dim_i, dim_j = arr.shape
            for i in range(dim_i):
                for j in range(dim_j):
                    v = arr[i, j]
                    if v != 0:
                        yield (i, j, v)

        # Backward compatibility v5.0 and v5.1
        def do_dia_5_0(data_dia):
            data = data_dia.as_scipy()
            dim_i, dim_j = data.shape
            for offset, diag_data in zip(data.offsets, data.data):
                if offset < 0:
                    for j_pos, value in enumerate(diag_data):
                        i_pos = j_pos - offset
                        if i_pos >= dim_i:
                            break
                        yield (
                            i_pos,
                            j_pos,
                            value,
                        )
                else:
                    for i_pos, value in enumerate(diag_data):
                        j_pos = i_pos + offset
                        if j_pos >= dim_j:
                            break
                        yield (
                            i_pos,
                            j_pos,
                            value,
                        )

        def do_dia_5_2(data_dia):
            data = data_dia.as_scipy()
            dim_i, dim_j = data.shape
            for offset, diag_data in zip(data.offsets, data.data):
                if offset < 0:
                    for j_pos, value in enumerate(diag_data):
                        i_pos = j_pos - offset
                        if i_pos >= dim_i:
                            break
                        yield (
                            i_pos,
                            j_pos,
                            value,
                        )
                else:
                    for indx, value in enumerate(diag_data[offset:]):
                        i_pos = indx
                        j_pos = i_pos + offset
                        yield (
                            i_pos,
                            j_pos,
                            value,
                        )

        # Diagonal format
        if hasattr(data, "num_diag"):
            # For 5.0 and 5.1
            if int(qutip_version[2]) < 2:
                for item in do_dia_5_0(data):
                    yield item
            # For 5.2
            else:
                for item in do_dia_5_2(data):
                    yield item
            return
        elif hasattr(data, "as_scipy"):
            data = data.as_scipy()
            if hasattr(data, "tocoo"):
                coo = data.tocoo()
                for i, j, v in zip(coo.row, coo.col, coo.data):
                    yield (i, j, v)
            else:
                # Fallback: try nonzero and data.data
                try:
                    i_ind, j_ind = data.nonzero()
                    for idx in range(len(i_ind)):
                        yield (i_ind[idx], j_ind[idx], data.data[idx])
                except Exception:
                    # Last resort: try dense
                    for item in do_dense():
                        yield item
                    return
        else:
            for item in do_dense():
                yield item

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
            try:
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
            except ValueError:
                a00 = data[0, 0]
                for i in range(dim1):
                    for j in range(dim1):
                        if i == j:
                            if data[i, i] != a00:
                                return None
                        else:
                            if data[i, i]:
                                return None
                return a00

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


def data_has_nan(data) -> bool:
    """
    Check if data has `nan` entries
    """
    for i, j, val in data_element_iterator(data):
        if not (val == val):
            return True
    return False


def data_is_scalar(data) -> bool:
    """
    Check if data is a multiple of the identity matrix.
    """
    return scalar_value(data) is not None


def hermitician_part(op: Qobj, tol=None) -> Qobj:
    """Returns the hermitician part of the operator `op`"""
    if op.isherm:
        return op
    return (op + op.dag()).tidyup(tol) * 0.5


def is_scalar_op(op: Qobj) -> bool:
    """Check if op is a multiple of the identity operator"""
    return data_is_scalar(op.data)


def isnan_qutip(op: Qobj) -> bool:
    return data_has_nan(op.data)


def norm(
    op: Qobj,
    ord: Optional[int | str | float],
    axis: Optional[int | Tuple[int, int]] = None,
    keepdims: bool = False,
    check_finite: bool = True,
):
    """
    Compute the norm of `op` by converting it to a numpy.array.

    Parameters
    ----------
    a : array_like
        Input array. If `axis` is None, `a` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``a.ravel`` will be returned.
    ord : {int, inf, -inf, 'fro', 'nuc', None}, optional
        Order of the norm (see table under ``Notes``). inf means NumPy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `a` along which to
        compute the vector norms. If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed. If `axis` is None then either a vector norm (when `a`
        is 1-D) or a matrix norm (when `a` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one. With this option the result will
        broadcast correctly against the original `a`.

    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    `ord` is interpreted as:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(a), axis=1))      max(abs(a))
    -inf   min(sum(abs(a), axis=1))      min(abs(a))
    0      --                            sum(a != 0)
    1      max(sum(abs(a), axis=0))      as below
    -1     min(sum(abs(a), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(a)**ord)**(1./ord)
    =====  ============================  ==========================

    See scipy.linalg.norm
    """
    try:
        scipy_norm(op)
    except TypeError:
        # Version Qutip 5.2 does not support Qutip as ufunc. Handle
        # specific cases
        pass

    data = op.data
    if op.isbra or op.isket:
        return scipy_norm(data.to_array(), ord, axis, keepdims, check_finite)
    if op.isoper:
        data = op.data
        return scipy_norm(data.to_array(), ord, axis, keepdims, check_finite)


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
) -> Tuple[List[Qobj], ...]:
    """
    Decompose a qutip operator acting over H_1 (x) H_2 (x) H_3 (x)
    as a sum of terms of the form Q_{k} (x) Rest_{k}
    """
    dims = operator.dims[0]
    if len(dims) < 2:
        return ([operator],)
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


def schmidt_dec_first_rest_qutip_operator_hermitician(
    operator: Qobj, tol: float = 1e-10
) -> Tuple[List[Qobj], ...]:
    """
    Decompose a hermitician qutip operator acting over H_1 (x) H_2 (x) H_3 (x)
    as a sum of terms of the form Q_{k} (x) Rest_{k}
    """
    opsh_1 = []
    opsh_2 = []
    ops_1, *rest = schmidt_dec_first_rest_qutip_operator(operator, tol)
    if not rest:
        return (ops_1,)
    ops_2 = rest[0]

    # First, remove pairs of non-hermitician terms whose
    # sum is hermitician:
    candidates = [i for i, op_1 in enumerate(ops_1) if not op_1.isherm]
    remove_me = []
    while candidates:
        src = candidates.pop()
        op_1 = ops_1[src]
        for tgt in candidates:
            if data_is_zero(((ops_1[tgt].dag() - op_1).tidyup(tol)).data):
                remove_me.append(tgt)
                candidates.remove(tgt)
                ops_1[src] = op_1 * 2
                break

    ops_1 = [op_1 for pos, op_1 in enumerate(ops_1) if pos not in remove_me]
    ops_2 = [op_2 for pos, op_2 in enumerate(ops_2) if pos not in remove_me]

    # Process products of hermitician terms
    for op_1, op_2 in zip(ops_1, ops_2):
        if op_1.isherm:
            opsh_1.append(op_1)
            opsh_2.append(hermitician_part(op_2, tol))
            continue

        op_1h = hermitician_part(op_1, tol)
        if data_is_zero(op_1h.data):
            continue
        op_2h = hermitician_part(op_2, tol)
        opsh_1.append(op_1h)
        opsh_2.append(op_2h)

    # process products of anti-hermitician terms
    for op_1, op_2 in zip(ops_1, ops_2):
        if op_1.isherm:
            continue
        op_1h = hermitician_part(op_1 * 1j, tol)
        if data_is_zero(op_1h.data):
            continue
        op_2h = hermitician_part(op_2 * (-1j), tol)
        opsh_1.append(op_1h)
        opsh_2.append(op_2h)

    return opsh_1, opsh_2


def schmidt_dec_rest_last_qutip_operator(
    operator: Qobj, tol: float = 1e-10
) -> Tuple[List[Qobj], ...]:
    """
    Decompose a qutip operator acting over H_1 (x) H_2 (x) ... (x) H_n (x)
    as a sum of terms of the form Rest_{k} (x)  Q_{k}
    """
    dims = operator.dims[0]
    if len(dims) < 2:
        return ([operator],)
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


def schmidt_dec_rest_last_qutip_operator_hermitician(
    operator: Qobj, tol: float = 1e-10
) -> Tuple[List[Qobj], ...]:
    """
    Decompose a qutip operator acting over H_1 (x) H_2 (x) H_3 (x)
    as a sum of terms of the form Q_{k} (x) Rest_{k}
    """
    opsh_1 = []
    opsh_2 = []
    ops_1, *rest = schmidt_dec_rest_last_qutip_operator(operator, tol)
    if not rest:
        return (ops_1,)
    ops_2 = rest[0]

    # First, remove pairs of non-hermitician terms whose
    # sum is hermitician:
    candidates = [i for i, op_2 in enumerate(ops_2) if not op_2.isherm]
    remove_me = []
    while candidates:
        src = candidates.pop()
        op_2 = ops_2[src]
        for tgt in candidates:
            if data_is_zero(((ops_2[tgt].dag() - op_2).tidyup(tol)).data):
                remove_me.append(tgt)
                candidates.remove(tgt)
                ops_2[src] = op_2 * 2
                break

    ops_1 = [op_1 for pos, op_1 in enumerate(ops_1) if pos not in remove_me]
    ops_2 = [op_2 for pos, op_2 in enumerate(ops_2) if pos not in remove_me]

    for op_1, op_2 in zip(ops_1, ops_2):
        if op_2.isherm:
            opsh_1.append(hermitician_part(op_1, tol))
            opsh_2.append(op_2)
            continue

        op_2h = hermitician_part(op_2, tol)
        if data_is_zero(op_2h.data):
            continue
        op_1h = hermitician_part(op_1)
        opsh_1.append(op_1h)
        opsh_2.append(op_2h)

    for op_1, op_2 in zip(ops_1, ops_2):
        if op_2.isherm:
            continue
        op_2h = hermitician_part(op_2 * 1j, tol)
        if data_is_zero(op_2h.data):
            continue
        op_1h = hermitician_part(op_1 * (-1j), tol)
        opsh_1.append(op_1h)
        opsh_2.append(op_2h)

    return opsh_1, opsh_2


def decompose_qutip_operator(operator: Qobj, tol: float = 1e-10) -> List[Tuple]:
    """
    Decompose a qutip operator q123... into a sum
    of tensor products sum_{ka, kb, kc...} q1^{ka} q2^{kakb} q3^{kakbkc}...
    return a list of tuples, with each factor.
    """
    dims = operator.dims[0]
    ops_1, *rest = schmidt_dec_first_rest_qutip_operator(operator, tol)
    if len(rest) == 0:
        return [(op_l,) for op_l in ops_1]
    ops_2 = rest[0]
    if not ops_1:
        return []

    if len(dims) < 3:
        return [(op_1, op_2) for op_1, op_2 in zip(ops_1, ops_2)]
    ops_2_factors = [decompose_qutip_operator(op2, tol) for op2 in ops_2]
    result = [
        (op1,) + tuple((op_2 for op_2 in factors))
        for op1, op21_factors in zip(ops_1, ops_2_factors)
        for factors in op21_factors
    ]
    if result:
        return result
    return []


def decompose_qutip_operator_hermitician(
    operator: Qobj, tol: float = 1e-10
) -> List[Tuple]:
    """
    Decompose a hermitician qutip operator q123... into a sum
    of tensor products sum_{ka, kb, kc...} q1^{ka} q2^{kakb} q3^{kakbkc}...
    return a list of tuples, with each factor.
    """
    dims = operator.dims[0]

    ops_1, *rest = schmidt_dec_first_rest_qutip_operator_hermitician(operator, tol)

    if len(rest) == 0:
        return [(op_l,) for op_l in ops_1]
    ops_2 = rest[0]
    if not ops_1:
        return []

    if len(dims) < 3:
        return [(op_1, op_2) for op_1, op_2 in zip(ops_1, ops_2)]
    ops_2_factors = [decompose_qutip_operator_hermitician(op2, tol) for op2 in ops_2]
    result = [
        (op1,) + tuple((op_2 for op_2 in factors))
        for op1, op21_factors in zip(ops_1, ops_2_factors)
        for factors in op21_factors
    ]
    if result:
        return result
    return []


def get_proper_spaces(spectrum: Iterable) -> List[List[int]]:
    """
    Given a diagonal operator, find the proper spaces
    associated to each eigenvalue.
    """
    sectors_dict = {}
    for idx, sector in enumerate(spectrum):
        sectors_dict.setdefault(np.real(sector), []).append(idx)
    return list(sectors_dict.values())


def reduce_to_proper_spaces(operator: Qobj, observable: Qobj) -> Qobj:
    """
    Reduce operator to a block diagonal operator
    on each sector.

    If `observable` is of the form
    ```
    Q =  sum_i  lambda_i Pi_i
    ```
    with `Pi_i` a set of orthogonal projectors,
    for an `operator` `T`, this function returns
    ```
    Delta(T) = sum_i  Pi_i T P_i
    ```

    """
    # TODO: consider extend for the case of non-diagonal observables
    assert observable.isherm

    def build_proj_data(old_data, spectrum):
        proj_data = np.zeros(operator.shape, dtype=full_operator.dtype)
        for sector in get_proper_spaces(spectrum):
            for i in sector:
                for j in sector:
                    proj_data[i, j] = old_data[i, j]
        return proj_data

    is_diag = data_is_diagonal(observable.data)
    full_operator = operator.full()
    if is_diag:
        spectrum = observable.diag()
        new_data = build_proj_data(full_operator, spectrum)
    else:
        spectrum, unitary = eigh(observable.full())
        unitary_dag = unitary.T.conj()
        full_operator = unitary_dag @ full_operator @ unitary
        new_data = build_proj_data(full_operator, spectrum)
        new_data = unitary @ new_data @ unitary_dag

    return Qobj(
        new_data,
        dims=operator._dims,
        isherm=operator.isherm,
        isunitary=False,
        copy=False,
        # dtype=operator.dtype, # Not supported in Qutip <5.2
    )


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
    assert isinstance(operator, Qobj)
    num_eigvals = min(3, operator.shape[0])
    try:
        k_0 = max(
            np.real(
                operator.eigenenergies(sparse=True, sort="high", eigvals=num_eigvals)
            )
        )
    except ArpackNoConvergence as err_arpack:
        logging.warning(
            f"Convergence failed. try with {type(err_arpack.eigenvalues)}-> {err_arpack.eigenvalues}"
        )
        k_0 = (
            max([np.real(x) for x in err_arpack.eigenvalues])
            if err_arpack.eigenvalues
            else 0.0
        )

    op_exp = (operator - k_0).expm()
    op_exp_tr = op_exp.tr()
    op_exp = op_exp * (1.0 / op_exp_tr)
    k_0 = np.log(op_exp_tr) + k_0
    return op_exp, k_0
