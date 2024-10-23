from typing import Iterator, List, Tuple

from numpy import ndarray, zeros as np_zeros
from qutip import Qobj, __version__ as qutip_version
from scipy.linalg import svd

if int(qutip_version[0]) < 5:

    def data_element_iterator(data) -> Iterator:
        """
        Generator for the nontrivial elements.
        """
        i_idx, j_idx = data.nonzero()
        for item in zip(i_idx, j_idx, data.data):
            yield item

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

    def data_is_scalar(data) -> bool:
        """
        Check if data is a multiple of the identity matrix.
        """
        if data.nnz == 0:
            return True
        if all(a == b for a, b in zip(*data.nonzero())):
            dim1, dim2 = data.shape
            elems = data.data
            if len(elems) < dim1:
                return False
            val = elems[0]
            return all(val == elem for elem in elems)

        if any(val for val, a, b in zip(data.data, *data.nonzero()) if a != b):
            return False
        vals = [val for val, a, b in zip(data.data, *data.nonzero()) if a == b]
        val = vals[0]
        return all(elem == val for val in vals)

    def data_is_zero(data) -> bool:
        """
        check if the matrix is empty
        """
        if data.nnz == 0:
            return True
        return not any(data.data)

else:

    def data_element_iterator(data) -> Iterator:
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
            for item in zip(i_ind, j_ind, data.data):
                yield item
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

    def data_is_scalar(data) -> bool:
        """
        Check if data is a multiple of the identity matrix.
        """
        dim1, dim2 = data.shape
        if hasattr(data, "num_diag"):
            if data.num_diag == 0:
                return True
            if data.num_diag > 1:
                return False
            data = data.as_scipy()
            offsets = data.offsets
            if bool(offsets[0] != 0):
                return False
            diagonal = data.diagonal(0)
            scalar = diagonal[0]
            return len(diagonal) == dim1 and all(elem == scalar for elem in diagonal)

        if hasattr(data, "as_scipy"):
            data = data.as_scipy()
            if data.nnz == 0:
                return True
            if not all(a == b for a, b in zip(*data.nonzero())):
                return False
            dim = data.shape[0]
            data = data.data
            scalar = data[0]
            return len(data) == dim and all(value == scalar for value in data)

        print("must be dense")
        data = data.as_ndarray()
        dim_i, dim_j = data.shape
        if any(
            data[i_idx, j_idx]
            for i_idx in range(dim_i)
            for j_idx in range(dim_j)
            if i_idx != j_idx
        ):
            print("non null elements out of the diagonal")
            return False
        scalar = data[0, 0]
        print(scalar)
        print([data[i, i] for i in range(data.shape[0])])
        return all(scalar == data[i, i] for i in range(data.shape[0]))

    def data_is_zero(data) -> bool:
        """
        check if the matrix is empty
        """
        if hasattr(data, "num_diag"):
            return data.num_diag == 0
        if hasattr(data, "as_scipy"):
            return data.as_scipy().nnz == 0
        return not bool(data.as_ndarray().any())


def is_scalar_op(op: Qobj) -> bool:
    """Check if op is a multiple of the identity operator"""
    return data_is_scalar(op.data)


def reshape_qutip_data(data, dims) -> ndarray:
    """
    reshape the data representing an operator with dimensions
    dims = [[dim1, dim2,...],[dim1, dim2,...]]
    as an array with shape
    dims' = [[dim1,dim1],[dim2,dim3,... dim2,dim3,...]]
    """

    data_type = data_get_type(data)

    dim_1 = dims[0]
    dim_2 = int(data.shape[0] / dim_1)
    new_data = np_zeros(
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


def factorize_qutip_operator(operator: Qobj, tol: float = 1e-10) -> List[Tuple]:
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
    dim_2 = int(data.shape[0] / dim_1)
    dims_1 = [[dim_1], [dim_1]]
    dims_2 = [dims[1:], dims[1:]]
    u_mat, s_mat, vh_mat = svd(
        reshape_qutip_data(data, dims), full_matrices=False, overwrite_a=True
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
    if len(dims) < 3:
        return list(zip(ops_1, ops_2))
    ops_2_factors = [factorize_qutip_operator(op2, tol) for op2 in ops_2]
    return [
        (op1,) + factors
        for op1, op21_factors in zip(ops_1, ops_2_factors)
        for factors in op21_factors
    ]
