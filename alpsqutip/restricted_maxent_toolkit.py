"""
Functions used to run MaxEnt simulations.
"""

from typing import Callable, List

import numpy as np

from alpsqutip.operators import Operator
from alpsqutip.operators.functions import commutator

# function used to safely and robustly map K-states to states


def build_hierarchical_basis(generator, seed_op, deep) -> List[Operator]:
    """
    Constructs a hierarchical basis of operators, formed from iterated
    commutators of a seed operator.

    Parameters:
        generator: The generator operator (e.g., a Hamiltonian).
                    It should be passed as a QutipAlps operator
                    (not Qutip.qobj)
        seed_op: The initial seed operator for generating the basis.
                 If None, the function will return an empty list.
        deep: An integer indicating the depth of the hierarchy
        (number of iterated commutators).

    Returns:
        A list of operators representing the hierarchical basis,
        starting with the seed operator, followed by operators generated by
        successive commutators.
    """
    basis = []
    if seed_op is not None and deep > 0:
        basis += [
            seed_op.to_qutip_operator()
        ]  # Include the seed operator in the basis.
        for _ in range(deep):
            # Generate new operators by computing the commutator
            # of the generator with the last operator.
            basis.append(commutator(generator, 1j * basis[-1]))

    return basis


def fn_hij_tensor(basis, sp: Callable, generator):
    """
    Computes the Hij-tensor, a local matrix representation of the Hamiltonian
    onto the given basis.

    For each pair of basis operators (op1, op2), the matrix element is defined
    as:
        Hij = sp(op1, commutator(-1j * generator, op2))

    Parameters:
        basis: A list of basis operators.
        sp: A callable that defines a scalar product function between two
        operators.
        generator: The operator (e.g., Hamiltonian) for which the commutators
        are computed.

    Returns:
        A real-valued NumPy array representing the Hamiltonian matrix in the
        given basis.
    """
    local_h_ij = np.array(
        [[sp(op1, commutator(-1j * generator, op2)) for op2 in basis] for op1 in basis]
    )
    return np.real(local_h_ij)


def fn_hij_tensor_with_errors(basis, sp: Callable, generator):
    """Compute the tensor Hij and the norm of the orthogonal projection"""
    hgen = -1j * generator
    comm_h_ops = [commutator(hgen, op2).simplify() for op2 in basis]

    local_h_ij = np.zeros([len(basis), len(basis)], dtype=complex)
    for i, b in enumerate(basis):
        for j, comm_op in enumerate(comm_h_ops):
            res = sp(b, comm_op)
            local_h_ij[i, j] = res

    # local_h_ij = np.array(
    #    [[sp(op1, comm_op) for comm_op in comm_h_ops] for op1 in basis]
    # )
    proj_comm_norms_sq = (sum(col**2) for col in local_h_ij.transpose())
    comm_full_norms_sq = (sp(comm_op, comm_op) for comm_op in comm_h_ops)
    errors_w = [
        (max(full_sq - proj_sq, 0.0)) ** 0.5
        for full_sq, proj_sq in zip(comm_full_norms_sq, proj_comm_norms_sq)
    ]
    return local_h_ij, errors_w


def k_state_from_phi_basis(phi: np.array, basis):
    """
    Constructs the operator K from a given set of coefficients and
    basis operators.

    Parameters:
        phi: A NumPy array containing the coefficients for the linear
             combination.
        basis: A list of basis operators.

    Returns:
        The operator K, defined as the negative linear combination of the basis
        operators weighted by the coefficients in `phi`. If `phi` is shorter
        than the basis, it is padded with zeros.
    """
    if len(phi) < len(basis):
        phi = np.array(list(phi) + [0.0 for _ in range(len(basis) - len(phi))])
    return -sum(phi_a * opa for phi_a, opa in zip(phi, basis))


def slice_times(tlist: np.array, tcuts):
    """
    Divides a time list (`tlist`) into slices based on a sequence of cutoff
    times (`tcuts`).

    Parameters:
        tlist: A NumPy array of time values to be sliced.
        tcuts: A list or array of cutoff times used to define the time slices.

    Returns:
        A list of NumPy arrays, where each array corresponds to a segment of
        `tlist`
        based on the intervals defined by `tcuts`.
        - The first slice includes times up to `tcuts[1]`.
        - Subsequent slices include times between `tcuts[d-1]` and `tcuts[d]`.
        - If there are remaining times beyond `tcuts[-1]`, they are included
        in the last slice.
    """
    sliced_times = [np.array([t for t in tlist if t <= tcuts[1]])]

    for d in range(2, len(tcuts)):
        local_tlist = np.array([t for t in tlist if tcuts[d - 1] <= t <= tcuts[d]])
        sliced_times.append(local_tlist)

    if tlist[-1] > tcuts[-1]:
        sliced_times.append(np.array([t for t in tlist if t >= tcuts[-1]]))

    return sliced_times


def m_th_partial_sum(phi, m=int):
    """
    Computes the $m$-th partial sum of the squared magnitudes of the last `m`
    coefficients of `phi`.

    Parameters:
        phi: A NumPy array containing coefficients.
        m: An integer specifying how many of the last coefficients to include
        in the sum.

    Returns:
        The partial sum of the squared magnitudes of the last `m` coefficients
        in `phi`.
    """
    return sum(abs(phi_n) ** 2 for phi_n in phi[-m:])
