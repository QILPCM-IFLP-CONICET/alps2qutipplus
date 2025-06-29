"""
Functions to fetch specific scalar product functions.
"""

# from datetime import datetime
from typing import Callable

import numpy as np
from numpy import real

from alpsqutip.operators.basic import Operator
from alpsqutip.operators.functions import anticommutator
from alpsqutip.operators.states import DensityOperatorMixin

#  ### Functions that build the scalar products ###


class CovariantScalarProductFunction(Callable):
    """
    A callable object that computes the Covariance scalar
    product of two operators, relative to a given
    reference state sigma.
    """

    def __init__(self, state):

        self.sigma = state

    def __call__(self, op1, op2):
        sigma = self.sigma
        if op1 is op2:
            op1 = op1.simplify()
            op1_herm = op1.isherm
            if op1_herm:
                return abs(sigma.expect(op1 * op1))
            return abs(0.5 * sigma.expect(anticommutator(op1.dag(), op1).simplify()))

        op1 = op1.simplify()
        op1_herm = op1.isherm
        op2 = op2.simplify()
        op2_herm = op2.isherm

        if op1_herm:
            if op2_herm:
                o1o2 = (op1 * op2).simplify()
                return real(sigma.expect(o1o2))
            op1_dag = op1
        else:
            op1_dag = op1.dag()
        if op1_dag is op2:
            return sigma.expect((op1_dag * op2).simplify())
        else:
            return 0.5 * sigma.expect(anticommutator(op1_dag, op2).simplify())


def fetch_kubo_scalar_product(sigma: Operator, threshold=0) -> Callable:
    """
    Build a KMB scalar product function
    associated to the state `sigma`
    """
    evals_evecs = sorted(zip(*sigma.eigenstates()), key=lambda x: -x[0])
    w = 1
    for i, val_vec in enumerate(evals_evecs):
        p = val_vec[0]
        w -= p
        if w < threshold or p <= 0:
            evals_evecs = evals_evecs[: i + 1]
            break

    def ksp(op1, op2):
        result = sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * (p1 if p1 == p2 else (p1 - p2) / np.log(p1 / p2))
            )
            for p1, v1 in evals_evecs
            for p2, v2 in evals_evecs
            if (p1 > 0 and p2 > 0)
        )

        #    stored[key] = result
        return result

    return ksp


def fetch_kubo_int_scalar_product(sigma: Operator) -> Callable:
    """
    Build a KMB scalar product function
    associated to the state `sigma`, from
    its integral form.
    """

    evals, evecs = sigma.eigenstates()

    def return_func(op1, op2):
        return 0.01 * sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * ((p1) ** (1.0 - tau))
                * ((p1) ** (tau))
            )
            for p1, v1 in zip(evals, evecs)
            for p2, v2 in zip(evals, evecs)
            for tau in np.linspace(0.0, 1.0, 100)
            if (p1 > 0.0 and p2 > 0.0)
        )

    return return_func


def fetch_covar_scalar_product(sigma: DensityOperatorMixin) -> Callable:
    """
    Returns a scalar product function based on the covariance of a density
    operator.

    The scalar product for two operators op1 and op2 is defined as:
        0.5 * Tr(sigma * {op1†, op2}),
    where sigma is a density operator, {op1†, op2} is the anticommutator of
    the Hermitian conjugate of op1 and op2, and Tr denotes the trace.

    Parameters:
        sigma: The density operator (quantum state) used to define the scalar
        product.

    Returns:
        A function that takes two operators (op1, op2) and computes their
        covariance-based scalar product.
    """
    return CovariantScalarProductFunction(sigma)


def fetch_HS_scalar_product() -> Callable:
    """
    Build a HS scalar product function
    """
    return lambda op1, op2: (op1.dag() * op2).tr()
