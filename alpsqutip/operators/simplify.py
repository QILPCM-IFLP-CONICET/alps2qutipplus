# -*- coding: utf-8 -*-
"""
Functions to simplify sums of operators
"""
import numpy as np
from qutip import tensor
from scipy.linalg import svd

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.states import DensityOperatorMixin
from alpsqutip.qutip_tools.tools import data_is_diagonal, factorize_qutip_operator


def collect_nbody_terms(operator: Operator) -> dict:
    """
    build a dictionary whose keys are subsystems and
    the values are lists of operators acting exactly
    over the subsystem.
    """
    terms_by_block = {None: []}
    scalar_term = 0.0
    system = operator.system

    for term in operator.terms:
        acts_over = term.acts_over()
        if acts_over is None:
            acts_over_key = None
            terms_by_block[None].append(term)
            continue

        acts_over_key = tuple(acts_over)
        num_bodies = len(acts_over_key)
        if num_bodies == 0:
            scalar_term += term.prefactor
        else:
            acts_over_key = tuple(acts_over)
            terms_by_block.setdefault(acts_over_key, []).append(term)

    # Add a scalar term
    if scalar_term:
        terms_by_block[tuple()] = [ScalarOperator(scalar_term, system)]
    return terms_by_block


def sums_as_blocks(operator, fn=None):
    """
    Rewrite a sum of operators as a sum
    of a ScalarOperator, a OneBodyOperator
    and terms acting on different blocks.

    For many-body terms, apply fn for further
    simplifications.

    For example
    ```
    sums_as_blocks(operator, lambda op:op.to_qutip_operator())
    ```
    convert these many-body terms into Qutip operators,
    which for small blocks could provide a more efficient
    representation.

    """
    system = operator.system
    if not isinstance(operator, SumOperator):
        return operator
    if isinstance(operator, (OneBodyOperator, DensityOperatorMixin)):
        return operator

    operator = operator.flat()
    terms_dict = collect_nbody_terms(operator)
    new_terms = []
    one_body_terms = []
    isherm = operator._isherm
    for block, terms in terms_dict.items():
        if block is None or len(block) == 0:
            new_terms.extend(terms)
        elif len(block) == 1:
            one_body_terms.extend(block)
        else:
            new_term = SumOperator(tuple(terms), system, isherm=isherm)
            if fn is not None:
                new_term = fn(new_term)
            new_terms.append(new_term)

    new_term = OneBodyOperator(tuple(one_body_terms), system)
    new_terms.append(new_term)
    return SumOperator(tuple(new_terms), system, isherm=isherm)


def post_process_collections(collection: dict) -> dict:
    """
    Collect terms acting on blocks or subblocks
    """
    new_collection = {}
    keys = sorted((c for c in collection if c is not None), key=lambda x: -len(x))

    for key in keys:
        found = False
        for existent_key, existent_dict in new_collection.items():
            if all(q in existent_key for q in key):
                existent_dict.extend(collection[key])
                found = True
                break
        if not found:
            new_collection[key] = collection[key].copy()

    if None in collection:
        new_collection[None] = collection[None]
    return new_collection


def rewrite_nbody_term_using_qutip(
    operator_list: list,
    block: tuple,
    system: SystemDescriptor,
    isherm: bool = None,
    isdiag: bool = None,
) -> Operator:
    """
    Do the decomposition work using qutip
    """
    block_sites = sorted(block)
    sites_identity = {}

    def op_or_identity(term, site):
        result = term.sites_op.get(site, None) or sites_identity.get(site, None)
        if result is None:
            result = system.sites[site]["operators"]["identity"]
            sites_identity[site] = result
        return result

    qutip_subop = sum(
        tensor(*(op_or_identity(term, site) for site in block_sites)) * term.prefactor
        for term in operator_list
    )
    if isherm is None:
        isherm = qutip_subop.isherm
    if isdiag is None:
        isdiag = data_is_diagonal(qutip_subop.data)
    # Now, decompose the operator again as a sum of n-body terms
    factor_terms = factorize_qutip_operator(qutip_subop)
    new_terms = (
        ProductOperator(
            dict(zip(block_sites, factors)),
            1,
            system,
        )
        for factors in factor_terms
    )
    return SumOperator(
        tuple(new_terms),
        system,
        isherm=isherm,
        isdiag=isdiag,
    )


def rewrite_nbody_term_using_orthogonal_decomposition(
    operator_list: list,
    block: tuple,
    system: SystemDescriptor,
    isherm: bool = None,
    isdiag: bool = None,
) -> Operator:
    """
    Do the decomposition work using qutip
    """
    # Build the Gram's matrix
    # TODO: exploit isherm
    basis = operator_list

    def sp(a, b):
        """HS scalar product over block"""
        sites_op_a, sites_op_b = a.sites_op, b.sites_op
        result = 0
        for site in block:
            op_a_i = sites_op_a.get(site, None)
            op_b_i = sites_op_b.get(site, None)
            if op_a_i is None:
                result += op_b_i.tr()
            elif op_b_i is None:
                result += np.conj(op_a_i.tr())
            else:
                result += (op_a_i.dag() * op_b_i).tr()
        return result

    gram_matrix = np.array([[sp(op1, op2) for op2 in basis] for op1 in basis])
    u_mat, s_diag, uh_mat = svd(gram_matrix, full_matrices=False, overwrite_a=True)
    nontrivial = s_diag > 1e-12
    u_mat, uh_mat = u_mat[:, nontrivial], uh_mat[nontrivial]
    coeffs = sum(u_mat.dot(uh_mat))
    new_terms = (
        op_i * coeff for coeff, op_i in zip(coeffs, basis) if abs(coeff) > 1e-10
    )
    return SumOperator(
        tuple(new_terms),
        system,
        isherm=isherm,
        isdiag=isdiag,
    )


def simplify_sum_using_qutip(operator: Operator) -> Operator:
    """
    Decompose Operator as a sum of n-body terms,
    convert each term to a qutip operator,
    and decompose each operator again as a sum
    of n-body terms
    """
    operator = operator.flat()
    if not isinstance(operator, SumOperator):
        return operator

    system = operator.system
    isherm = operator._isherm
    isdiag = operator._isdiagonal

    new_terms = []
    terms_by_block = post_process_collections(collect_nbody_terms(operator))

    # Process the n-body terms
    for block, block_list in terms_by_block.items():
        # For one-body terms, just add all of them as qutip operators
        if block is None or len(block) == 0:
            new_terms.extend(block_list)
            continue
        if len(block) == 1:
            new_terms.append(
                LocalOperator(block[0], sum(term.to_qutip() for term in block_list))
            )
            continue

        # For n>1 n-body terms, rebuild the local operator
        # Notice that if Operator is diagonal / hermitician,
        # each independent N-body term must be too.
        new_term = rewrite_nbody_term_using_qutip(
            block_list, block, system, isherm, isdiag
        )
        new_terms.append(new_term)

    if len(new_terms) == 0:
        return ScalarOperator(0, system)
    if len(new_terms) == 1:
        return new_terms[0]
    return SumOperator(tuple(new_terms), system, isherm=isherm, isdiag=isdiag)


def simplify_sum_using_orthogonal_decomposition(operator: Operator) -> Operator:
    """
    Decompose Operator as a sum of n-body terms,
    convert each term to a qutip operator,
    and decompose each operator again as a sum
    of n-body terms
    """
    operator = operator.flat()
    if not isinstance(operator, SumOperator):
        return operator

    system = operator.system
    isherm = operator._isherm
    isdiag = operator._isdiagonal

    new_terms = []
    terms_by_block = collect_nbody_terms(operator)

    # Process the n-body terms
    for block, block_list in terms_by_block.items():
        # For one-body terms, just add all of them as qutip operators
        if block is None or len(block) == 0:
            new_terms.extend(block_list)
            continue
        if len(block) == 1:
            new_terms.append(
                LocalOperator(
                    tuple(block)[0], sum(term.to_qutip() for term in block_list)
                )
            )
            continue

        # For n>1 n-body terms, rebuild the local operator
        # Notice that if Operator is diagonal / hermitician,
        # each independent N-body term must be too.
        new_term = rewrite_nbody_term_using_orthogonal_decomposition(
            block_list, block, system, isherm, isdiag
        )
        new_terms.append(new_term)

    if len(new_terms) == 0:
        return ScalarOperator(0, system)
    if len(new_terms) == 1:
        return new_terms[0]
    return SumOperator(tuple(new_terms), system, isherm=isherm, isdiag=isdiag)
