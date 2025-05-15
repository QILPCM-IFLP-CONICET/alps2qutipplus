# -*- coding: utf-8 -*-
"""
Functions to simplify sums of operators
"""
from numbers import Number

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
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states import DensityOperatorMixin
from alpsqutip.qutip_tools.tools import data_is_diagonal, decompose_qutip_operator
from alpsqutip.scalarprod import orthogonalize_basis
from alpsqutip.settings import ALPSQUTIP_TOLERANCE


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
            acts_over_key = tuple(sorted(acts_over))
            terms_by_block.setdefault(acts_over_key, []).append(term)

    # Add a scalar term
    if scalar_term:
        terms_by_block[tuple()] = [ScalarOperator(scalar_term, system)]
    return terms_by_block


def group_terms_by_blocks(operator, fn=None):
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
    changed = False
    system = operator.system
    assert operator is not None

    if (
        not isinstance(operator, SumOperator)
        or operator._simplified
        or isinstance(operator, (OneBodyOperator, DensityOperatorMixin))
    ):
        return operator

    operator_flat = operator.flat()
    if operator_flat is not operator:
        changed = True
        operator = operator_flat

    assert operator is not None
    terms_dict = collect_nbody_terms(operator)
    new_terms = []
    one_body_terms = []
    isherm = operator._isherm
    isdiag = operator._isdiagonal

    for block, terms in terms_dict.items():
        if block is None or len(block) == 0:
            new_terms.extend(terms)
        elif len(block) == 1:
            one_body_terms.extend(terms)
        else:
            if len(terms) > 1:
                new_term = SumOperator(tuple(terms), system=system)

                try:
                    new_term_simpl = simplify_qutip_sums(new_term)
                    if new_term_simpl is not new_term:
                        changed = True
                        new_term = new_term_simpl
                except Exception:
                    pass

            else:
                new_term = terms[0]
            if fn is not None:
                try:
                    new_term_fn = fn(new_term)
                    if new_term_fn is not new_term:
                        new_term = new_term_fn
                        changed = True
                except Exception:
                    pass
            new_terms.append(new_term)

    new_terms = [term for term in new_terms if term]

    if one_body_terms:
        new_term = OneBodyOperator(tuple(one_body_terms), system)
        changed = True
        if new_term:
            if len(new_terms) == 0:
                return new_term
            new_terms.append(new_term)
        else:
            if len(new_terms) == 0:
                return ScalarOperator(0, system)
            if len(new_terms) == 1:
                return new_terms[0]
    else:
        if len(new_terms) == 0:
            return ScalarOperator(0, system)
        if len(new_terms) == 1:
            return new_terms[0]

    if not changed:
        operator._simplified = True
        return operator
    return SumOperator(
        tuple(new_terms), system=system, isherm=isherm, isdiag=isdiag, simplified=True
    )


def simplify_qutip_sums(sum_operator):
    """
    collect terms acting on the same block of sites,
    and reduce it to a single qutip operator.
    """
    if not isinstance(sum_operator, SumOperator):
        return sum_operator

    changed = False
    system = sum_operator.system
    terms = []
    qutip_terms = dict()
    product_terms = dict()
    for term in sum_operator.terms:
        if isinstance(term, ProductOperator):
            product_terms.setdefault(frozenset(term.acts_over()), []).append(term)
        elif isinstance(term, QutipOperator):
            qutip_terms.setdefault(frozenset(term.acts_over()), []).append(term)
        else:
            terms.append(term)

    # Process first the product operator terms
    for block, p_terms in product_terms.items():
        # If block is in qutip_terms, or there are more than a single product term,
        # and each product term acts on few sites, it is more efficient to store
        # them as a single qutip operator:
        if block in qutip_terms or (len(p_terms) > 1 and len(block) < 6):
            changed = True
            block_tuple = tuple(sorted(block))
            sum_qutip_op = sum(
                term.to_qutip_operator().to_qutip(block_tuple) for term in p_terms
            )
            qutip_terms.setdefault(block, []).append(
                QutipOperator(
                    sum_qutip_op,
                    names={site: idx for idx, site in enumerate(block_tuple)},
                    system=system,
                )
            )
            continue
        # Otherwise, just add as terms
        terms.extend(p_terms)

    # Now,
    for block, q_terms in qutip_terms.items():
        block_tuple = tuple(sorted(block))
        if len(q_terms) == 1:
            terms.append(q_terms[0])
            continue
        changed = True
        new_qterm = sum(q_term.to_qutip(block_tuple) for q_term in q_terms)
        terms.append(
            QutipOperator(
                new_qterm,
                names={site: pos for pos, site in enumerate(block_tuple)},
                system=system,
            )
        )
    strip_terms = tuple((term for term in terms if not term.is_zero))
    if len(strip_terms) != len(terms):
        changed = True

    terms = tuple(terms)
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    if changed:
        return SumOperator(terms, system, simplified=True)
    return sum_operator


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


def reduce_by_orthogonalization(operator_list):
    """
    From a list of operators whose sum spans another operator,
    produce a new list with linear independent terms
    """

    def scalar_product(op_1, op_2):
        return (op_1.dag() * op_2).tr()

    basis = orthogonalize_basis(operator_list, sp=scalar_product)
    if len(basis) > len(operator_list):
        return operator_list
    coeffs = [
        sum(scalar_product(op_b, term) for term in operator_list) for op_b in basis
    ]

    return [op_b * coeff for coeff, op_b in zip(coeffs, basis)]


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
    factor_terms = decompose_qutip_operator(qutip_subop)
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
    nontrivial = s_diag > ALPSQUTIP_TOLERANCE
    u_mat, uh_mat = u_mat[:, nontrivial], uh_mat[nontrivial]
    coeffs = sum(u_mat.dot(uh_mat))
    new_terms = (
        op_i * coeff
        for coeff, op_i in zip(coeffs, basis)
        if abs(coeff) > ALPSQUTIP_TOLERANCE
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


def simplify_sum_operator(operator):
    """
    Try a more aggressive simplification that self.simplify()
    by classifing the terms according to which subsystem acts,
    reducing the partial sums by orthogonalization.
    """
    simplified_op = operator.simplify().flat()

    if isinstance(simplified_op, OneBodyOperator) or not isinstance(
        simplified_op, SumOperator
    ):
        return simplified_op

    operator = simplified_op
    # Now, operator has at least two non-trivial terms.
    operator_terms = operator.terms

    system = operator.system
    isherm = operator._isherm

    terms_by_subsystem = {}
    one_body_terms = []
    scalar_terms = []

    for term in operator_terms:
        assert not isinstance(
            term, SumOperator
        ), f"{type(term)} should not be here. Check simplify."
        assert not isinstance(term, Number), (
            "In a sum, numbers should be represented by "
            f"ScalarOperator's, but {type(term)} was found."
        )

    for term in operator_terms:
        if isinstance(term, LocalOperator):
            one_body_terms.append(term)
        elif isinstance(term, ScalarOperator):
            scalar_terms.append(term)
        else:
            sites = term.acts_over()
            sites = tuple(sites) if sites is not None else None
            terms_by_subsystem.setdefault(sites, []).append(term)

    # Simplify the scalars:
    if len(scalar_terms) > 1:
        assert all(isinstance(t, ScalarOperator) for t in scalar_terms)
        value = sum(value.prefactor for value in scalar_terms)
        scalar_terms = [ScalarOperator(value, system)] if value else []
    elif len(scalar_terms) == 1:
        if scalar_terms[0].prefactor == 0:
            scalar_terms = []

    one_body_terms = (
        [OneBodyOperator(tuple(one_body_terms), system)]
        if len(one_body_terms) != 0
        else []
    )
    new_terms = scalar_terms + one_body_terms

    # Try to reduce the other terms
    for subsystem, block_terms in terms_by_subsystem.items():
        if subsystem is None:
            # Maybe here we should convert block_terms into
            # qutip, add the terms and if the result is not zero,
            # store as a single term
            new_terms.extend(block_terms)
        elif len(subsystem) > 1:
            if len(block_terms) > 1:
                block_terms = reduce_by_orthogonalization(block_terms)
            new_terms.extend(block_terms)
        else:
            # Never reached?
            assert False
            new_terms.extend(block_terms)

    # Build the return value
    if new_terms:
        if len(new_terms) == 1:
            return new_terms[0]
        if not isherm:
            isherm = None
        return SumOperator(tuple(new_terms), system, isherm)
    return ScalarOperator(0.0, system)


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
