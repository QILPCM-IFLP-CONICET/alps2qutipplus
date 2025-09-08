# -*- coding: utf-8 -*-
"""
Functions to simplify sums of operators
"""
import logging
from typing import Callable, Optional, Sequence

from qutip import tensor

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    Operator,
    ProductOperator,
    ScalarOperator,
    empty_op,
)
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.qutip_tools.tools import (
    data_is_diagonal,
    decompose_qutip_operator,
    decompose_qutip_operator_hermitician,
)


def sum_operator_sequence(
    seq: Sequence[Operator], system: SystemDescriptor, **attrs
) -> Operator:
    """Convert a sequence of operators in its sume.

    Parameters
    ----------
    seq : Sequence[Operator]
        A sequence of operators.
    system : SystemDescriptor

    seq: Sequence[Operator] :

    system: SystemDescriptor :

    **attrs :


    Returns
    -------


    """
    if not seq:
        return ScalarOperator(0, system)
    if len(seq) == 1:
        return seq[0]
    return SumOperator(tuple(seq), system=system, **attrs)


def collect_nbody_terms(operator: Operator) -> dict:
    """Build a dictionary whose keys are subsystems and
    the values are lists of operators acting exactly
    over the subsystem.

    Parameters
    ----------
    operator : Operator
        The operator to be decomposed.
    operator: Operator :


    Returns
    -------


    """
    terms_by_block = {}
    scalar_term = 0.0
    system = operator.system

    if not isinstance(operator, SumOperator):
        return {operator.acts_over(): [operator]}

    full_acts_over = frozenset()
    for term in operator.terms:
        acts_over = term.acts_over()
        if acts_over is None:
            acts_over_key = None
            terms_by_block.setdefault(acts_over_key, []).append(term)
            continue

        acts_over_key = acts_over
        assert isinstance(acts_over_key, frozenset)
        if not acts_over_key:
            scalar_term += term.prefactor
        else:
            full_acts_over = full_acts_over.union(acts_over_key)
            terms_by_block.setdefault(acts_over_key, []).append(term)

    if None in terms_by_block:
        terms_by_block.setdefault(full_acts_over, []).extend(terms_by_block.pop(None))

    # Add a scalar term
    if scalar_term:
        terms_by_block[frozenset()] = [ScalarOperator(scalar_term, system)]
    return terms_by_block


def group_terms_by_blocks(operator: Operator, fn: Optional[Callable] = None):
    """Rewrite a sum of operators as a sum
    of a ScalarOperator, a OneBodyOperator
    and terms acting on different blocks.

    For many-body terms, apply fn for further
    simplifications.

    For example
    ```
    group_terms_by_blocks(operator, lambda op:op.to_qutip_operator())
    ```
    convert these many-body terms into Qutip operators,
    which for small blocks could provide a more efficient
    representation.

    Parameters
    ----------
    operator : Operator
        The operator to be reduced.
    fn : Optional[Callable], optional
        A function to implement specific simplifications. The default is None.
    operator: Operator :

    fn: Optional[Callable] :
         (Default value = None)

    Returns
    -------


    """

    if (
        not isinstance(operator, SumOperator)
        or getattr(operator, "_simplified", False)
        or isinstance(operator, OneBodyOperator)
        or hasattr(operator, "expect")
    ):
        return operator

    changed = False
    system = operator.system
    acts_over = operator.acts_over()
    operator_flat = operator.flat()
    isherm = getattr(operator, "_isherm", None)
    if operator is not operator_flat:
        changed = True
    terms_dict = collect_nbody_terms(operator_flat)

    if any(isinstance(t, QutipOperator) for t in terms_dict.get(acts_over, [])):
        return operator.to_qutip_operator()

    new_terms = []
    one_body_terms = []
    scalar_terms = []

    def apply_simplification_fn(op_in: Operator, fn: Optional[Callable]):
        """

        Parameters
        ----------
        op_in: Operator :

        fn: Optional[Callable] :

        Returns
        -------

        """
        try:
            if isinstance(op_in, SumOperator):
                op_in = simplify_qutip_sums(op_in)
            if fn is None:
                return op_in
            return fn(op_in)
        except Exception as exc:
            logging.warning(exc)
            return op_in

    for block, terms in terms_dict.items():
        if block is None:
            new_terms.extend(terms)
        elif len(block) == 0:
            assert all(
                isinstance(term, ScalarOperator) for term in terms
            ), "Should be a scalar term..."
            scalar_terms.extend(terms)
        elif len(block) == 1:
            one_body_terms.extend(terms)
        else:
            if len(terms) == 1:
                new_term = terms[0]
                if isherm and not new_term.isherm:
                    changed = True
                    new_term = (
                        SumOperator(
                            tuple([new_term, new_term.dag()]),
                            system=system,
                            isherm=isherm,
                        )
                        * 0.5
                    )
            else:
                new_term = SumOperator(tuple(terms), system=system, isherm=isherm)

            new_term_simpl = apply_simplification_fn(new_term, fn)
            if new_term_simpl is not new_term:
                changed = True
                new_term = new_term_simpl

            new_terms.append(new_term)

    new_terms = [term for term in new_terms if term]
    if len(scalar_terms) > 1:
        scalar_terms = [sum(scalar_terms)]

    if not new_terms:
        if one_body_terms:
            result = OneBodyOperator(
                tuple(one_body_terms + scalar_terms), system, isherm=isherm
            ).simplify()
            return result
        return scalar_terms[0] if scalar_terms else ScalarOperator(0.0, system)

    if not changed:
        setattr(operator, "_simplified", True)
        if isherm:
            setattr(operator, "_isherm", True)
        return operator

    new_terms = new_terms + scalar_terms

    if one_body_terms:
        new_term = OneBodyOperator(tuple(one_body_terms), system, isherm=isherm)
        new_terms.append(new_term)
        changed = True

    if not new_terms:
        return ScalarOperator(0, system)
    if len(new_terms) == 1:
        return new_terms[0]

    return sum_operator_sequence(
        new_terms,
        system=system,
        isherm=isherm,
        isdiag=getattr(operator, "_isdiagonal", None),
        simplified=True,
    )


def simplify_qutip_sums(sum_operator: SumOperator) -> Operator:
    """Collect terms acting on the same block of sites,
    and reduce it to a single qutip operator.

    Parameters
    ----------
    sum_operator : SumOperator
        The operator to be reduced.
    sum_operator: SumOperator :


    Returns
    -------


    """
    if not isinstance(sum_operator, SumOperator):
        return sum_operator

    changed = False
    isherm = sum_operator._isherm
    system = sum_operator.system
    terms = []
    qutip_terms = {}
    product_terms = {}

    for term in sum_operator.terms:
        if isinstance(term, ProductOperator):
            product_terms.setdefault(term.acts_over(), []).append(term)
        elif isinstance(term, QutipOperator):
            qutip_terms.setdefault(term.acts_over(), []).append(term)
        else:
            terms.append(term)

    # Process first the product operator terms
    for block, p_terms in product_terms.items():
        # If block is in qutip_terms, or there are more than a single
        # product term, and each product term acts on few sites, it is more
        # efficient to store them as a single qutip operator:
        if block in qutip_terms or (len(p_terms) > 1 and len(block) < 6):
            changed = True
            block_tuple = tuple(sorted(block))
            sum_qutip_op = sum(
                term.to_qutip_operator().to_qutip(block_tuple) for term in p_terms
            )
            if not empty_op(sum_qutip_op):
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
        # Reinforce hermiticity if the operator is hermitician, but the term isn't.
        if isherm and not new_qterm.isherm:
            new_qterm = (new_qterm + new_qterm.dag()) * 0.5

        if not empty_op(new_qterm):
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

    if not changed:
        return sum_operator
    return sum_operator_sequence(terms, system=system, simplified=True, isherm=isherm)


def rewrite_nbody_term_using_qutip(
    operator_list: list,
    block: tuple,
    system: SystemDescriptor,
    isherm: bool = None,
    isdiag: bool = None,
) -> Operator:
    """Do the decomposition work using qutip

    Parameters
    ----------
    operator_list: list :

    block: tuple :

    system: SystemDescriptor :

    isherm: bool :
         (Default value = None)
    isdiag: bool :
         (Default value = None)

    Returns
    -------

    """
    block_sites = sorted(block)
    sites_identity = {}

    def op_or_identity(term, site):
        """

        Parameters
        ----------
        term :

        site :


        Returns
        -------

        """
        result = term.sites_op.get(site, None) or sites_identity.get(site, None)
        if result is None:
            result = system.sites[site]["operators"]["identity"]
            sites_identity[site] = result
        return result

    qutip_subop = sum(
        tensor(*(op_or_identity(term, site) for site in block_sites)) * term.prefactor
        for term in operator_list
    )
    if isherm and not qutip_subop.isherm:
        qutip_subop = (qutip_subop + qutip_subop.dag()) * 0.5
    elif isherm is None:
        isherm = qutip_subop.isherm
    if isdiag is None:
        isdiag = data_is_diagonal(qutip_subop.data)
    # Now, decompose the operator again as a sum of n-body terms
    if isherm:
        factor_terms = decompose_qutip_operator(qutip_subop)
    else:
        factor_terms = decompose_qutip_operator_hermitician(qutip_subop)
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
