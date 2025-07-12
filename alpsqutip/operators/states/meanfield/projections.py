"""
Module that implements a meanfield approximation of a Gibbsian state
"""

import logging
from functools import reduce
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import qutip
from qutip import Qobj

from alpsqutip.operators import (
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.quadratic import QuadraticFormOperator
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states.basic import (
    DensityOperatorMixin,
    ProductDensityOperator,
)
from alpsqutip.operators.states.gibbs import (
    GibbsProductDensityOperator,
)
from alpsqutip.operators.states.utils import (
    acts_over_order,
    compute_operator_expectation_value,
)
from alpsqutip.qutip_tools.tools import schmidt_dec_firsts_last_qutip_operator
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

# Alias: the type of the functions that project operators to a n-body sector, relative to a
# given reference state.
ProjectingOperatorFunction = Callable[
    [Operator, int, Optional[DensityOperatorMixin]], Operator
]


def _project_product_operator_to_m_body_recursive(
    full_operator: Operator,
    m_max: int,
    sigma_0: Optional[ProductDensityOperator | GibbsProductDensityOperator],
) -> Operator:
    # reduce op1 (x) op2 (x) op3 ...
    # to <op1> Proj_{m}(op2 (x) op3) +
    #         Delta op1 (x) Proj_{m-1}(op2 (x) op3)
    # and sum the result.

    sites_op = full_operator.sites_op

    if len(sites_op) <= m_max:
        return full_operator
    system = full_operator.system

    # Special case: m_max=0
    if m_max == 0:
        return ScalarOperator(
            compute_operator_expectation_value(full_operator, sigma_0),
            full_operator.system,
        )

    # m_max>0
    first_site, *rest = tuple(sites_op)

    op_first = sites_op[first_site]
    weight_first = op_first
    sigma_rest = sigma_0
    if sigma_0 is not None:
        sigma_rest = sigma_rest.partial_trace(frozenset(rest))
        sigma_first = sigma_0.partial_trace(frozenset({first_site})).to_qutip()
        weight_first = op_first * sigma_first
    else:
        weight_first = weight_first / op_first.dims[0][0]

    first_av = weight_first.tr()
    delta_op = LocalOperator(first_site, op_first - first_av, system)
    sites_op_rest = {site: op for site, op in sites_op.items() if site != first_site}
    rest_prod_operator = ProductOperator(
        sites_op_rest, prefactor=full_operator.prefactor, system=system
    )

    if m_max > 1:
        result = delta_op * _project_product_operator_to_m_body_recursive(
            rest_prod_operator, m_max - 1, sigma_rest
        )
    else:
        result = delta_op * compute_operator_expectation_value(
            rest_prod_operator, sigma_rest
        )

    if first_av:
        result = result + first_av * _project_product_operator_to_m_body_recursive(
            rest_prod_operator, m_max, sigma_rest
        )
    return result


def _project_qutip_operator_to_m_body_recursive(
    full_operator: QutipOperator, m_max=2, sigma_0=None
) -> Operator:
    """
    Recursive implementation for the m-body Projection
    over QutipOperators.
    """
    if full_operator.is_zero:
        return ScalarOperator(0.0, full_operator.system)

    if m_max == 0:
        return ScalarOperator(
            compute_operator_expectation_value(full_operator, sigma_0),
            full_operator.system,
        )

    system = full_operator.system

    if sigma_0 is None:
        sigma_0 = ProductDensityOperator({}, system=system)

    # Reduce a qutip operator
    site_names = full_operator.site_names
    if len(site_names) <= m_max:
        return full_operator

    names = tuple(sorted(site_names, key=lambda s: site_names[s]))
    firsts, last_site = names[:-1], names[-1]
    rest_sitenames = {site: site_names[site] for site in firsts}

    block_qutip_op = full_operator.to_qutip(names)
    qutip_ops_firsts, qutip_ops_last = schmidt_dec_firsts_last_qutip_operator(
        block_qutip_op
    )
    if sigma_0 is None:
        averages = [op_loc.tr() / op_loc.dims[0][0] for op_loc in qutip_ops_last]
        sigma_firsts = None
    else:
        sigma_last_qutip = sigma_0.partial_trace(frozenset({last_site})).to_qutip()
        averages = [qutip.expect(sigma_last_qutip, op_loc) for op_loc in qutip_ops_last]
        sigma_firsts = sigma_0.partial_trace(frozenset(rest_sitenames))

    firsts_ops = [
        QutipOperator(op_c.tidyup(), names=rest_sitenames, system=system)
        for op_c in qutip_ops_firsts
    ]
    delta_ops = [
        LocalOperator(last_site, (op - av).tidyup(), system=system).simplify()
        for av, op in zip(averages, qutip_ops_last)
    ]

    terms = []
    term_index = 0
    for av, delta, firsts_op in zip(averages, delta_ops, firsts_ops):
        term_index += 1
        if abs(av) > ALPSQUTIP_TOLERANCE:
            new_term = _project_qutip_operator_to_m_body_recursive(
                firsts_op, m_max=m_max, sigma_0=sigma_firsts
            )
            new_term = new_term * av
            terms.append(new_term)
        if bool(delta):
            if m_max > 1:
                reduced_op = _project_qutip_operator_to_m_body_recursive(
                    firsts_op, m_max=m_max - 1, sigma_0=sigma_firsts
                )
            else:
                reduced_op = compute_operator_expectation_value(firsts_op, sigma_firsts)
            if reduced_op:
                new_term = delta * reduced_op
                terms.append(new_term)

    if terms:
        if len(terms) == 1:
            return terms[0]
        result = SumOperator(tuple(terms), system).simplify()
        # error_ev = compute_operator_expectation_value(full_operator - result, sigma_0)
        # assert (
        #    abs(error_ev) < ALPSQUTIP_TOLERANCE
        # ), f"The difference should have a vanishing expectation value. Got {error_ev}."
        return result
    return ScalarOperator(0, full_operator.system)


def one_body_from_qutip_operator(
    operator: Union[Operator, Qobj], sigma0: Optional[DensityOperatorMixin] = None
) -> Operator:
    """
    Decompose a qutip operator as a sum of an scalar term,
    a one-body term and a remainder, with
    the one-body term and the remainder having zero mean
    regarding sigma0.

    Parameters
    ----------
    operator : Union[Operator, qutip.Qobj]
        the operator to be decomposed.
    sigma0 : DensityOperatorMixin, optional
        A Density matrix. If None (default) it is assumed to be
        the maximally mixed state.

    Returns
    -------
    SumOperator
        A sum of a Scalar Operator (the expectation value of `operator`
       w.r.t `sigma0`), a LocalOperator and a QutipOperator.

    """
    if isinstance(operator, (ScalarOperator, OneBodyOperator, LocalOperator)):
        return operator

    if isinstance(operator, Qobj):
        if sigma0 is None:
            operator = QutipOperator(operator)
            system = operator.system
        else:
            system = sigma0.system
            operator = QutipOperator(operator, system=system)
    else:
        system = operator.system

    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=system)

    av = sigma0.expect(operator)
    scalar_term: ScalarOperator = ScalarOperator(av, system)
    one_body_term = project_operator_to_m_body(operator - av, 1, sigma0).simplify()

    # If the one_body_term is a SumOperator, but not a OneBodyOperator, reduce it.
    if isinstance(one_body_term, SumOperator) and not isinstance(
        one_body_term, OneBodyOperator
    ):
        one_body_term = one_body_term.flat()
        local_terms: List[Operator] = []
        for term in one_body_term.terms:
            if isinstance(term, LocalOperator):
                local_terms.append(term)
            elif isinstance(term, ScalarOperator):
                local_terms.append(term)
            elif isinstance(term, OneBodyOperator):
                local_terms.extend(term.terms)
            else:
                raise TypeError(
                    f"Got an unexpected type {type(term)} for a OneBodyOperator term."
                )
        one_body_term = OneBodyOperator(
            tuple(local_terms), system, one_body_term.isherm
        )

    remainder: Operator = (
        (operator - one_body_term - scalar_term).simplify().to_qutip_operator()
    )
    return SumOperator(
        (scalar_term, one_body_term, remainder), operator.system, operator.isherm
    )


def project_operator_to_m_body(
    full_operator: Operator, m_max=2, sigma_0=None
) -> Operator:
    """
    Project a Operator onto a m_max - body operators sub-algebra
    relative to the local states `local_sigmas`.
    If `local_sigmas` is not given, maximally mixed states are assumed.
    """
    # Special case: m_max=0
    if m_max == 0:
        return ScalarOperator(
            compute_operator_expectation_value(full_operator, sigma_0),
            full_operator.system,
        )
    if m_max > 0:
        # Special cases: m_max>0, and the operator is already a one-body
        # operator.
        if isinstance(full_operator, (OneBodyOperator, LocalOperator)):
            return full_operator

        acts_over = full_operator.acts_over()
        if acts_over is not None:
            if len(acts_over) <= m_max:
                return full_operator
            if sigma_0 is not None:
                sigma_0 = sigma_0.partial_trace(acts_over)

    # Special case: m=0, implies that the operator is reduced to its
    # expectation value.

    full_operator = full_operator.simplify()

    if isinstance(full_operator, SumOperator):
        system = full_operator.system

        # Key to order the terms by the size of the block
        # where they act.

        # Now, we cache the local states. Each term is projected using the
        # corresponding local state.
        # reduced_states_cache = {None: sigma_0}
        terms = tuple(
            (
                project_operator_to_m_body(
                    term,
                    m_max,
                    sigma_0,  # reduced_state_by_block(term, reduced_states_cache)
                )
                for term in sorted(full_operator.terms, key=acts_over_order)
            )
        )
        if len(terms) == 0:
            return ScalarOperator(0, system)
        if len(terms) == 1:
            return terms[0]
        if len(full_operator.terms) == len(terms) and all(
            t1 is t2 for t1, t2 in zip(full_operator.terms, terms)
        ):
            return full_operator
        return SumOperator(terms, system).simplify()

    if isinstance(full_operator, ProductOperator):
        return _project_product_operator_to_m_body_recursive(
            full_operator, m_max, sigma_0
        )

    if isinstance(full_operator, QutipOperator):
        return _project_qutip_operator_to_m_body_recursive(
            full_operator, m_max, sigma_0
        )

    return _project_qutip_operator_to_m_body_recursive(
        full_operator.to_qutip_operator(), m_max, sigma_0
    )


def project_product_operator_as_n_body_operator(
    operator: Operator,
    nmax: int = 1,
    sigma: Optional[ProductDensityOperator] = None,
) -> Operator:
    """
    Project a product operator to the manifold of n-body operators
    """
    # Trivial case
    src_operator: ProductOperator = cast(ProductOperator, operator)
    sites_op = src_operator.sites_op
    prefactor = src_operator.prefactor
    system = operator.system
    if prefactor == 0.0:
        return ScalarOperator(0, system)

    if len(sites_op) <= nmax:
        return operator

    def mul_func(x, y):
        return x * y

    if sigma is None:
        sigma = ProductDensityOperator({}, system=system)

    terms = []
    averages: Dict[str, Operator] = cast(
        Dict[str, Operator],
        sigma.expect(
            {site: LocalOperator(site, l_op, system) for site, l_op in sites_op.items()}
        ),
    )
    fluct_op = {site: l_op - averages[site] for site, l_op in sites_op.items()}
    # Now, we run a loop over
    for n_factors in range(nmax + 1):
        # subterms = terms_by_factors.setdefault(n_factors, [])
        for subcomb in combinations(sites_op, n_factors):
            num_factors = (val for site, val in averages.items() if site not in subcomb)
            term_prefactor = reduce(mul_func, num_factors, prefactor)
            if term_prefactor == 0:
                continue
            sub_site_ops = {site: fluct_op[site] for site in subcomb}
            terms.append(ProductOperator(sub_site_ops, term_prefactor, system))

    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(tuple(terms), system)


def project_quadraticform_operator_as_n_body_operator(
    operator, nmax: Optional[int] = 1, sigma: Optional[ProductDensityOperator] = None
) -> Operator:
    """
    Project a product operator to the manifold of n-body operators
    """
    from alpsqutip.operators.quadratic import QuadraticFormOperator

    if nmax != 2:
        project_to_n_body_operator(operator, nmax, sigma)

    linear_term = operator.linear_term
    offset = project_to_n_body_operator(operator.offset, nmax, sigma)
    if offset is operator.offset:
        return operator
    return QuadraticFormOperator(
        operator.basis, operator.weights, operator.system, linear_term, offset
    )


def project_qutip_operator_as_n_body_operator(
    operator, nmax: int = 1, sigma_ref: Optional[ProductDensityOperator] = None
) -> Operator:
    """
    Project a qutip operator to the manifold of n-body operators
    """
    acts_over = operator.acts_over()
    if acts_over is not None and len(cast(frozenset, acts_over)) <= nmax:
        return operator

    if nmax == 0:
        return ScalarOperator(
            compute_operator_expectation_value(operator, sigma_ref),
            operator.system,
        )

    system = operator.system
    sigma: ProductDensityOperator
    if sigma_ref is None:
        sigma = ProductDensityOperator({}, system=system)
    else:
        sigma = sigma_ref

    operator = operator.as_sum_of_products()

    terms_by_block: Dict[Optional[frozenset], List[Operator]] = {}
    one_body_terms: List[Operator] = []
    scalar: complex = 0.0
    # local_states_cache = {None: sigma}

    for term in (
        sorted(operator.terms, key=acts_over_order)
        if isinstance(operator, SumOperator)
        else (operator,)
    ):
        acts_over = term.acts_over()
        # assert isinstance(
        #    acts_over, frozenset
        # ), f"{type(term)}.acts_over() should return a frozenset. Got({type(acts_over)})"
        block_size = len(acts_over)
        if block_size == 0:
            scalar += term.prefactor
            continue
        elif block_size == 1:
            one_body_terms.append(term.simplify())
            continue
        elif block_size <= nmax:
            terms_by_block.setdefault(acts_over, []).append(term)
            continue

        #  project_product_operator_as_n_body_operator
        term = _project_product_operator_to_m_body_recursive(
            cast(ProductOperator, term),
            nmax,
            sigma,  # reduced_state_by_block(term, local_states_cache),
        )  # .simplify()
        if isinstance(term, OneBodyOperator):
            one_body_terms.append(term)
        elif isinstance(term, SumOperator):
            for sub_term in term.terms:
                acts_over_subterm = sub_term.acts_over()
                if isinstance(sub_term, (OneBodyOperator, LocalOperator)) or (
                    acts_over_subterm is not None and len(acts_over_subterm) < 2
                ):
                    one_body_terms.append(sub_term)
                else:
                    terms_by_block.setdefault(sub_term.acts_over(), []).append(
                        sub_term.to_qutip_operator()
                    )
        else:
            term_acts_over2 = term.acts_over()
            if len(term_acts_over2) > -1:
                terms_by_block.setdefault(term_acts_over2, []).append(
                    term.to_qutip_operator()
                )
            else:
                terms_by_block.setdefault(term_acts_over2, []).append(term)

    terms_list: List[Operator] = []
    if scalar:
        terms_list.append(ScalarOperator(scalar, system))
    if one_body_terms:
        terms_list.append(cast(Operator, sum(one_body_terms)).simplify())
    for block, block_terms in terms_by_block.items():
        if block_terms:
            try:
                terms_list.append(SumOperator(tuple(block_terms), system))
            except Exception as e:
                logging.error(e)

    if len(terms_list) == 0:
        return ScalarOperator(0, system)
    if len(terms_list) == 1:
        return terms_list[0]
    return SumOperator(tuple(terms_list), system)


def project_to_n_body_operator(operator, nmax=1, sigma=None) -> Operator:
    """
    Approximate `operator` by a sum of (up to) nmax-body
    terms, relative to the state sigma.
    By default, `sigma` is the identity matrix.

    ``operator`` can be a SumOperator or a Product Operator.
    """

    terms_tuple: Tuple[Operator]
    system = operator.system
    # Handle the trivial case
    if nmax == 0:
        return ScalarOperator(
            compute_operator_expectation_value(operator, sigma), system
        )

    # Special cases: the operator is already a one-body
    # operator.
    if isinstance(operator, (OneBodyOperator, LocalOperator)):
        return operator

    acts_over = operator.acts_over()
    if acts_over is not None and len(acts_over) <= nmax:
        return operator

    untouched_operator = operator

    if isinstance(operator, SumOperator):
        operator = operator.simplify().flat()
    # If still a sum operator
    if isinstance(operator, SumOperator):
        terms_tuple = operator.terms
        if sigma is None:
            sigma = ProductDensityOperator({}, 1, system=system)
        if hasattr(sigma, "to_product_state"):
            if acts_over is None or len(acts_over) >= 10:
                sigma = sigma.to_product_state()
    else:
        terms_tuple = (operator,)

    changed = False
    one_body_terms = []
    block_terms: Dict[Optional[frozenset], Operator] = {}

    def dispatch_term(t):
        """
        If t is a nbody-term acting on not more than
        nmax sites, stores in the proper place and return True.
        Otherwise, return False.
        """
        if isinstance(t, OneBodyOperator):
            one_body_terms.append(t)
            return True
        acts_over_t = t.acts_over()
        # assert isinstance(
        #    acts_over_t, frozenset
        # ), f"{type(t)}.acts_over() should return a frozenset. Got({type(acts_over_t)})"
        n_body_sector = len(acts_over_t)
        if n_body_sector <= 1:
            one_body_terms.append(t)
            return True
        if n_body_sector <= nmax:
            if acts_over_t in block_terms:
                block_terms[acts_over_t] = (
                    block_terms[acts_over_t].to_qutip_operator() + t.to_qutip_operator()
                )
            else:
                block_terms[acts_over_t] = t
            return True
        return False

    dispatch_project_method = {
        # ProductOperator: project_product_operator_as_n_body_operator,
        ProductOperator: _project_product_operator_to_m_body_recursive,
        QutipOperator: project_qutip_operator_as_n_body_operator,
        # QutipOperator: _project_qutip_operator_to_m_body_recursive,
        QuadraticFormOperator: project_quadraticform_operator_as_n_body_operator,
    }

    for term in terms_tuple:
        if dispatch_term(term):
            continue
        changed = True
        try:
            term = dispatch_project_method[type(term)](term, nmax, sigma)
        except KeyError:
            raise TypeError(f"{type(term)} not in {dispatch_project_method.keys()}")

        if isinstance(term, (ScalarOperator, LocalOperator, OneBodyOperator)):
            one_body_terms.append(term)
        elif isinstance(term, SumOperator):
            for sub_term in term.terms:
                dispatch_term(sub_term)
        else:
            if not dispatch_term(term):
                raise TypeError(f"term of type {type(term)} could not be dispatched.")

    if not changed:
        return untouched_operator

    scalar = sum(
        term.prefactor for term in one_body_terms if isinstance(term, ScalarOperator)
    )
    proper_local_terms = tuple(
        (term for term in one_body_terms if not isinstance(term, ScalarOperator))
    )

    terms: List[Operator] = list(block_terms.values())
    if scalar != 0:
        terms.append(ScalarOperator(scalar, system))
    if proper_local_terms:
        terms.append(sum(proper_local_terms).simplify())

    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(tuple(terms), system)
