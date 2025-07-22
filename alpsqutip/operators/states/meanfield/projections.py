"""
Module that implements a meanfield approximation of a Gibbsian state
"""

from functools import reduce
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
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
from alpsqutip.operators.arithmetic import iterable_to_operator
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
from alpsqutip.parallel import (
    DISPATCH_PROJECTION_METHOD_PARALLEL,
    USE_PARALLEL,
    parallel_process_non_dispatched_terms,
)
from alpsqutip.qutip_tools.tools import schmidt_dec_firsts_last_qutip_operator
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

# Alias: the type of the functions that project operators to a n-body sector, relative to a
# given reference state.
ProjectingOperatorFunction = Callable[
    [Operator, int, Optional[DensityOperatorMixin]], Operator
]


# ######    Specialized for one-body   ###################


def _project_product_operator_to_one_body(
    full_operator: ProductOperator,
    sigma_0: Optional[ProductDensityOperator | GibbsProductDensityOperator] = None,
):
    """
    Project a Product operator to the algebra of one-body operators,
    relative to the state `sigma_0`.
    """
    system = full_operator.system
    prefactor = full_operator.prefactor
    if not prefactor:
        return ScalarOperator(0, system)

    sites_op = full_operator.sites_op
    n_sites = len(sites_op)
    if n_sites <= 1:
        return full_operator

    local_ops = tuple(LocalOperator(site, op, system) for site, op in sites_op.items())
    if sigma_0 is None:
        sigma_0 = ProductDensityOperator({}, system=system)
    local_av = compute_operator_expectation_value(local_ops, sigma_0)
    zero_pos = [i for i, val in enumerate(local_av) if not val]
    if zero_pos:
        if len(zero_pos) > 1:
            return ScalarOperator(0, system)
        pos = zero_pos[0]

        prefactor = np.prod(tuple(u for u in local_av if u), initial=prefactor)
        return local_ops[pos] * prefactor
    # all the operators have a non-zero average
    prefactor = np.prod(local_av, initial=prefactor)
    terms = tuple(op * prefactor / op_av for op, op_av in zip(local_ops, local_av))
    return iterable_to_operator(terms, system) + prefactor * (1 - n_sites)


def _project_qutip_operator_to_one_body(
    full_operator: QutipOperator,
    state: Optional[ProductDensityOperator | GibbsProductDensityOperator] = None,
):
    """
    Project `full_operator` over the one-body operator subspace,
    relative to the state `state`.

    The idea is that the projection of a product operator on the one-body
    operator sector can be written as
    $$
    \\Pi(Q_1\\otimes Q_2\\otimes\\ldots) = \\sum_i Tr_{\\overline i}
    Q \\sigma_i - (N-i)\\langle Q\\rangle
    $\\sigma_i = {\\bf 1}_i \\otimes (Tr_{i} \\sigma_i)$
    """
    site_names = full_operator.site_names
    block_size = len(site_names)
    if block_size <= 1:
        return full_operator
    system = full_operator.system
    block = sorted(site_names, key=lambda x: site_names[x])
    operator_qutip = full_operator.to_qutip(block)

    if state is None:
        dimensions = system.dimensions
        dimensions = {site: system.dimensions[site] for site in site_names}
        dimensions_factor = np.prod(list(dimensions.values()))
        meanvalue = operator_qutip.tr() / dimensions_factor
        reduced_ops = (
            [ScalarOperator((1 - block_size) * meanvalue, system)] if meanvalue else []
        )
        reduced_ops.extend(
            [
                LocalOperator(
                    site,
                    operator_qutip.ptrace([site_names[site]])
                    * (dimensions[site] / dimensions_factor),
                    system,
                )
                for site in dimensions
            ]
        )
    else:
        if hasattr(state, "to_product_state"):
            state = state.to_product_state()
        meanvalue = state.expect(full_operator)
        sites_op_state = state.sites_op
        sites_op_state = {key: sites_op_state[key] for key in block}
        reduced_ops = (
            [ScalarOperator((1 - block_size) * meanvalue, system)] if meanvalue else []
        )
        for site in site_names:
            local_identity = system.site_identity(site)
            sigma_i = qutip.tensor(
                [
                    local_identity if c_site == site else sites_op_state[c_site]
                    for c_site in block
                ]
            )
            op_qutip_local = (operator_qutip * sigma_i).ptrace(site_names[site])
            local_term = LocalOperator(site, op_qutip_local, system=system)
            reduced_ops.append(local_term)

    return iterable_to_operator(tuple(reduced_ops), system)


# ####### Specialized for product operators ##############


def _project_product_operator_combinatorial(
    full_operator: Operator,
    nmax: int = 1,
    sigma: Optional[ProductDensityOperator] = None,
) -> Operator:
    """
    Project a product operator to the manifold of n-body operators
    """
    if nmax == 0:
        return ScalarOperator(
            compute_operator_expectation_value(full_operator, sigma),
            full_operator.system,
        )
    if nmax == 1:
        return one_body_product_projection(full_operator, sigma)

    # Trivial case
    src_operator: ProductOperator = cast(ProductOperator, full_operator)
    sites_op = src_operator.sites_op
    n_factors = len(sites_op)
    if n_factors <= nmax:
        return full_operator

    # When we project to a many-body subspace, it is better to use the
    # recursive approach, to discard negligible contributions.
    if nmax > 2:
        return _project_product_operator_recursive(full_operator, nmax, sigma)

    prefactor = src_operator.prefactor
    system = full_operator.system

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

    return iterable_to_operator(terms, system)


def _project_product_operator_recursive(
    full_operator: Operator,
    m_max: int,
    sigma_0: Optional[ProductDensityOperator | GibbsProductDensityOperator],
) -> Operator:
    """
    # reduce op1 (x) op2 (x) op3 ...
    # to <op1> Proj_{m}(op2 (x) op3) +
    #         Delta op1 (x) Proj_{m-1}(op2 (x) op3)
    # and sum the result.

    # Special case: m_max=0
    """
    if m_max == 0:
        return ScalarOperator(
            compute_operator_expectation_value(full_operator, sigma_0),
            full_operator.system,
        )
    if m_max == 1:
        return one_body_product_projection(full_operator, sigma_0)

    # Trivial case
    src_operator: ProductOperator = cast(ProductOperator, full_operator)
    sites_op = src_operator.sites_op
    n_factors = len(sites_op)
    if n_factors <= m_max:
        return full_operator

    # When we project to a few-body subspace, it is better to use the
    # combinatorial approach
    if m_max < 3:
        return _project_product_operator_combinatorial(full_operator, m_max, sigma_0)

    system = full_operator.system

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
        result = delta_op * _project_product_operator_recursive(
            rest_prod_operator, m_max - 1, sigma_rest
        )
    else:
        result = delta_op * compute_operator_expectation_value(
            rest_prod_operator, sigma_rest
        )

    if first_av:
        result = result + first_av * _project_product_operator_recursive(
            rest_prod_operator, m_max, sigma_rest
        )
    return result


# ########################### Specialized qutip  ###################


def _project_qutip_operator_combinatorial(
    full_operator, nmax: int = 1, sigma_ref: Optional[ProductDensityOperator] = None
) -> Operator:
    """
    Project a qutip operator to the manifold of n-body operators
    """
    if nmax == 0:
        return ScalarOperator(
            compute_operator_expectation_value(full_operator, sigma_ref),
            full_operator.system,
        )
    if nmax == 1:
        return project_qutip_to_one_body(full_operator, sigma_ref)

    # Reduce a qutip operator
    site_names = full_operator.site_names
    num_ops = len(site_names)
    if num_ops <= nmax:
        return full_operator
    return project_to_n_body_operator(
        full_operator.as_sum_of_products(), nmax, sigma_ref
    )


def _project_qutip_operator_recursive(
    full_operator: QutipOperator, m_max=2, sigma_0=None
) -> Operator:
    """
    Recursive implementation for the m-body Projection
    over QutipOperators.
    """
    if m_max == 0:
        return ScalarOperator(
            compute_operator_expectation_value(full_operator, sigma_0),
            full_operator.system,
        )
    if m_max == 1:
        return project_qutip_to_one_body(full_operator, sigma_0)

    # Reduce a qutip operator
    site_names = full_operator.site_names
    num_ops = len(site_names)
    if num_ops <= m_max:
        return full_operator

    system = full_operator.system
    if num_ops - m_max > 3:
        return project_qutip_operator_as_n_body_operator(full_operator, m_max, sigma_0)
    if sigma_0 is None:
        sigma_0 = ProductDensityOperator({}, system=system)

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

    return iterable_to_operator(terms, system)


####


def project_quadraticform_operator_as_n_body_operator(
    operator, nmax: Optional[int] = 1, sigma: Optional[ProductDensityOperator] = None
) -> Operator:
    """
    Project a product operator to the manifold of n-body operators
    """
    if nmax != 2:
        project_to_n_body_operator(operator, nmax, sigma)

    linear_term = operator.linear_term
    offset = project_to_n_body_operator(operator.offset, nmax, sigma)
    if offset is operator.offset:
        return operator
    return QuadraticFormOperator(
        operator.basis, operator.weights, operator.system, linear_term, offset
    )


# General routines


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
    one_body_term = one_body_qutip_projection(operator - av, sigma0).simplify()

    # If the one_body_term is a SumOperator, but not a OneBodyOperator,
    # reduce it.
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
                    (
                        f"Got an unexpected type {type(term)} "
                        "for a OneBodyOperator term."
                    )
                )
        one_body_term = OneBodyOperator(
            tuple(local_terms), system, one_body_term.isherm
        )

    remainder: Operator = (
        (operator - one_body_term - scalar_term).simplify().to_qutip_operator()
    )
    return iterable_to_operator(
        (scalar_term, one_body_term, remainder), system, isherm=operator.isherm
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
                    # reduced_state_by_block(term, reduced_states_cache)
                    sigma_0,
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


def _project_monomial(operator, nmax, sigma):
    """
    Apply the projection function specific for the given type of operator.
    """
    return DISPATCH_PROJECTION_METHOD[type(operator)](operator, nmax, sigma).simplify()


def project_sum_operator(
    full_operator: Operator,
    nmax: int,
    sigma: Optional[ProductDensityOperator | GibbsProductDensityOperator] = None,
) -> Operator:
    """
    Project a sum operator
    """
    system = full_operator.system
    terms_tuple: Tuple[Operator] = full_operator.flat().terms

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

    non_dispatched_terms = tuple(
        term for term in terms_tuple if not dispatch_term(term)
    )
    if not non_dispatched_terms:
        return full_operator

    # Process all the terms
    non_dispatched_length = len(non_dispatched_terms)
    if USE_PARALLEL and non_dispatched_length > 10:
        non_dispatched_terms = parallel_process_non_dispatched_terms(
            non_dispatched_terms, nmax, sigma
        )
    else:
        non_dispatched_terms = tuple(
            _project_monomial(term, nmax, sigma) for term in non_dispatched_terms
        )

    terms = list(block_terms.values())

    for term in non_dispatched_terms:
        if isinstance(term, (ScalarOperator, LocalOperator, OneBodyOperator)):
            one_body_terms.append(term)
        elif isinstance(term, SumOperator):
            for sub_term in term.terms:
                assert sub_term.system is system
                dispatch_term(sub_term)
        else:
            if not dispatch_term(term):
                raise TypeError(f"term of type {type(term)} could not be dispatched.")

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

    return iterable_to_operator(terms, system)


def project_to_n_body_operator(
    full_operator: Operator, nmax: int = 1, sigma=None
) -> Operator:
    """
    Approximate `operator` by a sum of (up to) nmax-body
    terms, relative to the state sigma.
    By default, `sigma` is the identity matrix.

    ``operator`` can be a SumOperator or a Product Operator.
    """

    system = full_operator.system
    # Handle the trivial case
    if nmax == 0:
        return ScalarOperator(
            compute_operator_expectation_value(full_operator, sigma), system
        )

    # Special cases: the operator is already a one-body
    # operator.
    if isinstance(full_operator, (OneBodyOperator, LocalOperator)):
        return full_operator

    if isinstance(full_operator, SumOperator):
        return project_sum_operator(full_operator, nmax, sigma)

    return _project_monomial(full_operator, nmax, sigma)


n_body_projection = project_to_n_body_operator
n_body_qutip_projection = _project_qutip_operator_recursive
n_body_product_projection = _project_product_operator_recursive
one_body_product_projection = _project_product_operator_to_one_body
one_body_qutip_projection = _project_qutip_operator_to_one_body


# Deprecated
project_qutip_to_one_body = _project_qutip_operator_to_one_body
project_product_operator_as_n_body_operator = _project_product_operator_combinatorial
_project_product_operator_to_m_body_recursive = _project_product_operator_recursive
project_qutip_operator_as_n_body_operator = _project_qutip_operator_combinatorial
_project_qutip_operator_to_m_body_recursive = _project_qutip_operator_recursive


DISPATCH_PROJECTION_METHOD = {
    ScalarOperator: lambda x, y, z: x,
    ProductOperator: _project_product_operator_to_m_body_recursive,
    QutipOperator: project_qutip_operator_as_n_body_operator,
    QuadraticFormOperator: project_quadraticform_operator_as_n_body_operator,
}

DISPATCH_PROJECTION_METHOD_PARALLEL.update(DISPATCH_PROJECTION_METHOD)
