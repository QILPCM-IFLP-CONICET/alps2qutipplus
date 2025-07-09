from collections import defaultdict
from functools import lru_cache, reduce
from itertools import combinations
from operator import mul
from typing import Dict, FrozenSet, Optional

from alpsqutip.operators.arithmetic import (
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)

from .operators.states.gibbs import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    ProductDensityOperator,
)

### locally defined functions and operator classes


def precompute_sigma_reductions(sigma, max_support_size):
    """
    Precompute partial traces of sigma onto all subsets of the system
    up to a given size.
    """
    sites = list(sigma.system.sites)
    reductions = {}
    for k in range(1, max_support_size + 1):
        for subset in combinations(sites, k):
            subset = frozenset(subset)
            traced_out = frozenset(sites) - subset
            reductions[subset] = sigma.partial_trace(traced_out)
    return reductions


def cached_partial_trace(op, traced_out, sigma_reductions):
    """
    Cached version of partial trace using precomputed sigma reductions.
    """
    key = (id(op), frozenset(traced_out))
    if key in _partial_trace_cache:
        return _partial_trace_cache[key]
    sigma_reduced = sigma_reductions.get(
        frozenset(op.acts_over()) - frozenset(traced_out)
    )
    result = op.partial_trace(traced_out, sigma_reduced)
    _partial_trace_cache[key] = result
    return result


_expectation_cache = {}


def cached_sigma_expect(sigma, site_ops_dict):
    """
    Cache sigma.expect(...) for given site-wise local operators.
    Keyed by (id(sigma), tuple of (site, id(op))) — assumes operator instances reused.
    """
    key = (id(sigma), tuple((site, id(op)) for site, op in site_ops_dict.items()))
    if key in _expectation_cache:
        return _expectation_cache[key]
    result = sigma.expect(site_ops_dict)
    _expectation_cache[key] = result
    return result


### projection functions for different types of data


def opt_project_product_operator_as_n_body_operator(
    operator: ProductOperator,
    nmax: Optional[int] = 1,
    sigma: Optional[ProductDensityOperator] = None,
) -> Operator:
    """
    Project a product operator to the manifold of n-body operators
    """
    sites_op = operator.sites_op
    prefactor = operator.prefactor
    system = operator.system

    if prefactor == 0.0:
        return ScalarOperator(0, system)

    if len(sites_op) <= nmax:
        return operator

    if sigma is None:
        sigma = ProductDensityOperator({}, system=system)

    # Get expectation values with caching
    local_ops = {
        site: LocalOperator(site, l_op, system) for site, l_op in sites_op.items()
    }
    averages = cached_sigma_expect(sigma, local_ops)

    # Compute fluctuation operators (op - ⟨op⟩)
    fluct_op = {site: l_op - averages[site] for site, l_op in sites_op.items()}

    terms = []
    site_list = list(sites_op.keys())
    for n_factors in range(nmax + 1):
        for subcomb in combinations(site_list, n_factors):
            # Multiply expectation values of the sites NOT in subcomb
            complement_sites = [s for s in site_list if s not in subcomb]
            num_factors = (averages[s] for s in complement_sites)
            term_prefactor = reduce(mul, num_factors, prefactor)
            if term_prefactor == 0:
                continue
            sub_site_ops = {site: fluct_op[site] for site in subcomb}
            terms.append(ProductOperator(sub_site_ops, term_prefactor, system))

    if not terms:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(tuple(terms), system)


def opt_project_qutip_operator_as_n_body_operator(
    operator, nmax: Optional[int] = 1, sigma: Optional["ProductDensityOperator"] = None
) -> "Operator":
    """
    Project a qutip operator to the manifold of n-body operators.
    """
    acts_over = operator.acts_over()
    assert isinstance(acts_over, frozenset)

    if len(acts_over) <= nmax:
        return operator

    system = operator.system
    if sigma is None:
        sigma = ProductDensityOperator({}, system=system)

    operator = operator.as_sum_of_products()  # QutipOperator → Sum[ProductOperator]
    terms_by_block = {}
    one_body_terms = []
    scalar = 0

    terms = operator.terms if isinstance(operator, SumOperator) else (operator,)

    for term in terms:
        acts_over = term.acts_over()
        assert isinstance(acts_over, frozenset)
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

        # Delegate to the now optimized project_product_operator
        term = opt_project_product_operator_as_n_body_operator(
            term, nmax, sigma
        ).simplify()

        if isinstance(term, OneBodyOperator):
            one_body_terms.append(term)
        elif isinstance(term, SumOperator):
            for sub_term in term.terms:
                sub_support = sub_term.acts_over()
                if (
                    isinstance(sub_term, (OneBodyOperator, LocalOperator))
                    or len(sub_support) < 2
                ):
                    one_body_terms.append(sub_term)
                else:
                    terms_by_block.setdefault(sub_support, []).append(
                        sub_term.to_qutip_operator()
                    )
        else:
            term_support = term.acts_over()
            terms_by_block.setdefault(term_support, []).append(term.to_qutip_operator())

    terms_list = []
    if scalar:
        terms_list.append(ScalarOperator(scalar, system))
    if one_body_terms:
        terms_list.append(sum(one_body_terms).simplify())
    for block, block_terms in terms_by_block.items():
        if block_terms:
            try:
                terms_list.append(SumOperator(tuple(block_terms), system))
            except Exception as e:
                print(f"Error forming SumOperator over block {block}: {e}")

    if len(terms_list) == 0:
        return ScalarOperator(0, system)
    if len(terms_list) == 1:
        return terms_list[0]
    return SumOperator(tuple(terms_list), system)


def opt_project_to_n_body_operator(operator, nmax=1, sigma=None) -> Operator:
    """
    Approximate operator by a sum of (up to) nmax-body
    terms, relative to the state sigma.
    By default, sigma is the identity matrix.
    """
    from alpsqutip.operators.quadratic import QuadraticFormOperator

    system = operator.system
    if sigma is None:
        sigma = ProductDensityOperator({}, system=system)

    if nmax == 0:
        return ScalarOperator(sigma.expect(operator), system)

    untouched_operator = operator

    if isinstance(operator, SumOperator):
        operator = operator.simplify().flat()
    terms = operator.terms if isinstance(operator, SumOperator) else (operator,)

    changed = False
    one_body_terms = []
    block_terms = {}

    def dispatch_term(t):
        if isinstance(t, OneBodyOperator):
            one_body_terms.append(t)
            return True
        acts_over_t = t.acts_over()
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
        ProductOperator: opt_project_product_operator_as_n_body_operator,
        QutipOperator: opt_project_qutip_operator_as_n_body_operator,
    }

    for term in terms:
        if dispatch_term(term):
            continue
        changed = True
        try:
            project_fn = dispatch_project_method[type(term)]
        except KeyError:
            raise TypeError(f"{type(term)} not handled.")
        projected = project_fn(term, nmax, sigma)

        if isinstance(projected, (ScalarOperator, LocalOperator, OneBodyOperator)):
            one_body_terms.append(projected)
        elif isinstance(projected, SumOperator):
            for sub_term in projected.terms:
                dispatch_term(sub_term)
        else:
            if not dispatch_term(projected):
                raise TypeError(f"Term {projected} could not be dispatched.")

    if not changed:
        return untouched_operator

    scalar = sum(
        term.prefactor for term in one_body_terms if isinstance(term, ScalarOperator)
    )
    proper_local_terms = [
        term for term in one_body_terms if not isinstance(term, ScalarOperator)
    ]

    terms = list(block_terms.values())
    if scalar != 0:
        terms.append(ScalarOperator(scalar, system))
    if proper_local_terms:
        terms.append(sum(proper_local_terms).simplify())

    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(tuple(terms), system)
