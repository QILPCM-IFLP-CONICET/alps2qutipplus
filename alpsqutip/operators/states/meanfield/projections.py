"""
Module that implements a meanfield approximation of a Gibbsian state
"""

from functools import reduce
from itertools import combinations
from typing import Optional, Tuple, Union

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
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states.basic import (
    DensityOperatorMixin,
    ProductDensityOperator,
)
from alpsqutip.qutip_tools.tools import schmidt_dec_firsts_last_qutip_operator
from alpsqutip.settings import ALPSQUTIP_TOLERANCE


def one_body_from_qutip_operator(
    operator: Union[Operator, Qobj], sigma0: Optional[DensityOperatorMixin] = None
) -> SumOperator:
    """
    Decompose a qutip operator as a sum of an scalar term,
    a one-body term and a remainder, with
    the one-body term and the reamainder having zero mean
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

    # Determine the system and ensure that operator is a QutipOperator.

    system = sigma0.system if sigma0 is not None else None
    if isinstance(operator, Qobj):
        operator = QutipOperator(operator, system=system)

    if system is None:
        system = operator.system

    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=system)

    site_names = operator.site_names
    subsystem = system.subsystem(frozenset(site_names))

    # Reduce the problem to the subsystem where operator acts:
    sigma0 = sigma0.partial_trace(subsystem)
    operator = QutipOperator(
        operator.to_qutip(tuple()), names=site_names, system=subsystem
    )

    # Scalar term
    scalar_term_value = sigma0.expect(operator)
    scalar_term = ScalarOperator(scalar_term_value, system)
    if scalar_term_value != 0:
        operator = operator - scalar_term_value

    # One-body terms
    local_states = {
        name: sigma0.partial_trace(frozenset((name,))).to_qutip() for name in site_names
    }

    local_terms = []
    for name in local_states:
        # Build a product operator Sigma_compl
        # s.t. Tr_{i}Sigma_i =Tr_i sigma0
        #      Tr{/i} Sigma_i = Id
        # Then, for any local operators q_i, q_j s.t.
        # Tr[q_i sigma0]= Tr[q_j sigma0]=0,
        # Tr_{/i}[q_i  Sigma_compl] = q_i
        # Tr_{/i}[q_j  Sigma_compl] = 0
        # Tr_{/i}[q_i q_j Sigma_compl] = 0
        block: Tuple[str] = (name,)
        sigma_compl_factors = {
            name_loc: s_loc
            for name_loc, s_loc in local_states.items()
            if name != name_loc
        }
        sigma_compl = ProductOperator(
            sigma_compl_factors,
            system=system,
        )
        local_term = (sigma_compl * operator).partial_trace(frozenset(block))
        # Split the zero-average part from the average

        if isinstance(local_term, ScalarOperator):
            assert (
                abs(local_term.prefactor) < ALPSQUTIP_TOLERANCE
            ), f"{abs(local_term.prefactor)} should be 0."
        else:
            local_term_qutip = local_term.to_qutip(block)
            local_average = (local_term_qutip * local_states[name]).tr()
            assert (
                abs(local_average) < ALPSQUTIP_TOLERANCE
            ), f"{abs(local_average)} should be 0."
            local_terms.append(LocalOperator(name, local_term_qutip, system))

    one_body_term = OneBodyOperator(tuple(local_terms), system=system)
    # Comunte the remainder of the opertator
    remaining_qutip = operator.to_qutip(tuple(site_names)) - one_body_term.to_qutip(
        tuple(site_names)
    )
    remaining = QutipOperator(
        remaining_qutip,
        system=system,
        names={name: pos for pos, name in enumerate(site_names)},
    )
    return SumOperator(
        tuple(
            (
                scalar_term,
                one_body_term,
                remaining,
            )
        ),
        system,
    )


def project_operator_to_m_body(
    full_operator: Operator, m_max=2, sigma_0=None
) -> Operator:
    """
    Project a Operator onto a m_max - body operators sub-algebra
    relative to the local states `local_sigmas`.
    If `local_sigmas` is not given, maximally mixed states are assumed.
    """
    assert sigma_0 is None or hasattr(sigma_0, "expect"), f"{type(sigma_0)} invalid"
    if m_max == 0:
        if sigma_0:
            return ScalarOperator(sigma_0.expect(full_operator), full_operator.system)
        return ScalarOperator(full_operator.tr(), full_operator.system)

    if (isinstance(full_operator, OneBodyOperator)) or (
        len(full_operator.acts_over()) <= m_max
    ):
        return full_operator

    full_operator = full_operator.simplify()
    system = full_operator.system
    if isinstance(full_operator, SumOperator):
        terms = tuple(
            (
                project_operator_to_m_body(term, m_max, sigma_0)
                for term in full_operator.terms
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
        # reduce op1 (x) op2 (x) op3 ...
        # to <op1> Proj_{m}(op2 (x) op3) +
        #         Delta op1 (x) Proj_{m-1}(op2 (x) op3)
        # and sum the result.
        sites_op = full_operator.sites_op
        if len(sites_op) <= m_max:
            return full_operator

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
        sites_op_rest = {
            site: op for site, op in sites_op.items() if site != first_site
        }
        rest_prod_operator = ProductOperator(
            sites_op_rest, prefactor=full_operator.prefactor, system=system
        )

        result = delta_op * project_operator_to_m_body(
            rest_prod_operator, m_max - 1, sigma_rest
        )
        if first_av:
            result = result + first_av * project_operator_to_m_body(
                rest_prod_operator, m_max, sigma_rest
            )
        result = result.simplify()
        return result

    if isinstance(full_operator, QutipOperator):
        project_qutip_operator_to_m_body(full_operator, m_max, sigma_0)

    return project_qutip_operator_to_m_body(
        full_operator.to_qutip_operator(), m_max, sigma_0
    )


def project_qutip_operator_to_m_body(
    full_operator: Operator, m_max=2, sigma_0=None
) -> Operator:
    """
    Specialized version for QutipOperators.
    """
    system = full_operator.system
    if full_operator.is_zero:
        return ScalarOperator(0, system)
    assert sigma_0 is None or hasattr(sigma_0, "expect"), f"{type(sigma_0)} is invalid."
    if sigma_0 is None:
        sigma_0 = ProductDensityOperator({}, system=system)
    assert sigma_0 is None or hasattr(sigma_0, "expect"), f"{type(sigma_0)} is invalid."
    if m_max == 0:
        return ScalarOperator(sigma_0.expect(full_operator), system)

    # Reduce a qutip operator
    site_names = full_operator.site_names
    if len(site_names) < 2:
        return full_operator

    names = tuple(sorted(site_names, key=lambda s: site_names[s]))
    firsts, last_site = names[:-1], names[-1]
    rest_sitenames = {site: site_names[site] for site in firsts}

    qutip_ops_firsts, qutip_ops_last = schmidt_dec_firsts_last_qutip_operator(
        full_operator.to_qutip(names)
    )

    sigma_last_qutip = sigma_0.partial_trace(frozenset({last_site})).to_qutip()
    averages = [qutip.expect(sigma_last_qutip, op_loc) for op_loc in qutip_ops_last]
    sigma_firsts = sigma_0.partial_trace(frozenset(rest_sitenames))
    assert hasattr(
        sigma_firsts, "expect"
    ), f"{type(sigma_0)}->{type(sigma_firsts)} is invalid."

    firsts_ops = [
        QutipOperator(op_c, names=rest_sitenames, system=system)
        for op_c in qutip_ops_firsts
    ]
    delta_ops = [
        LocalOperator(last_site, op - av, system=system)
        for av, op in zip(averages, qutip_ops_last)
    ]

    terms = []
    for av, delta, firsts_op in zip(averages, delta_ops, firsts_ops):
        reduced_op = project_operator_to_m_body(
            firsts_op, m_max=m_max - 1, sigma_0=sigma_firsts
        )
        if reduced_op:
            new_term = delta * reduced_op
            terms.append(new_term)
        if not av:
            continue

        new_term = project_qutip_operator_to_m_body(
            firsts_op, m_max=m_max, sigma_0=sigma_firsts
        )
        terms.append(ScalarOperator(av, system) * new_term)

    if terms:
        if len(terms) == 1:
            return terms[0]
        return SumOperator(tuple(terms), system).simplify()
    return ScalarOperator(0, full_operator.system)


def project_product_operator_as_n_body_operator(
    operator: ProductOperator,
    nmax: Optional[int] = 1,
    sigma: Optional[ProductDensityOperator] = None,
) -> Operator:
    """
    Project a product operator to the manifold of n-body operators
    """
    # Trivial case
    sites_op = operator.sites_op
    prefactor = operator.prefactor
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
    averages = sigma.expect(
        {site: LocalOperator(site, l_op, system) for site, l_op in sites_op.items()}
    )
    fluct_op = {site: l_op - averages[site] for site, l_op in sites_op.items()}
    # Now, we run a loop over
    for n_factors in range(nmax + 1):
        # subterms = terms_by_factors.setdefault(n_factors, [])
        for subcomb in combinations(sites_op, n_factors):
            num_factors = (val for site, val in averages.items() if site not in subcomb)
            term_prefactor = reduce(mul_func, num_factors, prefactor)
            if prefactor == 0:
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
    operator, nmax: Optional[int] = 1, sigma: Optional[ProductDensityOperator] = None
) -> Operator:
    """
    Project a qutip operator to the manifold of n-body operators
    """
    acts_over = operator.acts_over()
    if len(acts_over) <= nmax:
        return operator

    system = operator.system
    if sigma is None:
        sigma = ProductDensityOperator({}, system=system)

    operator = operator.as_sum_of_products()
    terms = operator.terms if isinstance(operator, SumOperator) else (operator,)
    terms_by_block = {}
    one_body_terms = []
    scalar = 0
    for term in terms:
        acts_over = term.acts_over()
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

        term = project_product_operator_as_n_body_operator(term, nmax, sigma)
        if isinstance(term, OneBodyOperator):
            one_body_terms.append(term)
        elif isinstance(term, SumOperator):
            for sub_term in term.terms:
                terms_by_block.setdefault(sub_term.acts_over(), []).append(
                    sub_term.to_qutip_operator()
                )
        else:
            terms_by_block.setdefault(term.acts_over(), []).append(
                term.to_qutip_operator()
            )

    terms = []
    if scalar:
        terms.append(ScalarOperator(scalar, system))
    if one_body_terms:
        terms.append(sum(one_body_terms).simplify())
    for block, block_terms in terms_by_block.items():
        if block_terms:
            terms.append(sum(block_terms))

    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(tuple(terms), system)


def project_to_n_body_operator(operator, nmax=1, sigma=None) -> Operator:
    """
    Approximate `operator` by a sum of (up to) nmax-body
    terms, relative to the state sigma.
    By default, `sigma` is the identity matrix.

    ``operator`` can be a SumOperator or a Product Operator.
    """
    from alpsqutip.operators.quadratic import QuadraticFormOperator

    system = operator.system
    if sigma is None:
        sigma = ProductDensityOperator({}, system=system)
    # Handle the trivial case
    if nmax == 0:
        return ScalarOperator(sigma.expect(operator), system)

    untouched_operator = operator

    if isinstance(operator, SumOperator):
        operator = operator.simplify().flat()
    # If still a sum operator
    if isinstance(operator, SumOperator):
        terms = operator.terms
    else:
        terms = (operator,)

    changed = False
    one_body_terms = []
    block_terms = {}

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
        ProductOperator: project_product_operator_as_n_body_operator,
        QutipOperator: project_qutip_operator_as_n_body_operator,
        QuadraticFormOperator: project_quadraticform_operator_as_n_body_operator,
    }

    for term in terms:
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
            dispatch_term(term)

    if not changed:
        return untouched_operator

    scalar = sum(
        term.prefactor for term in one_body_terms if isinstance(term, ScalarOperator)
    )
    proper_local_terms = tuple(
        (term for term in one_body_terms if not isinstance(term, ScalarOperator))
    )

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
