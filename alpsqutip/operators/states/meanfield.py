"""
Module that implements a meanfield approximation of a Gibbsian state
"""

# import logging
from functools import reduce
from itertools import combinations
from typing import Optional, Tuple, Union

import numpy as np
import qutip
from qutip import Qobj
from scipy.optimize import minimize_scalar

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
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator
from alpsqutip.qutip_tools.tools import schmidt_dec_firsts_last_qutip_operator


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
                abs(local_term.prefactor) < 1e-6
            ), f"{abs(local_term.prefactor)} should be 0."
        else:
            local_term_qutip = local_term.to_qutip(block)
            local_average = (local_term_qutip * local_states[name]).tr()
            assert abs(local_average) < 1e-9, f"{abs(local_average)} should be 0."
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


def project_meanfield(k_op, sigma0=None, max_it=100):
    """
    Look for a one-body operator kmf s.t
    Tr (k_op-kmf)exp(-kmf)=0

    following a self-consistent, iterative process
    assuming that exp(-kmf)~sigma0

    If sigma0 is not provided, sigma0 is taken as the
    maximally mixed state.

    """
    sigma0 = self_consistent_project_meanfield(k_op, sigma0, max_it)[1]

    return project_operator_to_m_body(k_op, 1, sigma0)


def self_consistent_project_meanfield(
    k_op, sigma=None, max_it=100
) -> Tuple[Operator, Operator]:
    """
    Iteratively computes the one-body component from a QuTip operator and state
    using a self-consistent Mean-Field Projection (MF).

    Parameters:
        k_op: The initial operator, a QuTip.Qobj, to be decomposed into
        one-body components.
        sigma: The referential state to be used as the initial guess
               in the calculations.
        k_0: if given, the logarithm of sigma.
        max_it: Maximum number of iterations.

    Returns:
        A tuple (K_one_body, sigma_one_body):
        - K_one_body: The one-body component of the operator K, an
        AlpsQuTip.one_body_operator object.
        - sigma_one_body: The one-body state normalized through the
        MFT process.
    """
    if sigma is None:
        sigma = GibbsProductDensityOperator(k={}, system=k_op.system)
        neg_log_sigma = -sigma.logm()
    else:
        neg_log_sigma = -sigma.logm()
        if not isinstance(sigma, GibbsProductDensityOperator):
            sigma = GibbsProductDensityOperator(neg_log_sigma)

    rel_s = 10000

    for it in range(max_it):
        k_one_body = project_operator_to_m_body(k_op, 1, sigma)
        new_sigma = GibbsProductDensityOperator(k_one_body)
        k_one_body = -new_sigma.logm()
        rel_s_new = np.real(sigma.expect(k_op - k_one_body))
        rel_entropy_txt = f"     S(curr||target)={rel_s_new}"
        # logging.debug(rel_entropy_txt)
        print(rel_entropy_txt)
        if it > 20 and rel_s_new > 2 * rel_s:
            break
        rel_s = rel_s_new
        sigma = new_sigma

    return k_one_body, sigma


def project_operator_to_m_body(full_operator: Operator, m_max=2, sigma_0=None):
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

    system = full_operator.system
    if isinstance(full_operator, SumOperator):
        terms = tuple(
            (
                project_operator_to_m_body(term, m_max, sigma_0)
                for term in full_operator.terms
            )
        )
        return SumOperator(terms, system).simplify()

    if isinstance(full_operator, ProductOperator):
        # reduce op1 (x) op2 (x) op3 ...
        # to <op1> Proj_{m}(op2 (x) op3) +
        #         Delta op1 (x) Proj_{m-1}(op2 (x) op3)
        # and sum the result.
        sites_op = full_operator.sites_op
        first_site, *rest = tuple(sites_op)
        op_first = sites_op[first_site]
        weight_first = op_first
        sigma_rest = sigma_0
        if sigma_0 is not None:
            sigma_rest = sigma_rest.partial_trace(frozenset(rest))
            sigma_first = sigma_0.partial_trace(frozenset({first_site})).to_qutip()
            weight_first = op_first * sigma_first
        else:
            weight_first = weight_first / op_first.dimensions[0][0]

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


def project_qutip_operator_to_m_body(full_operator: Operator, m_max=2, sigma_0=None):
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


def self_consistent_quadratic_mfa(ham: Operator):
    """
    Find the Mean field approximation for the exponential
    of a quadratic form.

    Starts by decomposing ham as a quadratic form

    """
    from alpsqutip.operators.quadratic import build_quadratic_form_from_operator

    system = ham.system
    ham_qf = build_quadratic_form_from_operator(ham, isherm=True, simplify=True)
    # TODO: use random choice
    basis = sorted(
        [(w, b) for w, b in zip(ham_qf.weights, ham_qf.basis) if w < 0],
        key=lambda x: x[0],
    )
    w_0, b_0 = basis[0]
    offset = ham_qf.offset or ScalarOperator(0, system)

    def try_state(beta):
        k_op = beta * w_0 * b_0 + offset
        return GibbsProductDensityOperator(k_op, system=b_0.system), k_op

    def hartree_free_energy(state, k_op):
        free_energy = np.real(sum(state.free_energies.values()))
        h_av, k_av = np.real(state.expect([state.expect(ham), state.expect(k_op)]))
        return np.real(h_av - k_av + free_energy)

    # Start by optimizing with the best candidate:
    def try_function(beta):
        return hartree_free_energy(*try_state(beta)) + 0.001 * beta**2

    res = minimize_scalar(try_function, (-0.2, 0.25), method="golden")
    # Now, run a self consistent loop
    h_fe = res.fun
    phis = [0 for b in basis]
    phis[0] = res.x
    state_mf, kappa = try_state(res.x)

    for _ in range(10):
        exp_vals = state_mf.expect([w_b[1] for w_b in basis]).real
        new_phis = [2 * o_av for w_b, o_av in zip(basis, exp_vals)]
        phis = new_phis
        # phis = [.75*a + .25*b for a,b in zip(phis, new_phis)]
        kappa = offset + sum(
            w_b[1] * (w_b[0] * coeff) for coeff, w_b in zip(phis, basis)
        )
        kappa = kappa.simplify()
        state_mf = GibbsProductDensityOperator(kappa, system=system)
        new_h_fe = hartree_free_energy(state_mf, kappa)
        if new_h_fe > h_fe:
            break
        h_fe = new_h_fe
    return state_mf


def project_to_n_body_operator(operator, nmax=1, sigma=None):
    """
    Approximate `operator` by a sum of (up to) nmax-body
    terms, relative to the state sigma.
    By default, `sigma` is the identity matrix.

    ``operator`` can be a SumOperator or a Product Operator.
    """

    def mul_func(x, y):
        return x * y

    if isinstance(operator, SumOperator):
        terms = operator.simplify().flat().terms
    else:
        terms = (operator,)

    system = operator.system
    if sigma is None:
        sigma = ProductDensityOperator({}, system=system)
    terms_by_factors = {0: [], 1: [], nmax: []}
    untouched = True
    for term in terms:
        acts_over = term.acts_over()
        if acts_over is None:
            continue
        n = len(acts_over)
        if nmax >= n:
            terms_by_factors.setdefault(n, []).append(term)
            continue
        untouched = False
        if n == 1:
            term = ScalarOperator(sigma.expect(term), system)
            terms_by_factors[0].append(term)
            continue

        # Now, let's assume that `term` is a `ProductOperator`
        sites_op = term.sites_op
        averages = sigma.expect(
            {
                site: sigma.expect(LocalOperator(site, l_op, system))
                for site, l_op in sites_op.items()
            }
        )
        fluct_op = {site: l_op - averages[site] for site, l_op in sites_op.items()}
        # Now, we run a loop over
        for n_factors in range(nmax + 1):
            # subterms = terms_by_factors.setdefault(n_factors, [])
            for subcomb in combinations(sites_op, n_factors):
                num_factors = (
                    val for site, val in averages.items() if site not in subcomb
                )
                prefactor = reduce(mul_func, num_factors, term.prefactor)
                if prefactor == 0:
                    continue
                sub_site_ops = {site: fluct_op[site] for site in subcomb}
                terms_by_factors[nmax].append(
                    ProductOperator(sub_site_ops, prefactor, system)
                )

    if untouched:
        # The projection is trivial. Return the original operator.
        return operator

    scalars = terms_by_factors[0]
    if len(scalars) > 1:
        total = sum(term.prefactor for term in scalars)
        terms_by_factors[0] = [ScalarOperator(total, system)] if total else []
    one_body = terms_by_factors.get(1, [])
    if len(one_body) > 1:
        terms_by_factors[1] = [OneBodyOperator(tuple(one_body), system).simplify()]

    terms = tuple((term for terms in terms_by_factors.values() for term in terms))
    result = SumOperator(terms, system).simplify()
    return result
