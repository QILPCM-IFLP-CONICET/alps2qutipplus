"""
Module that implements a meanfield approximation of a Gibbsian state
"""

from functools import reduce
from itertools import combinations, permutations
from typing import Optional, Union

import numpy as np
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
from alpsqutip.operators.states.states import (
    DensityOperatorMixin,
    GibbsProductDensityOperator,
    ProductDensityOperator,
)


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

    system = sigma0.system if sigma0 is not None else None
    if system is None:
        system = operator.system if isinstance(operator, Operator) else None

    if isinstance(operator, Qobj):
        operator = QutipOperator(operator, system)

    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=operator.system)
        sigma0 = sigma0 / sigma0.tr()
        system = sigma0.system

    local_states = {
        name: sigma0.partial_trace([name]).to_qutip() for name in system.dimensions
    }

    local_terms = []
    averages = 0
    for name in local_states:
        # Build a product operator Sigma_compl
        # s.t. Tr_{i}Sigma_i =Tr_i sigma0
        #      Tr{/i} Sigma_i = Id
        # Then, for any local operators q_i, q_j s.t.
        # Tr[q_i sigma0]= Tr[q_j sigma0]=0,
        # Tr_{/i}[q_i  Sigma_compl] = q_i
        # Tr_{/i}[q_j  Sigma_compl] = 0
        # Tr_{/i}[q_i q_j Sigma_compl] = 0

        sigma_compl_factors = {
            name_loc: s_loc
            for name_loc, s_loc in local_states.items()
            if name != name_loc
        }
        sigma_compl = ProductOperator(
            sigma_compl_factors,
            system=system,
        )
        local_term = (sigma_compl * operator).partial_trace([name])
        # Split the zero-average part from the average
        local_average = (local_term * local_states[name]).tr()
        averages += local_average
        local_term = local_term - local_average
        local_terms.append(LocalOperator(name, local_term.to_qutip(), system))

    average_term = ScalarOperator(averages / len(local_terms), system)
    one_body_term = OneBodyOperator(tuple(local_terms), system=system)
    remaining = (operator - one_body_term - average_term).to_qutip_operator()
    return SumOperator(
        tuple(
            (
                average_term,
                one_body_term,
                remaining,
            )
        ),
        system,
    )


def project_to_n_body_operator(operator, nmax=1, sigma=None):
    """
    Approximate `operator` by a sum of (up to) nmax-body
    terms, relative to the state sigma.

    ``operator`` can be a SumOperator or a Product Operator.
    """
    mul_func = lambda x, y: x * y

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
            subterms = terms_by_factors.setdefault(n_factors, [])
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


def project_meanfield(operator, sigma0=None, **kwargs):
    """
    Build a self-consistent meand field approximation
    of -log(exp(-operator)) as a OneBodyOperator
    """

    # Operators that are already "self consistent
    if type(operator) in (LocalOperator, ScalarOperator, OneBodyOperator):
        return operator

    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=operator.system)

    # cache already computed mean values
    current_sigma = kwargs.get("current_sigma", None)
    if sigma0 is not current_sigma:
        kwargs["current_sigma"] = sigma0
        kwargs["meanvalues"] = {}
    meanvalues = kwargs["meanvalues"]

    if isinstance(operator, SumOperator):
        return sum(project_meanfield(term, sigma0, **kwargs) for term in operator.terms)

    def get_meanvalue(name, op_l):
        """compute the local mean values regarding sigma0"""
        key = (name, id(op_l))
        result = meanvalues.get(key, None)
        if result is None:
            sigma_local = sigma0.partial_trace([name]).to_qutip()
            result = (op_l * sigma_local).tr()
            meanvalues[key] = result
        return result

    if isinstance(operator, ProductOperator):
        system = operator.system
        # Compute the mean values of each factor
        factors = {
            name: get_meanvalue(name, factor)
            for name, factor in operator.sites_op.items()
        }

        # The projection for a product operator
        # o = o_1 ... o_i ... o_j
        # is given by
        # prod_i <op_i> + sum_i  (op_i-<op_i> ) * prod_{j\neq i} <op_j>

        # Here we collect these terms. The first term
        # is the expectation value of the operator
        terms = [np.prod(list(factors.values()))]

        for name, local_op in operator.sites_op.items():
            prefactor = np.prod([f for name_f, f in factors.items() if name_f != name])
            loc_mv = factors[name]
            terms.append(
                LocalOperator(name, prefactor * (local_op - loc_mv), system=system)
            )
        return sum(terms)

    raise TypeError(f"Unsupported operator type '{type(operator)}'")


def self_consistent_meanfield(operator, sigma0=None, max_it=100) -> ProductOperator:
    """
    Build a self-consistent approximation of
    rho \\propto \\exp(-operator)
    as a product operator.

    If sigma0 is given, it is used as the first step.
    """
    operator = operator.flat()
    if sigma0 is None:
        sigma0 = ProductDensityOperator({}, system=operator.system)
        sigma0 = sigma0 / sigma0.tr()

    curr_it = max_it
    while curr_it:
        curr_it -= 1
        kappa = project_meanfield(operator, sigma0)
        # sigma0 = (-kappa).expm()
        # sigma0 = sigma0 / sigma0.tr()
        sigma0 = GibbsProductDensityOperator(kappa)

    return sigma0


def self_consistent_quadratic_mfa(ham: Operator):
    """
    Find the Mean field approximation for the exponential
    of a quadratic form.

    Starts by decomposing ham as a quadratic form

    """
    from alpsqutip.operators.quadratic import (
        QuadraticFormOperator,
        build_quadratic_form_from_operator,
    )

    system = ham.system
    print("Decomposing as simplified quadratic form")
    ham_qf = build_quadratic_form_from_operator(ham, isherm=True, simplify=True)
    # TODO: use random choice
    print("Reduce the basis")
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
    print("s=", res.x)
    h_fe = res.fun
    phis = [0 for b in basis]
    phis[0] = res.x
    state_mf, kappa = try_state(res.x)

    for it_int in range(10):
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
        print("checking if", new_h_fe, ">", h_fe)
        if new_h_fe > h_fe:
            break
        h_fe = new_h_fe
        print("it:", it_int)
    return state_mf
