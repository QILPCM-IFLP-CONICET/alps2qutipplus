import logging

from qutip import tensor as qutip_tensor

from alpsqutip.operators import (
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.arithmetic import iterable_to_operator
from alpsqutip.operators.states.basic import (
    DensityOperatorMixin,
    ProductDensityOperator,
)
from alpsqutip.operators.states.gibbs import GibbsDensityOperator
from alpsqutip.operators.states.meanfield import (
    variational_quadratic_mfa,
)


def project_boundary_term(term, sigma: ProductDensityOperator, sites: frozenset):
    """
    Convert terms of the form O_a Q_b in to O_a <Q_b>
    with <Q_b> the expectation value regarding sigma, and
    Q_b acting on the sub-system associated to sigma.
    """
    acts_over = term.acts_over()
    sites = {site for site in acts_over if site in sites}
    environment = frozenset(site for site in acts_over if site not in sites)
    system = term.system
    if len(sites) == 0:
        return ScalarOperator(sigma.expect(term), system)
    if all(site in sites for site in acts_over):
        return term

    local_states = sigma.sites_op
    local_states = {site: local_states[site] for site in environment}

    if isinstance(term, SumOperator):
        return iterable_to_operator(
            (project_boundary_term(sub_term, sigma, sites) for sub_term in term.terms),
            system,
            isherm=True,
        )
    if isinstance(term, ProductOperator):
        prefactor = term.prefactor
        sites_op = term.sites_op
        for site in environment:
            prefactor = prefactor * (sites_op[site] * local_states[site]).tr()
        sites_op = {site: op for site, op in sites_op.items() if site in sites}
        return ProductOperator(sites_op, prefactor, system)
    if isinstance(term, QutipOperator):
        block = tuple(sites) + tuple(environment)
        qutip_op = term.to_qutip(block)
        qutip_op = qutip_op * qutip_tensor(
            [
                local_states.get(site, None) or system.site_identity(site)
                for site in block
            ]
        )
        qutip_op = qutip_op.ptrace(list(range(len(sites)))) * term.prefactor
        names = {site: pos for pos, site in enumerate(sites)}
        return QutipOperator(qutip_op, system, names)
    # QuadraticFormOperator
    if hasattr(term, "as_sum_of_products"):
        term = term.as_sum_of_products()
        return project_boundary_term(term, sigma, sites)
    logging.warning(
        f"boundary term is not Product or Qutip ({type(term)}). Converting to QutipOperator"
    )
    return project_boundary_term(term.to_qutip_operator(), sigma, sites)


def gibbs_meanfield_partial_trace(
    state: GibbsDensityOperator, sites: frozenset
) -> DensityOperatorMixin:
    """
    Build a self-consistent Mean Field approximation to the local state.

    """
    prefactor = state.prefactor
    generator = state.k
    full_acts_over = generator.acts_over()
    environment = frozenset(site for site in full_acts_over if site not in sites)
    system = state.k.system
    subsystem = system.subsystem(sites)

    # For states in small subsystems, just compute the partial trace
    # *exactly* by exponentiating the state.
    if len(full_acts_over) <= 4:
        result = state.to_qutip_operator().partial_trace(sites)
        return result

    sigma_mf = state._meanfield
    if not environment:
        result = GibbsDensityOperator(
            generator, system=subsystem, prefactor=prefactor, meanfield=sigma_mf
        )
        return result

    generator = state.k.flat()
    all_terms = generator.terms if isinstance(generator, SumOperator) else [generator]
    terms_in, terms_boundary = [], []
    for term in all_terms:
        term_acts_over = term.acts_over()
        if term_acts_over is None:
            terms_boundary.append(term_acts_over)
        elif term_acts_over.issubset(sites):
            terms_in.append(term)
        elif term_acts_over.issubset(environment):
            continue
        else:
            terms_boundary.append(term)

    if terms_boundary:
        # If there are boundary terms, project them
        if sigma_mf is None:
            sigma_mf = variational_quadratic_mfa(-generator).to_product_state()
            state._meanfield = sigma_mf

        # Project the terms onto the algebra of the local subsystem
        terms_boundary = (
            project_boundary_term(term, sigma_mf, sites) for term in terms_boundary
        )
        # Remove empty terms
        terms_boundary = tuple(term for term in terms_boundary if term)
        terms_in.extend(terms_boundary)

    k_in = iterable_to_operator(terms_in, system, isherm=True)

    result = GibbsDensityOperator(
        k_in, subsystem, prefactor=prefactor
    ).to_qutip_operator()

    for symm in state.symmetry_projections:
        result_new = symm(result)
        # assert (
        #    (result_new - result).tidyup().is_zero
        # ), f"result is not invariant under {symm}."
        result = result_new
    return result
