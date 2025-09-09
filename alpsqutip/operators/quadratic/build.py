r"""
QuadraticForm Operators

Quadratic Form Operators provides a representation for quantum operators
of the form

Q= L + \sum_a w_a M_a ^2 + \delta Q

with L and M_a one-body operators, w_a certain weights and
\delta Q a *remainder* as a sum of n-body terms.



"""

# from numbers import Number
from typing import Dict, List, Optional

import numpy as np
from numpy.linalg import eigh
from qutip import Qobj

from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

from .quadratic import QuadraticFormOperator

LocalBasisDict = Dict[str, List[Qobj]]

# from typing import Union


def build_local_basis(
    terms_by_block: Dict[frozenset, List[Operator]]
) -> LocalBasisDict:
    """
    Build a local basis of operators from
    a list of two-body operators on each
    pair of sites
    """
    basis_by_site = {}
    # First, collect the one-body factors
    for sites, terms_list in terms_by_block.items():
        assert len(sites) == 2, sites
        product_terms = []
        for term in terms_list:
            # If a term is a QutipOperator, decompose
            # it first as a sum of product operators
            if hasattr(term, "as_sum_of_products"):
                term_ = term.as_sum_of_products()
                if hasattr(term_, "terms"):
                    product_terms.extend(term_.terms)
                else:
                    product_terms.append(term_)
            else:
                product_terms.append(term)

        for term in product_terms:
            site1, site2 = sites
            basis_by_site.setdefault(site1, []).append(term.sites_op[site1])
            basis_by_site.setdefault(site2, []).append(term.sites_op[site2])

    return orthonormal_hs_local_basis(basis_by_site)


def orthonormal_hs_local_basis(local_generators_dict: LocalBasisDict) -> LocalBasisDict:
    """
    From a set of operators associated to each site,
    build an orthonormalized basis of hermitian operators
    regarding the HS scalar product on each site.
    """
    basis_dict = {}
    for site, generators in local_generators_dict.items():
        basis = []
        # Now, go over each local basis:
        for generator in generators:
            # Split in hermitician and antihermitician parts:
            components = (
                (generator,)
                if generator.isherm
                else (
                    generator + generator.dag(),
                    generator * 1j - generator.dag() * 1j,
                )
            )
            # GS orthogonalization of each component regarding the existent base.
            # If the norm is under the tolerance, discard the element.
            for hcomponent in components:
                hcomponent = hcomponent - hcomponent.tr() / hcomponent.dims[0][0]
                hcomponent = hcomponent - sum(
                    (hcomponent * b_op).tr() * b_op for b_op in basis
                )
                normsq = (hcomponent * hcomponent).tr()
                if abs(normsq) < ALPSQUTIP_TOLERANCE:
                    continue
                basis.append(hcomponent * normsq ** (-0.5))
        #
        basis_dict[site] = basis
    return basis_dict


def zero_expectation_value_basis(basis: LocalBasisDict, sigma_ref: Optional):
    """
    add an offset of each element of the local basis
    in a way that each operator have zero mean regarding
    sigma_ref
    """
    local_sigmas = sigma_ref.sites_op

    new_basis = {}
    for site, local_basis in basis.items():
        local_sigma = local_sigmas[site]
        new_basis[site] = [elem - (elem * local_sigma).tr() for elem in basis[site]]
    return new_basis


def classify_terms(operator, sigma_ref: Optional):
    """
    Decompose `operator` as list of terms
    associated to each pairs of sites,
    and offset terms
    operator = sum_{ij} sum_a q_ija +  sum_{b} offset_{b}.

    If sigma_ref is None, two-body and many-body terms
    have zero trace. Otherwise, operators have zero expectation
    value relative to sigma_ref.
    """
    from alpsqutip.projections import n_body_projection

    local_sigmas = (
        sigma_ref.sites_op
        if sigma_ref is not None
        else {
            site: 1 / dimension
            for site, dimension in operator.system.dimensions.items()
        }
    )

    def decompose_two_body_product_operator(prod_op):
        prefactor = prod_op.prefactor
        system = prod_op.system
        sites_op = operator.sites_op
        assert len(sites_op) == 2
        averages = {
            site: (
                (loc_op * local_sigmas[site]).tr()
                if isinstance(loc_op, Qobj)
                else loc_op
            )
            for site, loc_op in sites_op.items()
        }
        sites_op = {
            site: (loc_op - averages[site]) for site, loc_op in sites_op.items()
        }
        site1, site2 = sites_op
        one_body_term = (
            OneBodyOperator(
                (
                    LocalOperator(
                        site1, sites_op[site1] * (averages[site2] * prefactor), system
                    ),
                    LocalOperator(
                        site2, sites_op[site2] * (averages[site1] * prefactor), system
                    ),
                ),
                system,
            )
            + averages[site1] * averages[site2] * prefactor
        )
        one_body_term = one_body_term.simplify()
        return ProductOperator(sites_op, prefactor, system).simplify(), one_body_term

    terms_by_block = {}
    offset_terms = []
    linear_terms = []

    if isinstance(operator, OneBodyOperator):
        return terms_by_block, [operator], offset_terms

    operator = operator.flat()
    # If is a sum, collect contributions of each term:
    if isinstance(operator, SumOperator):
        for term in operator.terms:
            sub_terms_by_block, sub_linear_terms, sub_offset_terms = classify_terms(
                term, sigma_ref
            )
            linear_terms.extend(sub_linear_terms)
            offset_terms.extend(sub_offset_terms)
            for key, val in sub_terms_by_block.items():
                assert len(key) == 2
                terms_by_block.setdefault(key, []).extend(val)
        return terms_by_block, linear_terms, offset_terms

    acts_over = operator.acts_over()
    if acts_over is None or len(acts_over) > 2:
        two_body_part = n_body_projection(operator, 2, sigma_ref)
        if isinstance(operator, QutipOperator):
            operator = (operator - two_body_part).to_qutip_operator()
        else:
            operator = (operator - two_body_part).simplify()

        if operator:
            offset_terms.append(operator)
        terms_by_block, linear_terms, _ = classify_terms(two_body_part, sigma_ref)
        return terms_by_block, linear_terms, offset_terms
    elif len(acts_over) < 2:
        return terms_by_block, [operator], offset_terms

    # operator acts exactly on two sites
    if isinstance(operator, QutipOperator):
        return classify_terms(operator.as_sum_of_products(), sigma_ref)
    if isinstance(operator, ProductOperator):
        operator, linear_term = decompose_two_body_product_operator(operator)
        terms_by_block[tuple(sorted(acts_over))] = [operator]
        assert len(operator.acts_over()) == 2
        return terms_by_block, ([] if linear_term.is_zero else [linear_term]), []

    raise ValueError(f"operator of type {type(operator)} cannot be processed.")


def build_quadratic_form_matrix(terms_by_block, local_basis: LocalBasisDict):
    sizes = {site: len(local_base) for site, local_base in local_basis.items()}
    sorted_sites = sorted(sizes)
    positions = {
        site: sum(sizes[site_] for site_ in sorted_sites[:pos])
        for pos, site in enumerate(sorted_sites)
    }
    full_size = sum(sizes.values())
    result_array = np.zeros(
        (
            full_size,
            full_size,
        )
    )
    for block, terms in terms_by_block.items():
        site1, site2 = block
        position_1 = positions[site1]
        position_2 = positions[site2]
        basis1 = local_basis[site1]
        basis2 = local_basis[site2]
        for term in terms:
            prefactor = term.prefactor
            op1, op2 = (term.sites_op[site] for site in block)
            for mu, b1 in enumerate(basis1):
                for nu, b2 in enumerate(basis2):
                    i = position_1 + mu
                    j = position_2 + nu
                    result_array[i, j] += np.real(
                        prefactor * (op1 * b1).tr() * (op2 * b2).tr()
                    )
                    result_array[j, i] = result_array[i, j]
    return result_array, positions


def build_quadratic_form_from_operator(
    operator: Operator,
    simplify=True,
    isherm=None,
    sigma_ref=None,
) -> QuadraticFormOperator:
    """
    Build a QuadraticFormOperator from `operator`
    """
    # Required for `assert` test below
    # from alpsqutip.operators.states.basic import (
    #    ProductDensityOperator,
    # )

    def force_hermitic_t(t):
        if t is None:
            return t
        if not t.isherm:
            t = (t + t.dag()).simplify()
            t = t * 0.5
        return t

    def spectral_norm(ob_op):
        if isinstance(ob_op, ScalarOperator):
            return ob_op.prefactor
        if isinstance(ob_op, OneBodyOperator):
            return sum(spectral_norm(term) for term in ob_op.simplify().terms)
        if isinstance(ob_op, LocalOperator):
            return max((ob_op.operator**2).eigenenergies()) ** 0.5
        raise TypeError(f"spectral_norm can not be computed for {type(ob_op)}")

    if simplify:
        operator = operator.simplify()

    if sigma_ref is not None:
        if hasattr(sigma_ref, "to_product_state"):
            sigma_ref = sigma_ref.to_product_state()
        # assert isinstance(
        #    sigma_ref, ProductDensityOperator
        # ), f"sigma_ref must be a ProductDensityOperator. Got {type(sigma_ref)}"

    system = operator.system
    # Trivial cases
    if isinstance(operator, ScalarOperator):
        if isherm and not operator.isherm:
            operator = ScalarOperator(operator.prefactor.real, system)
        assert (
            not isherm or isherm == operator.isherm
        ), f"{operator} -> {isherm}!={operator.isherm}"
        return QuadraticFormOperator(tuple(), tuple(), system, operator, None)

    if (
        isinstance(operator, (LocalOperator, OneBodyOperator))
        or len(operator.acts_over()) < 2
    ):
        if isherm and not operator.isherm:
            operator = operator + operator.dag()
        return QuadraticFormOperator(
            tuple(), tuple(), system, operator.simplify(), None
        )

    # Already a quadratic form:
    if isinstance(operator, QuadraticFormOperator):
        if isherm and not operator.isherm:
            operator = QuadraticFormOperator(
                operator.basis,
                tuple((np.real(w) for w in operator.weights)),
                system,
                force_hermitic_t(operator.linear_term),
                force_hermitic_t(operator.offset),
            )
        return operator

    # SumOperators, and operators acting on at least size 2 blocks:
    isherm = isherm or operator.isherm

    # For non-hermitician, convert the hermitician
    # and the anti-hermitician parts, and sum both.
    if not isherm:
        real_part = (
            build_quadratic_form_from_operator(
                operator + operator.dag(),
                simplify=True,
                isherm=True,
                sigma_ref=sigma_ref,
            )
            * 0.5
        )
        imag_part = (
            build_quadratic_form_from_operator(
                operator.dag() * 1j - operator * 1j,
                simplify=True,
                isherm=True,
                sigma_ref=sigma_ref,
            )
            * 0.5j
        )
        return real_part + imag_part

    # Process hermitician operators
    # Classify terms
    system = operator.system
    terms_by_2body_block, linear_terms, offset_terms = classify_terms(
        operator, sigma_ref
    )
    linear_term = sum(linear_terms).simplify() if linear_terms else None
    offset = sum(offset_terms).simplify() if offset_terms else None

    if isherm:
        linear_term = force_hermitic_t(linear_term)
        offset = force_hermitic_t(offset)

    # Build the basis
    local_basis: Dict[str, List[Qobj]] = build_local_basis(terms_by_2body_block)
    # Build the matrix of the quadratic form
    qf_array, local_basis_offsets = build_quadratic_form_matrix(
        terms_by_2body_block, local_basis
    )
    if sigma_ref is not None:
        local_basis = zero_expectation_value_basis(local_basis, sigma_ref)

    # Decompose the matrix in the eigenbasis, and build the generators
    e_vals, e_vecs = eigh(qf_array)

    qf_basis = sorted(
        [
            (
                0.5 * e_val,
                OneBodyOperator(
                    tuple(
                        [
                            LocalOperator(
                                site,
                                sum(
                                    local_op * e_vec[mu + local_basis_offsets[site]]
                                    for mu, local_op in enumerate(local_base)
                                ),
                                system,
                            )
                            for site, local_base in local_basis.items()
                        ]
                    ),
                    system,
                ),
            )
            for e_val, e_vec in zip(e_vals, e_vecs.T)
            if abs(e_val) > ALPSQUTIP_TOLERANCE
        ],
        key=lambda x: x[0],
    )

    # Normalize the generators in the spectral norm.
    spectral_norms = (
        spectral_norm(weight_generator[1]) for weight_generator in qf_basis
    )
    qf_basis = tuple(
        (
            weight_generator[0] * sn**2,
            weight_generator[1] / sn,
        )
        for sn, weight_generator in zip(spectral_norms, qf_basis)
    )
    weights = tuple((weight_generator[0] for weight_generator in qf_basis))
    qf_basis = tuple((weight_generator[1] for weight_generator in qf_basis))

    return QuadraticFormOperator(
        basis=qf_basis,
        weights=weights,
        system=operator.system,
        linear_term=linear_term,
        offset=offset,
    )
