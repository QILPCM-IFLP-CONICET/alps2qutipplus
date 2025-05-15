r"""
QuadraticForm Operators

Quadratic Form Operators provides a representation for quantum operators
of the form

Q= L + \sum_a w_a M_a ^2 + \delta Q

with L and M_a one-body operators, w_a certain weights and
\delta Q a *remainder* as a sum of n-body terms.



"""

from numbers import Number

# from numbers import Number
from time import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.linalg import eigh
from numpy.random import random

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.states import (
    GibbsProductDensityOperator,
    MixtureDensityOperator,
    ProductDensityOperator,
)
from alpsqutip.settings import ALPSQUTIP_TOLERANCE

# from typing import Union


class QuadraticFormOperator(Operator):
    """
    Represents a two-body operator of the form
    sum_alpha w_alpha * Q_alpha^2
    with Q_alpha a local operator or a One body operator.
    """

    system: SystemDescriptor
    terms: list
    weights: list
    offset: Optional[Operator]

    def __init__(self, basis, weights, system=None, linear_term=None, offset=None):
        # If the system is not given, infer it from the terms
        if offset:
            offset = offset.simplify()
        if linear_term:
            linear_term = linear_term.simplify()
            assert (
                isinstance(linear_term, OneBodyOperator)
                or len(linear_term.acts_over()) < 2
            )

        assert isinstance(basis, tuple)
        assert isinstance(weights, tuple)
        assert all(gen.isherm for gen in basis)  # TODO: REMOVE ME
        assert (
            isinstance(linear_term, (OneBodyOperator, LocalOperator, ScalarOperator))
            or linear_term is None
        ), f"{type(offset)} should be a LocalOperator or a OneBodyOperator"
        if system is None:
            for term in basis:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

        # If check_and_simplify, ensure that all the terms are
        # one-body operators and try to use the simplified forms
        # of the operators.

        self.weights = weights
        self.basis = basis
        self.system = system
        self.offset = offset
        self.linear_term = linear_term
        self._simplified = False

    def __bool__(self):
        for term in (self.linear_term, self.offset):
            if term is not None:
                if not term.is_zero:
                    return True
        return len(self.weights) > 0 and any(self.weights) and any(self.basis)

    def __add__(self, other):

        # TODO: remove me and fix the sums
        if not bool(other):
            return self
        if isinstance(other, Number):
            other = ScalarOperator(other, system=self.system)

        assert isinstance(other, Operator), "other must be an operator."
        system = self.system or other.system
        if isinstance(other, QuadraticFormOperator):
            basis = self.basis + other.basis
            weights = self.weights + other.weights
            offset = self.offset
            linear_term = self.linear_term
            if offset is None:
                offset = other.offset
            else:
                if other.offset is not None:
                    offset = offset + other.offset

            if linear_term is None:
                offset = other.linear_term
            else:
                if other.linear_term is not None:
                    linear_term = linear_term + other.linear_term
            return QuadraticFormOperator(basis, weights, system, linear_term, offset)
        if isinstance(
            other,
            (
                ScalarOperator,
                LocalOperator,
                OneBodyOperator,
            ),
        ):
            linear_term = self.linear_term
            linear_term = (
                other if linear_term is None else (linear_term + other).simplify()
            )
            basis = self.basis
            weights = self.weights
            return QuadraticFormOperator(
                basis, weights, system, linear_term, offset=None
            )
        return SumOperator(
            (
                self,
                other,
            ),
            system,
        )

    def __mul__(self, other):
        system = self.system
        if isinstance(other, ScalarOperator):
            other = other.prefactor
            system = system or other.system
        if isinstance(other, (float, complex)):
            offset = self.offset
            if offset is not None:
                offset = offset * other
            linear_term = self.linear_term
            if linear_term is not None:
                linear_term = (linear_term * other).simplify()

            return QuadraticFormOperator(
                self.basis,
                tuple(w * other for w in self.weights),
                system,
                linear_term=linear_term,
                offset=offset,
            )
        standard_repr = self.to_sum_operator(False).simplify()
        return standard_repr * other

    def __neg__(self):
        offset = self.offset
        if offset is not None:
            offset = -offset
        linear_term = self.linear_term
        if linear_term is not None:
            linear_term = -linear_term
        return QuadraticFormOperator(
            self.basis,
            tuple(-w for w in self.weights),
            self.system,
            linear_term,
            offset,
        )

    def acts_over(self):
        """
        Set of sites over the state acts.
        """
        result = frozenset()
        for term in self.basis:
            try:
                result = result.union(term.acts_over())
            except TypeError:
                return None

        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            try:
                result = result.union(term.acts_over())
            except TypeError:
                return None
        return result

    def dag(self):
        linear_term = self.linear_term
        linear_term = None if linear_term is None else linear_term.dag()
        offset = self.offset
        offset = None if offset is None else offset.dag()
        result = QuadraticFormOperator(
            self.basis,
            tuple((np.conj(w) for w in self.weights)),
            self.system,
            linear_term,
            offset,
        )
        result._simplified = self._simplified
        return result

    def flat(self):
        return self

    @property
    def isdiagonal(self):
        """True if the operator is diagonal in the product basis."""
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            isdiagonal = term.isdiagonal
            if not isdiagonal:
                return isdiagonal

        if all(term.isdiagonal for term in self.basis):
            return True
        return False

    @property
    def isherm(self):
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            isherm = term.isherm
            if not isherm:
                return isherm

        weights = self.weights
        if len(weights) == 0:
            return True
        return all(abs(np.imag(weight)) < ALPSQUTIP_TOLERANCE for weight in weights)

    def partial_trace(self, sites: Union[tuple, SystemDescriptor]):

        if not isinstance(sites, SystemDescriptor):
            sites = self.system.subsystem(sites)

        result = None
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            if result:
                tpt = term.partial_trace(sites)
                assert isinstance(tpt, Operator)
                result = result + tpt
            else:
                result = term.partial_trace(sites)
                assert isinstance(
                    result, Operator
                ), f"partial trace of {type(term)} returns {type(result)}"

        if len(self.basis) == 0 and result is None:
            return ScalarOperator(0, sites)

        # TODO: Implement me to return a quadratic operator
        #
        #  (Sum_a  w_a(sum_i L_ai)^2).ptrace = Sum_a w_a ((sum_i L_ai)^2).ptrace
        #  (Sum_i L_ai)^2 = Sum_i (La_i L_aj).ptrace= (La_i1)^2*Tr[1_2] + I Tr(La_i2)^2+...
        #
        terms = tuple(
            w * (op_term * op_term).partial_trace(sites)
            for w, op_term in zip(self.weights, self.basis)
        )
        if result is not None:
            terms = terms + (result,)
        terms = tuple(terms)
        return SumOperator(
            terms,
            sites,
        ).simplify()

    def simplify(self):
        """
        Simplify the operator.
        Build a new representation with a smaller basis.
        """
        if self._simplified:
            return self

        result = simplify_quadratic_form(self, hermitic=False)
        result._simplified = True
        return result

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        """
        return a qutip object acting over the sites listed in
        `block`.
        By default (`block=None`) returns a qutip object
        acting over all the sites, in lexicographical order.
        """
        sites = self.system.sites
        if block is None:
            block = tuple(sorted(sites))
        else:
            block = block + tuple(
                (site for site in self.acts_over() if site not in block)
            )

        result = sum(
            (w * op_term.dag() * op_term).to_qutip(block)
            for w, op_term in zip(self.weights, self.basis)
        )
        for term in (self.offset, self.linear_term):
            if term is not None:
                result += term.to_qutip(block)
        return result

    def to_sum_operator(self, simplify: bool = True) -> SumOperator:
        """Convert to a linear combination of quadratic operators"""
        isherm = self.isherm
        isdiag = self.isdiagonal
        if all(b_op.isherm for b_op in self.basis):
            terms = tuple(
                (
                    ((op_term * op_term) * w)
                    for w, op_term in zip(self.weights, self.basis)
                )
            )
        else:
            terms = tuple(
                (
                    ((op_term * op_term) * w)
                    for w, op_term in zip(self.weights, self.basis)
                )
            )

        for term in (self.offset, self.linear_term):
            if term is not None:
                terms = terms + (term,)
        if len(terms) == 0:
            return ScalarOperator(0, self.system)
        if len(terms) == 1:
            return terms[0]
        result = SumOperator(terms, self.system, isherm, isdiag)
        if simplify:
            return result.simplify()
        return result


def build_local_basis(terms_by_block):
    """
    Build a local basis of operators from
    a list of two-body operators on each
    pair of sites
    """
    basis_by_site = {}
    # First, collect the one-body factors
    for sites, terms_list in terms_by_block.items():
        assert len(sites) == 2, sites
        for term in terms_list:
            site1, site2 = sites
            basis_by_site.setdefault(site1, []).append(term.sites_op[site1])
            basis_by_site.setdefault(site2, []).append(term.sites_op[site2])

    return orthonormal_hs_local_basis(basis_by_site)


def orthonormal_hs_local_basis(local_generators_dict: dict):
    """
    From a set of operators associated to each site,
    build an orthonormalized basis of hermitian operators
    regarding the HS scalar product on each site.
    """
    basis_dict = {}
    for site, generators in local_generators_dict.items():
        basis = []
        for generator in generators:
            components = (
                (generator,)
                if generator.isherm
                else (
                    generator + generator.dag(),
                    generator * 1j - generator.dag() * 1j,
                )
            )
            for hcomponent in components:
                hcomponent = hcomponent - hcomponent.tr() / hcomponent.dims[0][0]
                hcomponent = hcomponent - sum(
                    (hcomponent * b_op).tr() * b_op for b_op in basis
                )
                normsq = (hcomponent * hcomponent).tr()
                if abs(normsq) < ALPSQUTIP_TOLERANCE:
                    continue
                basis.append(hcomponent * normsq ** (-0.5))
        basis_dict[site] = basis
    return basis_dict


def classify_terms(operator):
    """
    Decompose `operator` as list of terms
    associated to each pairs of sites,
    and offset terms
    operator = sum_{ij} sum_a q_ija +  sum_{b} offset_{b}.
    """
    from alpsqutip.operators.qutip import QutipOperator

    def decompose_two_body_product_operator(prod_op):
        prefactor = prod_op.prefactor
        system = prod_op.system
        sites_op = operator.sites_op
        assert len(sites_op) == 2
        averages = {site: loc_op.tr() for site, loc_op in sites_op.items()}
        sites_op = {site: (loc_op - loc_op.tr()) for site, loc_op in sites_op.items()}
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

    if not isinstance(operator, SumOperator):
        acts_over = operator.acts_over()
        if acts_over is None or len(acts_over) > 2:
            return terms_by_block, linear_terms, [operator]
        elif len(acts_over) < 2:
            return terms_by_block, [operator], offset_terms

        # operator acts exactly on two sites
        if isinstance(operator, QutipOperator):
            operator = operator.as_sum_of_products()
        if isinstance(operator, ProductOperator):
            operator, linear_term = decompose_two_body_product_operator(operator)
            terms_by_block[tuple(sorted(acts_over))] = [operator]
            assert len(operator.acts_over()) == 2
            return terms_by_block, ([] if linear_term.is_zero else [linear_term]), []

    assert isinstance(operator, SumOperator)
    for term in operator.terms:
        sub_terms_by_block, sub_linear_terms, sub_offset_terms = classify_terms(term)
        linear_terms.extend(sub_linear_terms)
        offset_terms.extend(sub_offset_terms)
        for key, val in sub_terms_by_block.items():
            assert len(key) == 2
            terms_by_block.setdefault(key, []).extend(val)

    return terms_by_block, linear_terms, offset_terms


def build_quadratic_form_matrix(terms_by_block, local_basis):
    sizes = {site: len(local_base) for site, local_base in local_basis.items()}
    sorted_sites = sorted(sizes)
    offsets = {
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
        offset1 = offsets[site1]
        offset2 = offsets[site2]
        basis1 = local_basis[site1]
        basis2 = local_basis[site2]
        for term in terms:
            prefactor = term.prefactor
            op1, op2 = (term.sites_op[site] for site in block)
            for mu, b1 in enumerate(basis1):
                for nu, b2 in enumerate(basis2):
                    i = offset1 + mu
                    j = offset2 + nu
                    result_array[i, j] += np.real(
                        prefactor * (op1 * b1).tr() * (op2 * b2).tr()
                    )
                    result_array[j, i] = result_array[i, j]
    return result_array, offsets


def build_quadratic_form_from_operator(
    operator: Operator, simplify=True, isherm=None
) -> Operator:
    """
    Build a QuadraticFormOperator from `operator`
    """

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

    system = operator.system
    # Trivial cases
    if isinstance(operator, ScalarOperator):
        if isherm and not operator.isherm:
            operator = ScalarOperator(operator.prefactor.real, system)
        assert (
            isherm or operator.isherm == operator.isherm
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

    if not isherm:
        real_part = (
            build_quadratic_form_from_operator(
                operator + operator.dag(), simplify=True, isherm=True
            )
            * 0.5
        )
        imag_part = (
            build_quadratic_form_from_operator(
                operator.dag() * 1j - operator * 1j, simplify=True, isherm=True
            )
            * 0.5j
        )
        return real_part + imag_part

    # Process hermitician operators

    # Classify terms
    system = operator.system
    terms_by_2body_block, linear_terms, offset_terms = classify_terms(operator)
    linear_term = sum(linear_terms).simplify() if linear_terms else None
    offset = sum(offset_terms).simplify() if offset_terms else None

    if isherm:
        linear_term = force_hermitic_t(linear_term)
        offset = force_hermitic_t(offset)

    # Build the basis
    local_basis = build_local_basis(terms_by_2body_block)
    # Build the matrix of the quadratic form
    qf_array, local_basis_offsets = build_quadratic_form_matrix(
        terms_by_2body_block, local_basis
    )

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


def quadratic_form_expect(sq_op, state):
    """
    Compute the expectation value of op, taking advantage
    of its structure.
    """
    sq_op = sq_op.to_sum_operator(False)
    return state.expect(sq_op)


def selfconsistent_meanfield_from_quadratic_form(
    quadratic_form: QuadraticFormOperator, max_it, logdict=None
):
    """
    Build a self-consistent mean field approximation
    to the gibbs state associated to the quadratic form.
    """
    #    quadratic_form = simplify_quadratic_form(quadratic_form)
    system = quadratic_form.system
    terms = quadratic_form.terms
    weights = quadratic_form.weights

    operators = [2 * w * b for w, b in zip(weights, terms)]
    basis = [b for w, b in zip(weights, terms)]

    phi = [(2.0 * random() - 1.0)]

    evolution: list = []
    timestamps: list = []

    if isinstance(logdict, dict):
        logdict["states"] = evolution
        logdict["timestamps"] = timestamps

    remaining_iterations = max_it
    while remaining_iterations:
        remaining_iterations -= 1
        k_exp = OneBodyOperator(
            tuple(phi_i * operator for phi_i, operator in zip(phi, basis)),
            system,
        )
        k_exp = ((k_exp + k_exp.dag()).simplify()) * 0.5
        assert k_exp.isherm
        rho = GibbsProductDensityOperator(k_exp, 1.0, system)
        new_phi = -rho.expect(operators).conj()
        if isinstance(logdict, dict):
            evolution.append(new_phi)
            timestamps.append(time())

        change = sum(
            abs(old_phi_i - new_phi_i) for old_phi_i, new_phi_i in zip(new_phi, phi)
        )
        if change < 1.0e-10:
            break
        phi = new_phi

    return rho


def ensure_hermitician_basis(self: QuadraticFormOperator):
    """
    Ensure that the quadratic form is expanded using a
    basis of hermitician operators.
    """
    basis = self.basis
    if all(b.isherm for b in basis):
        return self

    coeffs = self.weights
    system = self.system
    offset = self.offset

    # Reduce the basis to an hermitician basis
    new_basis = []
    new_coeffs = []
    local_terms = []
    for la_coeff, b_op in zip(coeffs, basis):
        if b_op.isherm:
            new_basis.append(b_op)
            new_coeffs.append(la_coeff)
            continue
        # Not hermitician. Decompose as two hermitician terms
        # and an offset
        if la_coeff == 0:
            continue
        b_h = ((b_op + b_op.dag()) * 0.5).simplify()
        b_a = ((b_op - b_op.dag()) * 0.5j).simplify()
        if b_h:
            new_basis.append(b_h)
            new_coeffs.append(la_coeff)
            if b_a:
                new_basis.append(b_a)
                new_coeffs.append(la_coeff)
                comm = ((b_h * b_a - b_a * b_h) * (1j * la_coeff)).simplify()
                if comm:
                    local_terms.append(comm)
        elif b_a:
            new_basis.append(b_a)
            new_coeffs.append(la_coeff)

    local_terms = [term for term in local_terms if term]
    if offset is not None:
        local_terms = [offset] + local_terms
    if local_terms:
        new_offset = sum(local_terms).simplify()

    if not bool(new_offset):
        new_offset = None
    return QuadraticFormOperator(
        tuple(new_basis), tuple(new_coeffs), system, new_offset
    )


def one_body_operator_hermitician_hs_sp(x_op: OneBodyOperator, y_op: OneBodyOperator):
    """
    Hilbert Schmidt scalar product optimized for OneBodyOperators
    """
    result = 0
    terms_x: Tuple[LocalOperator] = (
        x_op.terms if isinstance(x_op, OneBodyOperator) else (x_op,)
    )
    terms_y: Tuple[LocalOperator] = (
        y_op.terms if isinstance(y_op, OneBodyOperator) else (y_op,)
    )

    for t_1 in terms_x:
        for t_2 in terms_y:
            if isinstance(t_1, ScalarOperator):
                result += t_2.tr() * t_1.prefactor
            elif isinstance(t_2, ScalarOperator):
                result += t_1.tr() * t_2.prefactor
            elif t_1.site == t_2.site:
                result += (t_1.operator * t_2.operator).tr()
            else:
                result += t_1.operator.tr() * t_2.operator.tr()
    return result


def simplify_quadratic_form(
    operator: QuadraticFormOperator,
    hermitic: bool = True,
    scalar_product: Callable = one_body_operator_hermitician_hs_sp,
):
    """
    Takes a 2-body operator and returns lists weights, ops
    such that the original operator is
    sum(w * op**2 for w,op in zip(weights,ops))
    """
    changed = False
    system = operator.system
    if not operator.isherm and hermitic:
        changed = True

    def simplify_other_terms(term):
        nonlocal changed
        new_term = term
        if hermitic:
            if term is not None:
                if not term.isherm:
                    new_term = 0.5 * new_term
                    new_term = new_term + new_term.dag()
                new_term = new_term.simplify()
        elif term is not None:
            new_term = term.simplify()
        if term is not new_term:
            changed = True
        return new_term

    # First, rebuild the quadratic form.
    qf_op = QuadraticFormOperator(operator.basis, operator.weights, system)
    new_qf_op = build_quadratic_form_from_operator(
        qf_op.to_sum_operator(), True, hermitic
    )
    # If the new basis is larger and the hermitician character haven´t changed, keep the older.
    if changed or len(new_qf_op.basis) < len(qf_op.basis):
        qf_op = new_qf_op
        changed = True

    # Now, work on the offset and the linear term

    linear_term = simplify_other_terms(operator.linear_term)
    offset = simplify_other_terms(operator.offset)

    if not changed:
        return operator

    if qf_op.linear_term:
        linear_term = (
            (linear_term + qf_op.linear_term).simplify()
            if linear_term
            else qf_op.linear_term
        )

    if qf_op.offset:
        offset = (
            (offset + qf_op.offset).simplify() if offset is not None else qf_op.offset
        )

    return QuadraticFormOperator(
        qf_op.basis, qf_op.weights, system, linear_term, offset
    )


# #####################
#
#  Arithmetic
#
# #######################


@Operator.register_add_handler(
    (
        ScalarOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_add_handler(
    (
        LocalOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_add_handler(
    (
        ProductOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_add_handler(
    (
        SumOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_add_handler(
    (
        OneBodyOperator,
        QuadraticFormOperator,
    )
)
def _(op1: Operator, op2: QuadraticFormOperator):
    return op2 + op1


# Products right products


@Operator.register_mul_handler(
    (
        ScalarOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        float,
        QuadraticFormOperator,
    )
)
def _(op1: ScalarOperator, op2: QuadraticFormOperator):
    return op2 * op1


@Operator.register_mul_handler(
    (
        LocalOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        SumOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        OneBodyOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductDensityOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        GibbsProductDensityOperator,
        QuadraticFormOperator,
    )
)
@Operator.register_mul_handler(
    (
        MixtureDensityOperator,
        QuadraticFormOperator,
    )
)
def _(op1: Operator, op2: QuadraticFormOperator):
    return op1 * op2.to_sum_operator()
