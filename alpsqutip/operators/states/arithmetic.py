"""
Arithmetic operations with states.

Essentially, arithmetic operations with states involves just mixing of operators,
implemented though the class MixtureDensityOperator.

"""

from numbers import Number
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states.basic import (
    DensityOperatorMixin,
    ProductDensityOperator,
)
from alpsqutip.operators.states.gibbs import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)


class MixtureDensityOperator(DensityOperatorMixin, SumOperator):
    """
    A mixture of density operators
    """

    terms: Tuple[DensityOperatorMixin]

    def __init__(self, terms: tuple, system: SystemDescriptor = None):
        super().__init__(terms, system, True)

    def __add__(self, rho: Operator):
        terms = self.terms
        system = self.system

        if isinstance(rho, MixtureDensityOperator):
            terms = terms + rho.terms
        elif isinstance(rho, DensityOperatorMixin):
            terms = terms + (rho,)
        elif isinstance(rho, (int, float)):
            terms = terms + (ProductDensityOperator({}, rho, system, False),)
        else:
            return super().__add__(rho)
        return MixtureDensityOperator(terms, system)

    def __mul__(self, a):
        if isinstance(a, float):
            return MixtureDensityOperator(tuple(term * a for term in self.terms))
        return super().__mul__(a)

    def __rmul__(self, a):
        if isinstance(a, float) and a >= 0:
            return MixtureDensityOperator(
                tuple(term * a for term in self.terms), self.system
            )
        return super().__rmul__(a)

    def acts_over(self) -> set:
        """
        Return a set with the name of the
        sites where the operator nontrivially acts
        """
        sites: set = set()
        for term in self.terms:
            sites.update(term.acts_over())
        return sites

    def expect(self, obs: Union[Operator, Iterable]) -> Union[np.ndarray, dict, Number]:
        strip = False
        if isinstance(obs, Operator):
            strip = True
            obs = [obs]

        av_terms = tuple((term.expect(obs), term.prefactor) for term in self.terms)

        if isinstance(obs, dict):
            return {
                op_name: sum(term[0][op_name] * term[1] for term in av_terms)
                for op_name in obs
            }
        if strip:
            return sum(np.array(term[0]) * term[1] for term in av_terms)[0]
        return sum(np.array(term[0]) * term[1] for term in av_terms)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        new_terms = tuple(t.partial_trace(sites) for t in self.terms)
        subsystem = new_terms[0].system
        return MixtureDensityOperator(new_terms, subsystem)

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip()

        if block is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(self.acts_over()) if site not in block)
            )

        # TODO: find a more efficient way to avoid element-wise
        # multiplications
        return sum(term.to_qutip(block) * term.prefactor for term in self.terms)


# ####################################
#  Arithmetic
# ####################################


# #### Sums #############


@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        MixtureDensityOperator,
    )
)
def sum_two_mixture_operators(
    x_op: MixtureDensityOperator, y_op: MixtureDensityOperator
):
    terms = x_op.terms + y_op.terms
    # If there is just one term, return it:
    if len(terms) == 1:
        return terms[0]

    # For empty terms, return 0
    system = x_op.system or y_op.system
    if len(terms) == 0:
        return ScalarOperator(0.0, system)
    # General case
    return MixtureDensityOperator(terms, system)


@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        DensityOperatorMixin,
    )
)
@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        GibbsDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        GibbsProductDensityOperator,
    )
)
def sum_mixture_and_density_operators(
    x_op: MixtureDensityOperator, y_op: DensityOperatorMixin
):
    terms = x_op.terms + (y_op,)
    # If there is just one term, return it:
    if len(terms) == 1:
        return terms[0]

    # For empty terms, return 0
    system = x_op.system or y_op.system
    if len(terms) == 0:
        return ScalarOperator(0.0, system)
    # General case
    return MixtureDensityOperator(terms, system)


@Operator.register_add_handler(
    (
        DensityOperatorMixin,
        DensityOperatorMixin,
    )
)
@Operator.register_add_handler(
    (
        ProductDensityOperator,
        DensityOperatorMixin,
    )
)
@Operator.register_add_handler(
    (
        DensityOperatorMixin,
        ProductDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        DensityOperatorMixin,
        GibbsDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        DensityOperatorMixin,
        GibbsProductDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        GibbsDensityOperator,
        GibbsDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        GibbsDensityOperator,
        GibbsProductDensityOperator,
    )
)
@Operator.register_add_handler(
    (
        GibbsProductDensityOperator,
        GibbsProductDensityOperator,
    )
)
def _(x_op, y_op):
    system = x_op.system.union(y_op.system)
    return MixtureDensityOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )


@Operator.register_add_handler(
    (
        GibbsProductDensityOperator,
        ProductOperator,
    )
)
def _(x_op, y_op):
    return SumOperator(
        (
            x_op,
            y_op,
        ),
        x_op.system or y_op.system,
    )


# #### Products #############


# ProductDensityOperator times ProductDensityOperator
@Operator.register_mul_handler((ProductDensityOperator, ProductDensityOperator))
def _(x_op: ProductDensityOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    sites_op = x_op.sites_op.copy()
    for site, factor in y_op.sites_op.items():
        if site in sites_op:
            sites_op[site] *= factor
        else:
            sites_op[site] = factor
    return ProductOperator(sites_op, 1, system)


# ProductDensityOperator times Operators

# ###   ScalarOperator


@Operator.register_mul_handler((ScalarOperator, ProductDensityOperator))
def _(x_op: ScalarOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    result = ProductOperator(y_op.sites_op, x_op.prefactor, system)
    return result


@Operator.register_mul_handler((ProductDensityOperator, ScalarOperator))
def _(x_op: ProductDensityOperator, y_op: ScalarOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    prefactor = y_op.prefactor
    # prefactor = prefactor * np.exp(sum(x_op.local_fs.values()))
    return ProductOperator(x_op.sites_op, prefactor, system)


# ###   LocalOperator


@Operator.register_mul_handler((LocalOperator, ProductDensityOperator))
def _(x_op: LocalOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    prefactor = x_op.prefactor
    sites_op = y_op.sites_op.copy()
    site = x_op.site
    if site in sites_op:
        sites_op[site] = x_op.operator * sites_op[site]
    else:
        sites_op[site] = x_op.operator

    return ProductOperator(sites_op, prefactor, system)


@Operator.register_mul_handler((ProductDensityOperator, LocalOperator))
def _(x_op: ProductDensityOperator, y_op: LocalOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    sites_op = x_op.sites_op.copy()
    site = y_op.site
    if site in sites_op:
        sites_op[site] *= y_op.operator
    else:
        sites_op[site] = y_op.operator
    return ProductOperator(sites_op, prefactor=1, system=system)


# ProductOperator


@Operator.register_mul_handler((ProductOperator, ProductDensityOperator))
def _(x_op: ProductOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    prefactor = x_op.prefactor
    sites_op = x_op.sites_op.copy()
    for site, factor in y_op.sites_op.items():
        if site in sites_op:
            sites_op[site] *= factor
        else:
            sites_op[site] = factor

    return ProductOperator(sites_op, prefactor, system)


@Operator.register_mul_handler((ProductDensityOperator, ProductOperator))
def _(x_op: ProductDensityOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    sites_op = x_op.sites_op.copy()
    for site, factor in y_op.sites_op.items():
        if site in sites_op:
            sites_op[site] *= factor
        else:
            sites_op[site] = factor

    return ProductOperator(sites_op, y_op.prefactor, system)


# SumOperators


@Operator.register_mul_handler(
    (
        ProductDensityOperator,
        SumOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductDensityOperator,
        OneBodyOperator,
    )
)
def _(x_op: ProductDensityOperator, y_op: SumOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return SumOperator(
        tuple(x_op * term for term in y_op.terms),
        system,
    )


@Operator.register_mul_handler(
    (
        OneBodyOperator,
        ProductDensityOperator,
    )
)
@Operator.register_mul_handler(
    (
        SumOperator,
        ProductDensityOperator,
    )
)
def _(x_op: SumOperator, y_op: ProductDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = tuple(term * y_op for term in x_op.terms)
    return SumOperator(
        terms,
        system,
    )


# ############    Mixtures  ##########################


# Anything that is not a SumOperator


@Operator.register_mul_handler(
    (
        Operator,
        MixtureDensityOperator,
    )
)
@Operator.register_mul_handler(
    (
        ScalarOperator,
        MixtureDensityOperator,
    )
)
@Operator.register_mul_handler(
    (
        LocalOperator,
        MixtureDensityOperator,
    )
)
@Operator.register_mul_handler(
    (
        ProductOperator,
        MixtureDensityOperator,
    )
)
@Operator.register_mul_handler((ProductDensityOperator, MixtureDensityOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, MixtureDensityOperator))
@Operator.register_mul_handler((QutipOperator, MixtureDensityOperator))
def _(x_op: Operator, y_op: MixtureDensityOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = tuple((x_op * term) * term.prefactor for term in y_op.terms)
    result = SumOperator(terms, system)
    return result


@Operator.register_mul_handler((MixtureDensityOperator, Operator))
@Operator.register_mul_handler((MixtureDensityOperator, ScalarOperator))
@Operator.register_mul_handler((MixtureDensityOperator, LocalOperator))
@Operator.register_mul_handler((MixtureDensityOperator, ProductOperator))
@Operator.register_mul_handler((MixtureDensityOperator, ProductDensityOperator))
@Operator.register_mul_handler((MixtureDensityOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((MixtureDensityOperator, QutipOperator))
def _(
    x_op: MixtureDensityOperator,
    y_op: Operator,
):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = []
    for term in x_op.terms:
        prefactor = term.prefactor
        if prefactor == 0:
            continue
        new_term = (term * y_op) * prefactor
        terms.append(new_term)

    return SumOperator(tuple(terms), system)


# MixtureDensityOperator times SumOperators
# and its derivatives


@Operator.register_mul_handler((MixtureDensityOperator, MixtureDensityOperator))
@Operator.register_mul_handler((MixtureDensityOperator, OneBodyOperator))
@Operator.register_mul_handler((MixtureDensityOperator, SumOperator))
def _(
    x_op: MixtureDensityOperator,
    y_op: Operator,
):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = []
    for term in x_op.terms:
        prefactor = term.prefactor
        if prefactor == 0:
            continue
        new_term = term * y_op
        new_term = new_term * prefactor
        terms.append(new_term)

    result = SumOperator(tuple(terms), system)
    return result


@Operator.register_mul_handler((OneBodyOperator, MixtureDensityOperator))
@Operator.register_mul_handler((SumOperator, MixtureDensityOperator))
def _(
    x_op: SumOperator,
    y_op: MixtureDensityOperator,
):

    terms = []
    for term in y_op.terms:
        prefactor = term.prefactor
        if prefactor == 0:
            continue
        new_term = x_op * term
        new_term = new_term * prefactor
        terms.append(new_term)

    result = SumOperator(tuple(terms), x_op.system or y_op.system)
    return result


# ########  GibbsDensityOperators


@Operator.register_mul_handler((ScalarOperator, GibbsDensityOperator))
def _(x_op: ScalarOperator, y_op: GibbsDensityOperator):

    y_qutip = y_op.to_qutip()
    result = QutipOperator(x_op.prefactor * y_qutip, x_op.system or y_op.system)
    return result


@Operator.register_mul_handler((GibbsDensityOperator, ScalarOperator))
def _(x_op: GibbsDensityOperator, y_op: ScalarOperator):
    x_qutip = x_op.to_qutip()
    result = QutipOperator(y_op.prefactor * x_qutip(), x_op.system or y_op.system)
    return result


# GibbsDensityOperators times any other operator


@Operator.register_mul_handler((GibbsDensityOperator, GibbsDensityOperator))
@Operator.register_mul_handler((LocalOperator, GibbsDensityOperator))
@Operator.register_mul_handler((ProductOperator, GibbsDensityOperator))
@Operator.register_mul_handler((OneBodyOperator, GibbsDensityOperator))
@Operator.register_mul_handler((ProductDensityOperator, GibbsDensityOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, GibbsDensityOperator))
@Operator.register_mul_handler((SumOperator, GibbsDensityOperator))
@Operator.register_mul_handler((OneBodyOperator, GibbsDensityOperator))
@Operator.register_mul_handler((MixtureDensityOperator, GibbsDensityOperator))
# ## and backward
@Operator.register_mul_handler((GibbsDensityOperator, LocalOperator))
@Operator.register_mul_handler((GibbsDensityOperator, ProductOperator))
@Operator.register_mul_handler((GibbsDensityOperator, OneBodyOperator))
@Operator.register_mul_handler((GibbsDensityOperator, ProductDensityOperator))
@Operator.register_mul_handler((GibbsDensityOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((GibbsDensityOperator, SumOperator))
@Operator.register_mul_handler((GibbsDensityOperator, OneBodyOperator))
@Operator.register_mul_handler((GibbsDensityOperator, MixtureDensityOperator))
def _(
    x_op: Operator,
    y_op: Operator,
):
    return x_op.to_qutip_operator() * y_op.to_qutip_operator()


# ############################
#    GibbsProductOperators
# ############################


@Operator.register_mul_handler(
    (GibbsProductDensityOperator, GibbsProductDensityOperator)
)
def _(x_op: GibbsProductDensityOperator, y_op: GibbsProductDensityOperator):
    return x_op.to_product_state() * y_op.to_product_state()


# times ScalarOperator, LocalOperator, ProductOperator
@Operator.register_mul_handler((GibbsProductDensityOperator, ScalarOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, LocalOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, ProductOperator))
def _(x_op: GibbsProductDensityOperator, y_op: Operator):
    return x_op.to_product_state() * y_op


@Operator.register_mul_handler((ScalarOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((LocalOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((ProductOperator, GibbsProductDensityOperator))
def _(x_op: Operator, y_op: GibbsProductDensityOperator):
    y_prod = y_op.to_product_state()
    return x_op * y_prod
