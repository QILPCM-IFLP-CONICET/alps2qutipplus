"""
Bindings for arithmetic operations
"""

from numbers import Number

import numpy as np

from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.quadratic import QuadraticFormOperator
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states.arithmetic import MixtureDensityOperator
from alpsqutip.operators.states.basic import (
    DensityOperatorMixin,
    ProductDensityOperator,
)
from alpsqutip.operators.states.gibbs import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)
from alpsqutip.operators.states.qutip import QutipDensityOperator

BASIC_OPERATOR_TYPES = (
    Operator,
    ScalarOperator,
    LocalOperator,
    ProductOperator,
    QutipOperator,
    OneBodyOperator,
    QuadraticFormOperator,
)


NON_PRODUCT_BASIC_OPERATOR_TYPES = (
    Operator,
    QutipOperator,
    OneBodyOperator,
    QuadraticFormOperator,
)

DENSITY_OPERATOR_BASIC_TYPES = (
    DensityOperatorMixin,
    ProductDensityOperator,
    QutipDensityOperator,
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)

NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES = (
    DensityOperatorMixin,
    QutipDensityOperator,
    GibbsDensityOperator,
)

NUMERIC_TYPES = tuple((Number, int, float, complex, np.float64, np.complex128))
TYPES_WITH_PREFACTOR = (ScalarOperator, ProductOperator, QutipOperator)
SUM_TYPES = (SumOperator, OneBodyOperator)


# ####################################
#  Arithmetic
# ####################################


# #### Sums #############


@Operator.register_add_handler(
    [
        (MixtureDensityOperator, state_type)
        for state_type in DENSITY_OPERATOR_BASIC_TYPES
    ]
)
def _(x_op: MixtureDensityOperator, y_op: DensityOperatorMixin):
    terms = x_op.terms + (y_op,)
    # If there is just one term, return it:
    if len(terms) == 1:
        return terms[0]

    # For empty terms, return 0
    system = x_op.system or y_op.system
    if len(terms) == 0:
        return system.global_operator("zero")
    # General case
    return MixtureDensityOperator(terms, system)


@Operator.register_mul_handler(
    [(op_type, MixtureDensityOperator) for op_type in BASIC_OPERATOR_TYPES]
)
def _(y_op: Operator, x_op: MixtureDensityOperator):
    terms = tuple((y_op * term) * term.prefactor for term in x_op.terms)
    # If there is just one term, return it:
    if len(terms) == 1:
        return terms[0]

    # For empty terms, return 0
    system = x_op.system.union(y_op.system)
    if len(terms) == 0:
        return system.global_operator("zero")
    # General case
    return SumOperator(terms, system, y_op.isherm)


@Operator.register_add_handler(
    [
        (type_1, type_2)
        for type_1 in DENSITY_OPERATOR_BASIC_TYPES
        for type_2 in DENSITY_OPERATOR_BASIC_TYPES
    ]
)
def _(x_op, y_op):
    system = x_op.system * y_op.system
    return MixtureDensityOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )


@Operator.register_add_handler(
    [
        (type_1, type_2)
        for type_1 in BASIC_OPERATOR_TYPES
        for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES
    ]
)
@Operator.register_add_handler(
    [(SumOperator, type_2) for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES]
)
def _(x_op: Operator, y_op: DensityOperatorMixin):
    y_op = y_op.to_qutip_operator()
    if isinstance(y_op, QutipDensityOperator):
        prefactor = 1
        system = y_op.system
        names = y_op.site_names
        y_op = QutipOperator(y_op.operator, system, names, prefactor)
    return x_op + y_op


@Operator.register_mul_handler(
    [
        (type_1, type_2)
        for type_1 in BASIC_OPERATOR_TYPES
        for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES
    ]
)
@Operator.register_mul_handler(
    [(SumOperator, type_2) for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES]
)
def _(x_op: Operator, y_op: DensityOperatorMixin):
    y_op = y_op.to_qutip_operator()
    if isinstance(y_op, QutipDensityOperator):
        prefactor = 1
        system = y_op.system
        names = y_op.site_names
        y_op = QutipOperator(y_op.operator, system, names, prefactor)
    return x_op * y_op


@Operator.register_mul_handler(
    [
        (type_2, type_1)
        for type_1 in BASIC_OPERATOR_TYPES
        for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES
    ]
)
@Operator.register_add_handler(
    [(type_2, SumOperator) for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES]
)
def _(y_op: DensityOperatorMixin, x_op: Operator):
    y_op = y_op.to_qutip_operator()
    if isinstance(y_op, QutipDensityOperator):
        prefactor = 1
        system = y_op.system
        names = y_op.site_names
        y_op = QutipOperator(y_op.operator, system, names, prefactor)
    return y_op * x_op


# #### Products #############


@Operator.register_add_handler(
    [(type_1, ProductDensityOperator) for type_1 in BASIC_OPERATOR_TYPES]
)
@Operator.register_add_handler(
    (
        SumOperator,
        ProductDensityOperator,
    )
)
def _(x_op: Operator, y_op: DensityOperatorMixin):
    y_op = ProductOperator(y_op.sites_op, 1, y_op.system)
    return x_op + y_op


@Operator.register_mul_handler(
    [(type_1, ProductDensityOperator) for type_1 in BASIC_OPERATOR_TYPES]
)
def _(x_op: Operator, y_op: DensityOperatorMixin):
    y_op = ProductOperator(y_op.sites_op, 1, y_op.system)
    return x_op * y_op


@Operator.register_mul_handler(
    [(ProductDensityOperator, type_1) for type_1 in BASIC_OPERATOR_TYPES]
)
def _(y_op: DensityOperatorMixin, x_op: Operator):
    y_op = ProductOperator(y_op.sites_op, 1, y_op.system)
    return y_op * x_op


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

# ###   LocalOperator


# SumOperators


@Operator.register_mul_handler(
    (
        ProductDensityOperator,
        SumOperator,
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


######################


# ############    Mixtures  ##########################


# MixtureDensityOperator times SumOperators
# and its derivatives


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


# ############################
#    GibbsProductOperators
# ############################


@Operator.register_add_handler(
    [(GibbsProductDensityOperator, type_op) for type_op in BASIC_OPERATOR_TYPES]
)
@Operator.register_add_handler(
    (
        GibbsProductDensityOperator,
        SumOperator,
    )
)
def _(x_op: GibbsProductDensityOperator, y_op: ScalarOperator):
    system = x_op.system.union(y_op.system)
    x_op = x_op.to_product_state()
    x_op = ProductOperator(x_op.sites_op, 1, system)
    return SumOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )


@Operator.register_mul_handler(
    [
        (GibbsProductDensityOperator, type_op)
        for type_op in NON_PRODUCT_BASIC_OPERATOR_TYPES
    ]
)
def _(x_op: GibbsProductDensityOperator, y_op: Operator):
    system = x_op.system.union(y_op.system)
    x_op = x_op.to_product_state()
    x_op = ProductOperator(x_op.sites_op, 1, system)
    return x_op * y_op


@Operator.register_mul_handler(
    [
        (type_op, GibbsProductDensityOperator)
        for type_op in NON_PRODUCT_BASIC_OPERATOR_TYPES
    ]
)
@Operator.register_mul_handler((SumOperator, GibbsProductDensityOperator))
def _(y_op: Operator, x_op: GibbsProductDensityOperator):
    system = x_op.system.union(y_op.system)
    x_op = x_op.to_product_state()
    x_op = ProductOperator(x_op.sites_op, 1, system)
    return y_op * x_op


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


@Operator.register_add_handler(
    (
        ProductDensityOperator,
        ScalarOperator,
    )
)
def _(x_op: ProductOperator, y_op: ScalarOperator):
    site_op = x_op.sites_op.copy()
    prefactor = x_op.prefactor
    system = x_op.system or y_op.system
    if len(site_op) == 0:
        return ScalarOperator(prefactor + y_op.prefactor, system)
    if len(site_op) == 1:
        first_site, first_loc_op = next(iter(site_op.items()))
        return LocalOperator(
            first_site, first_loc_op * prefactor + y_op.prefactor, system
        )

    return SumOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )
