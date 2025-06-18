"""
Bindings for overloading Arithmetic Operations with Operators and Numbers.
"""

from numbers import Number
from typing import Union

import numpy as np
from qutip import Qobj

from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.quadratic import QuadraticFormOperator
from alpsqutip.operators.qutip import QutipOperator

# ##########################################
#
#        Arithmetic for ScalarOperators
#
# #########################################

NUMERIC_TYPES = tuple((Number, int, float, complex, np.float64, np.complex128))
TYPES_WITH_PREFACTOR = (ScalarOperator, ProductOperator, QutipOperator)
SUM_TYPES = (SumOperator, OneBodyOperator)


@Operator.register_add_handler(
    [
        (Operator, Operator),
        (ProductOperator, OneBodyOperator),
    ]
)
def _standar_sum_operator(op1: Operator, op2: Operator):
    system = op1.system.union(op2.system)
    return SumOperator(tuple((op1, op2)), system)


@Operator.register_add_handler(
    (
        ScalarOperator,
        ScalarOperator,
    )
)
def _(x_op: ScalarOperator, y_op: ScalarOperator):
    return ScalarOperator(x_op.prefactor + y_op.prefactor, x_op.system or y_op.system)


@Operator.register_mul_handler(
    (
        ScalarOperator,
        ScalarOperator,
    )
)
def _(x_op: ScalarOperator, y_op: ScalarOperator):
    return ScalarOperator(x_op.prefactor * y_op.prefactor, x_op.system or y_op.system)


@Operator.register_add_handler([(ScalarOperator, t) for t in NUMERIC_TYPES])
def _(x_op: ScalarOperator, y_value: Number):
    return ScalarOperator(x_op.prefactor + y_value, x_op.system)


@Operator.register_mul_handler(
    [(ScalarOperator, num_type) for num_type in NUMERIC_TYPES]
)
def _(x_op: ScalarOperator, y_value: Number):
    return ScalarOperator(x_op.prefactor * y_value, x_op.system)


@Operator.register_mul_handler(
    [(num_type, ScalarOperator) for num_type in NUMERIC_TYPES]
)
def _(y_value: Number, x_op: ScalarOperator):
    return ScalarOperator(x_op.prefactor * y_value, x_op.system)


# #########################################
#
#        Arithmetic for LocalOperator
#
# #########################################


@Operator.register_add_handler(
    [
        (
            LocalOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(x_op: LocalOperator, y_val: Number):
    return LocalOperator(x_op.site, x_op.operator + y_val, x_op.system)


@Operator.register_add_handler(
    (
        LocalOperator,
        ScalarOperator,
    )
)
def _(x_op: LocalOperator, y_op: ScalarOperator):
    system = x_op.system.union(y_op.system)
    return LocalOperator(x_op.site, x_op.operator + y_op.prefactor, system)


@Operator.register_add_handler(
    (
        LocalOperator,
        LocalOperator,
    )
)
def _(x_op: LocalOperator, y_op: LocalOperator):
    system = x_op.system.union(y_op.system)
    if x_op.site == y_op.site:
        return LocalOperator(x_op.site, x_op.operator + y_op.operator, system)

    return OneBodyOperator((x_op, y_op), system)


@Operator.register_mul_handler(
    [
        (
            LocalOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(x_op: LocalOperator, y_val: Number):
    return LocalOperator(x_op.site, x_op.operator * y_val, x_op.system)


@Operator.register_mul_handler(
    [
        (
            num_type,
            LocalOperator,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(y_val: Number, x_op: LocalOperator):
    return LocalOperator(x_op.site, x_op.operator * y_val, x_op.system)


@Operator.register_mul_handler(
    (
        LocalOperator,
        ScalarOperator,
    )
)
def _(x_op: LocalOperator, y_op: ScalarOperator):
    return LocalOperator(
        x_op.site, x_op.operator * y_op.prefactor, x_op.system or y_op.system
    )


@Operator.register_mul_handler(
    (
        ScalarOperator,
        LocalOperator,
    )
)
def _(y_op: ScalarOperator, x_op: LocalOperator):
    return LocalOperator(
        x_op.site, x_op.operator * y_op.prefactor, x_op.system or y_op.system
    )


@Operator.register_mul_handler(
    (
        LocalOperator,
        LocalOperator,
    )
)
def _(x_op: LocalOperator, y_op: LocalOperator):
    site_x = x_op.site
    site_y = y_op.site
    system = x_op.system or y_op.system
    if site_x == site_y:
        return LocalOperator(site_x, x_op.operator * y_op.operator, system)
    return ProductOperator(
        sites_operators={
            site_x: x_op.operator,
            site_y: y_op.operator,
        },
        prefactor=1,
        system=system,
    )


# #########################################
#
#        Arithmetic for ProductOperator
#
# #########################################


@Operator.register_mul_handler(
    (
        ProductOperator,
        ProductOperator,
    )
)
def _(x_op: ProductOperator, y_op: ProductOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    site_op = x_op.sites_op.copy()
    site_op_y = y_op.sites_op
    for site, op_local in site_op_y.items():
        site_op[site] = site_op[site] * op_local if site in site_op else op_local
    prefactor = x_op.prefactor * y_op.prefactor
    if len(site_op) == 0 or prefactor == 0:
        return ScalarOperator(prefactor, system)
    if len(site_op) == 1:
        site, op_local = next(iter(site_op.items()))
        return LocalOperator(site, op_local * prefactor, system)
    return ProductOperator(site_op, prefactor, system)


@Operator.register_mul_handler(
    [
        (
            ProductOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(x_op: ProductOperator, y_value: Number):
    if y_value:
        prefactor = x_op.prefactor * y_value
        return ProductOperator(x_op.sites_op, prefactor, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler(
    [
        (
            num_type,
            ProductOperator,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(y_value: Number, x_op: ProductOperator):
    if y_value:
        prefactor = x_op.prefactor * y_value
        return ProductOperator(x_op.sites_op, prefactor, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler(
    (
        ProductOperator,
        ScalarOperator,
    )
)
def _(x_op: ProductOperator, y_op: ScalarOperator):
    prefactor = y_op.prefactor
    if prefactor:
        prefactor = x_op.prefactor * prefactor
        return ProductOperator(x_op.sites_op, prefactor, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler(
    (
        ScalarOperator,
        ProductOperator,
    )
)
def _(
    y_op: ScalarOperator,
    x_op: ProductOperator,
):
    prefactor = y_op.prefactor
    if prefactor:
        prefactor = x_op.prefactor * prefactor
        return ProductOperator(x_op.sites_op, prefactor, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler(
    (
        ProductOperator,
        LocalOperator,
    )
)
def _(x_op: ProductOperator, y_op: LocalOperator):
    site = y_op.site
    op_local = y_op.operator
    system = x_op.system * y_op.system if x_op.system else y_op.system
    site_op = x_op.sites_op.copy()
    if site in site_op:
        op_local = site_op[site] * op_local

    site_op[site] = op_local

    if len(site_op) == 1:
        site, op_local = next(iter(site_op.items()))
        return LocalOperator(site, op_local * x_op.prefactor, system)
    return ProductOperator(site_op, x_op.prefactor, system)


@Operator.register_mul_handler(
    (
        LocalOperator,
        ProductOperator,
    )
)
def _(y_op: LocalOperator, x_op: ProductOperator):
    site = y_op.site
    op_local = y_op.operator
    system = x_op.system * y_op.system if x_op.system else y_op.system
    site_op = x_op.sites_op.copy()
    if site in site_op:
        op_local = op_local * site_op[site]

    site_op[site] = op_local

    if len(site_op) == 1:
        site, op_local = next(iter(site_op.items()))
        return LocalOperator(site, op_local * x_op.prefactor, system)
    return ProductOperator(site_op, x_op.prefactor, system)


# #######################################################
#               Sum operators
# #######################################################


# Sum with numbers


@Operator.register_add_handler(
    [
        (
            SumOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(x_op: SumOperator, y_value: Number):
    return x_op + ScalarOperator(y_value, x_op.system)


@Operator.register_mul_handler(
    [
        (
            SumOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(x_op: SumOperator, y_value: Number):
    if y_value == 0:
        return ScalarOperator(0, x_op.system)

    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return SumOperator(terms, x_op.system, isherm).simplify()


@Operator.register_mul_handler(
    [
        (
            num_type,
            SumOperator,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(y_value: Number, x_op: SumOperator):
    if y_value == 0:
        return ScalarOperator(0, x_op.system)

    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return SumOperator(terms, x_op.system, isherm).simplify()


# SumOperator times ScalarOperator


@Operator.register_mul_handler(
    (
        SumOperator,
        ScalarOperator,
    )
)
def _(x_op: SumOperator, y_op: ScalarOperator):
    system = x_op.system or y_op.system
    y_value = y_op.prefactor
    if y_value == 0:
        return ScalarOperator(0, system)

    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return SumOperator(terms, system, isherm)


@Operator.register_mul_handler(
    (
        ScalarOperator,
        SumOperator,
    )
)
def _(y_op: ScalarOperator, x_op: SumOperator):
    system = x_op.system or y_op.system
    y_value = y_op.prefactor
    if y_value == 0:
        return ScalarOperator(0, system)

    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return SumOperator(terms, system, isherm)


# Sum with LocalOperator


@Operator.register_mul_handler(
    [
        (
            SumOperator,
            LocalOperator,
        ),
        (
            OneBodyOperator,
            LocalOperator,
        ),
    ]
)
def _(
    x_op: SumOperator,
    y_op: LocalOperator,
):
    system = x_op.system * y_op.system if x_op.system else y_op.system

    terms_it = (y_op * term for term in x_op.terms)
    terms = tuple(term for term in terms_it if bool(term))
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and y_op.isherm
    return SumOperator(terms, system, isherm)


@Operator.register_mul_handler(
    [
        (
            LocalOperator,
            SumOperator,
        ),
        (
            LocalOperator,
            OneBodyOperator,
        ),
    ]
)
def _(y_op: LocalOperator, x_op: SumOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system

    terms_it = (y_op * term for term in x_op.terms)
    terms = tuple(term for term in terms_it if bool(term))
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and y_op.isherm
    return SumOperator(terms, system, isherm)


# SumOperator and any Operator


@Operator.register_add_handler(
    [
        (
            SumOperator,
            op_type,
        )
        for op_type in (
            Operator,
            ScalarOperator,
            LocalOperator,
            ProductOperator,
            QutipOperator,
            OneBodyOperator,
            QuadraticFormOperator,
        )
    ]
)
def _(x_op: SumOperator, y_op: Operator):
    system = x_op.system or y_op.system
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and y_op.isherm
    return SumOperator(terms, system, isherm)


# Sum another sum operator
@Operator.register_add_handler(
    (
        SumOperator,
        SumOperator,
    )
)
def _(x_op: SumOperator, y_op: SumOperator):
    system = x_op.system or y_op.system
    terms = x_op.terms + y_op.terms
    isherm = x_op._isherm and y_op._isherm
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(terms, system, isherm)


@Operator.register_mul_handler(
    tuple(
        (sum_type_1, sum_type_2) for sum_type_1 in SUM_TYPES for sum_type_2 in SUM_TYPES
    )
)
def _(x_op: SumOperator, y_op: SumOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = tuple(
        factor_x * factor_y for factor_x in x_op.terms for factor_y in y_op.terms
    )
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]

    if all(
        acts_over and len(acts_over) < 2
        for acts_over in (term.acts_over() for term in terms)
    ):
        return OneBodyOperator(terms, system, False)
    return SumOperator(terms, system)


@Operator.register_mul_handler(
    [
        (sum_type, op_type)
        for sum_type in SUM_TYPES
        for op_type in (
            Operator,
            ProductOperator,
            QutipOperator,
        )
    ]
)
def _(x_op: SumOperator, y_op: Operator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = tuple(factor_x * y_op for factor_x in x_op.terms)
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(terms, system)


@Operator.register_mul_handler(
    [
        (op_type, sum_type)
        for sum_type in SUM_TYPES
        for op_type in (
            Operator,
            ProductOperator,
            QutipOperator,
        )
    ]
)
def _(y_op: Operator, x_op: SumOperator):
    system = x_op.system.union(y_op.system)
    terms = tuple(y_op * factor_x for factor_x in x_op.terms)
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(terms, system)


# ######################
#
#   OneBodyOperator
#
# ######################


@Operator.register_add_handler(
    (
        OneBodyOperator,
        OneBodyOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: OneBodyOperator):
    system = x_op.system or y_op.system
    terms = x_op.terms + y_op.terms
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_add_handler(
    [
        (
            OneBodyOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(x_op: OneBodyOperator, y_value: Number):
    system = x_op.system
    y_op = ScalarOperator(y_value, system)
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_add_handler(
    (
        OneBodyOperator,
        ScalarOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: ScalarOperator):
    system = x_op.system.union(y_op.system)
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    result = OneBodyOperator(terms, system)
    return result


@Operator.register_add_handler(
    (
        OneBodyOperator,
        LocalOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: LocalOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler(
    [
        (
            OneBodyOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(x_op: OneBodyOperator, y_value: Number):
    system = x_op.system
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler(
    [
        (
            num_type,
            OneBodyOperator,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(y_value: Number, x_op: OneBodyOperator):
    system = x_op.system
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler(
    (
        OneBodyOperator,
        ScalarOperator,
    )
)
def _(x_op: OneBodyOperator, y_op: ScalarOperator):
    system = x_op.system
    y_value = y_op.prefactor
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler(
    (
        ScalarOperator,
        OneBodyOperator,
    )
)
def _(y_op: ScalarOperator, x_op: OneBodyOperator):
    system = x_op.system
    y_value = y_op.prefactor
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


# ######################
#
#   LocalOperator
#
# ######################


@Operator.register_add_handler(
    (
        ScalarOperator,
        LocalOperator,
    )
)
def _(x_op: ScalarOperator, y_op: LocalOperator):
    if x_op.prefactor == 0:
        return y_op

    system = y_op.system or x_op.system
    site = y_op.site
    return LocalOperator(site, y_op.operator + x_op.prefactor, system)


# ######################
#
#   ProductOperator
#
# ######################


@Operator.register_add_handler(
    [
        (
            ProductOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(x_op: ProductOperator, y_value: Number):
    site_op = x_op.sites_op.copy()
    prefactor = x_op.prefactor
    system = x_op.system
    if len(site_op) == 0:
        return ScalarOperator(prefactor + y_value, system)
    if len(site_op) == 1:
        first_site, first_loc_op = next(iter(site_op.items()))
        return LocalOperator(first_site, first_loc_op * prefactor + y_value, system)
    y_op = ScalarOperator(y_value, system)
    return SumOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )


@Operator.register_add_handler(
    [
        (
            ProductOperator,
            ProductOperator,
        ),
        (
            ScalarOperator,
            ProductOperator,
        ),
        (
            ProductOperator,
            ScalarOperator,
        ),
    ]
)
def _(x_op: ProductOperator, y_op: ProductOperator):
    system = x_op.system or y_op.system
    site_op_x = x_op.sites_op
    site_op_y = y_op.sites_op
    if len(site_op_x) > 1 or len(site_op_y) > 1:
        return SumOperator(
            (
                x_op,
                y_op,
            ),
            system,
        )
    return x_op.simplify() + y_op.simplify()


@Operator.register_add_handler(
    (
        ProductOperator,
        LocalOperator,
    )
)
def _(x_op: ProductOperator, y_op: LocalOperator):
    system = x_op.system or y_op.system
    site_op_x = x_op.sites_op
    if len(site_op_x) > 1:
        return SumOperator(
            (
                x_op,
                y_op,
            ),
            system,
        )
    return x_op.simplify() + y_op.simplify()


# #######################################
#
#  QutipOperator
#
# #######################################


@Operator.register_add_handler(
    [
        (QutipOperator, op_type)
        for op_type in (
            Operator,
            ScalarOperator,
            LocalOperator,
            ProductOperator,
            OneBodyOperator,
        )
    ]
)
def _(x_op: QutipOperator, y_op: Operator):
    system = x_op.system or y_op.system
    return SumOperator(
        (
            x_op,
            y_op,
        ),
        system,
    )


# #################################
# Arithmetic
# #################################


# Sum Qutip operators
@Operator.register_add_handler(
    (
        QutipOperator,
        QutipOperator,
    )
)
def sum_qutip_operator_plus_operator(x_op: QutipOperator, y_op: QutipOperator):
    """Sum two qutip operators"""

    system = x_op.system.union(y_op.system)
    x_site_names = x_op.site_names
    y_site_names = y_op.site_names
    if x_site_names == y_site_names:
        return QutipOperator(
            x_op.operator * x_op.prefactor + y_op.operator * y_op.prefactor,
            system,
            names=x_site_names,
            prefactor=1,
        )
    block_set = set(x_site_names)
    block_set.update(y_site_names)
    if len(block_set) <= max(len(x_site_names), len(y_site_names)):
        block = sorted(block_set)
        qutip_sum_operator = x_op.to_qutip(tuple(block)) + y_op.to_qutip(tuple(block))
        return QutipOperator(
            qutip_sum_operator,
            system,
            names={site: i for i, site in enumerate(block)},
            prefactor=1,
        )

    return SumOperator(
        tuple(
            (
                x_op,
                y_op,
            )
        ),
        system,
    )


@Operator.register_add_handler(
    (
        ScalarOperator,
        QutipOperator,
    )
)
def sum_scalarop_with_qutipop(x_op: ScalarOperator, y_op: QutipOperator):
    """Sum a Scalar operator to a Qutip Operator"""
    system = y_op.system or x_op.system
    return QutipOperator(
        y_op.operator * y_op.prefactor + x_op.prefactor,
        system=system,
        names=y_op.site_names,
        prefactor=1,
    )


@Operator.register_add_handler(
    [
        (
            QutipOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def sum_qutip_operator_plus_number(x_op: QutipOperator, y_val: Union[Number, Qobj]):
    """Sum an operator and a number  or a Qobj"""
    return QutipOperator(
        x_op.operator * x_op.prefactor + y_val,
        x_op.system,
        names=x_op.site_names,
    )


@Operator.register_mul_handler(
    [
        (
            QutipOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def sum_qutip_operator_mul_number(x_op: QutipOperator, y_val: Union[Number, Qobj]):
    """Sum an operator and a number  or a Qobj"""
    return QutipOperator(
        x_op.operator,
        x_op.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor * y_val,
    )


@Operator.register_mul_handler(
    [
        (
            num_type,
            QutipOperator,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def sum_qutip_operator_mul_number_back(y_val: Number, x_op: QutipOperator):
    """Sum an operator and a number  or a Qobj"""
    return QutipOperator(
        x_op.operator,
        x_op.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor * y_val,
    )


@Operator.register_add_handler(
    [
        (
            op_type,
            QutipOperator,
        )
        for op_type in (LocalOperator,)
    ]
)
def _(x_op: Operator, y_qutip_op: QutipOperator):
    """Sum an operator and a number  or a Qobj"""
    return x_op.to_qutip_operator() + y_qutip_op


@Operator.register_mul_handler(
    [
        (
            type_op,
            QutipOperator,
        )
        for type_op in (
            Operator,
            LocalOperator,
            ProductOperator,
        )
    ]
)
def _(
    x_op: Operator,
    y_qutip_op: QutipOperator,
):
    """Multiply an operator and a number  or a Qobj"""
    return x_op.to_qutip_operator() * y_qutip_op


@Operator.register_mul_handler(
    [
        (
            QutipOperator,
            type_op,
        )
        for type_op in (
            Operator,
            LocalOperator,
            ProductOperator,
        )
    ]
)
def _(x_op: Operator, y_qutip_op: QutipOperator):
    """Multiply an operator and a number  or a Qobj"""
    return x_op * y_qutip_op.to_qutip_operator()


@Operator.register_mul_handler(
    (
        QutipOperator,
        QutipOperator,
    )
)
def mul_qutip_operator_qutip_operator(x_op: QutipOperator, y_op: QutipOperator):
    """Product of two qutip operators"""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    x_names = x_op.site_names
    y_names = y_op.site_names
    if x_names == y_names:
        return QutipOperator(
            x_op.operator * y_op.operator,
            system,
            names=x_names,
            prefactor=x_op.prefactor * y_op.prefactor,
        )
    names_set = set(x_names)
    names_set.update(y_names)
    block = tuple(sorted(names_set))
    operator_qutip = x_op.to_qutip(block) * y_op.to_qutip(block)
    return QutipOperator(
        operator_qutip,
        system,
        names={site: i for i, site in enumerate(block)},
        prefactor=1,
    )


@Operator.register_mul_handler(
    (
        ScalarOperator,
        QutipOperator,
    )
)
def mul_scalarop_with_qutipop(x_op: ScalarOperator, y_op: QutipOperator):
    """Sum a Scalar operator to a Qutip Operator"""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return QutipOperator(
        y_op.operator,
        names=y_op.site_names,
        prefactor=x_op.prefactor * y_op.prefactor,
        system=system,
    )


@Operator.register_mul_handler(
    (
        QutipOperator,
        ScalarOperator,
    )
)
def mul_qutipop_with_scalarop(y_op: QutipOperator, x_op: ScalarOperator):
    """Sum a Scalar operator to a Qutip Operator"""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return QutipOperator(
        y_op.operator,
        names=y_op.site_names,
        prefactor=x_op.prefactor * y_op.prefactor,
        system=system,
    )


@Operator.register_add_handler(
    [
        (
            num_type,
            QutipOperator,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def mul_number_and_qutipoperator(y_val: Number, x_op: QutipOperator):
    """product of a number and a QutipOperator."""
    return QutipOperator(
        x_op.operator,
        x_op.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor * y_val,
    )


# #####################
#
#  Quadratic
#
# #######################


@Operator.register_add_handler(
    [(QuadraticFormOperator, num_type) for num_type in NUMERIC_TYPES]
)
@Operator.register_add_handler(
    [
        (QuadraticFormOperator, op_type)
        for op_type in (OneBodyOperator, ScalarOperator, LocalOperator)
    ]
)
def _(qf_operator: QuadraticFormOperator, op_other: Operator):
    linear_term = qf_operator.linear_term
    if linear_term is None:
        if isinstance(op_other, NUMERIC_TYPES):
            op_other = ScalarOperator(op_other, qf_operator.system)
        linear_term = op_other
    else:
        linear_term = linear_term + op_other

    return QuadraticFormOperator(
        qf_operator.basis,
        qf_operator.weights,
        qf_operator.system,
        linear_term,
        qf_operator.offset,
    )


@Operator.register_add_handler(
    [(QuadraticFormOperator, op_type) for op_type in (ProductOperator, QutipOperator)]
)
def _(qf_operator: QuadraticFormOperator, op_other: Operator):
    offset = qf_operator.offset
    if offset is None:
        offset = op_other
    else:
        offset = offset + op_other

    return QuadraticFormOperator(
        qf_operator.basis,
        qf_operator.weights,
        qf_operator.system,
        qf_operator.linear_term,
        offset,
    )


# Multiplication


#  # Numbers
@Operator.register_mul_handler(
    [
        (
            num_type,
            QuadraticFormOperator,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(value: Number, qf_operator: QuadraticFormOperator):
    linear_term = qf_operator.linear_term
    offset = qf_operator.offset
    if linear_term is not None:
        linear_term = value * linear_term
    if offset is not None:
        offset = value * offset
    return QuadraticFormOperator(
        qf_operator.basis, qf_operator.weights, qf_operator.system, linear_term, offset
    )


@Operator.register_mul_handler(
    [
        (
            QuadraticFormOperator,
            num_type,
        )
        for num_type in NUMERIC_TYPES
    ]
)
def _(qf_operator: QuadraticFormOperator, value: Number):
    linear_term = qf_operator.linear_term
    offset = qf_operator.offset
    if linear_term is not None:
        linear_term = value * linear_term
    if offset is not None:
        offset = value * offset
    return QuadraticFormOperator(
        qf_operator.basis,
        [w * value for w in qf_operator.weights],
        qf_operator.system,
        linear_term,
        offset,
    )


# # Scalars


@Operator.register_mul_handler(
    (
        QuadraticFormOperator,
        ScalarOperator,
    )
)
def _(qf_operator: QuadraticFormOperator, sc_operator: ScalarOperator):
    system = qf_operator.system.union(sc_operator.system)
    linear_term = qf_operator.linear_term
    offset = qf_operator.offset
    value = sc_operator.prefactor
    if linear_term is not None:
        linear_term = value * linear_term
    if offset is not None:
        offset = value * offset
    return QuadraticFormOperator(
        qf_operator.basis,
        [w * value for w in qf_operator.weights],
        system,
        linear_term,
        offset,
    )


@Operator.register_mul_handler(
    (
        ScalarOperator,
        QuadraticFormOperator,
    )
)
def _(sc_operator: ScalarOperator, qf_operator: QuadraticFormOperator):
    system = qf_operator.system.union(sc_operator.system)
    linear_term = qf_operator.linear_term
    offset = qf_operator.offset
    value = sc_operator.prefactor
    if linear_term is not None:
        linear_term = value * linear_term
    if offset is not None:
        offset = value * offset
    return QuadraticFormOperator(
        qf_operator.basis,
        tuple((w * value for w in qf_operator.weights)),
        system,
        linear_term,
        offset,
    )


@Operator.register_mul_handler(
    [
        (op_type, QuadraticFormOperator)
        for op_type in (
            Operator,
            LocalOperator,
            ProductOperator,
            SumOperator,
            OneBodyOperator,
            QutipOperator,
        )
    ]
)
def _(op1: Operator, op2: QuadraticFormOperator):
    return op1 * op2.to_sum_operator()


@Operator.register_mul_handler(
    [
        (
            QuadraticFormOperator,
            op_type,
        )
        for op_type in (
            Operator,
            LocalOperator,
            ProductOperator,
            SumOperator,
            OneBodyOperator,
            QutipOperator,
        )
    ]
)
def _(op_1: QuadraticFormOperator, op_2: Operator):
    return op_1.to_sum_operator() * op_2
