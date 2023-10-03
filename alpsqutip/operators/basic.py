"""
Different representations for operators
"""

from functools import reduce
from typing import Optional

from numbers import Number

import numpy as np
import qutip
from qutip import Qobj

from alpsqutip.model import Operator, SystemDescriptor


class LocalOperator(Operator):
    """
    Operator acting over a single site.
    """

    def __init__(
        self,
        site,
        local_operator,
        system: Optional[SystemDescriptor] = None,
    ):
        assert isinstance(site, str)
        assert system is not None
        self.site = site
        self.operator = local_operator
        self.system = system

    def __bool__(self):
        operator = self.operator
        if isinstance(operator, Qobj):
            return operator.data.count_nonzero() > 0
        return bool(self.operator)

    def __neg__(self):
        return LocalOperator(self.site, -self.operator, self.system)

    def __pow__(self, exp):
        operator = self.operator
        if exp < 0 and hasattr(operator, "inv"):
            operator = operator.inv()
            exp = -exp

        return LocalOperator(self.site, operator**exp, self.system)

    def __repr__(self):
        return (
            f"Local Operator on site {self.site}:"
            f"\n {repr(self.operator.full())}"
        )

    def act_over(self):
        return set((self.site,))

    def dag(self):
        """
        Return the adjoint operator
        """
        operator = self.operator
        if operator.isherm:
            return self
        return LocalOperator(self.site, operator.dag(), self.system)

    def expm(self):
        return LocalOperator(self.site, self.operator.expm(), self.system)

    def inv(self):
        operator = self.operator
        system = self.system
        site = self.site
        return LocalOperator(
            site,
            operator.inv() if hasattr(operator, "inv") else 1 / operator,
            system,
        )

    @property
    def isherm(self) -> bool:
        operator = self.operator
        if isinstance(operator, (float, int)):
            return True
        if isinstance(operator, complex):
            return operator.imag == 0.0
        return operator.isherm

    def logm(self):
        def log_qutip(loc_op):
            evals, evecs = loc_op.eigenstates()
            evals[abs(evals) < 1.0e-30] = 1.0e-30
            return sum(
                np.log(e_val) * e_vec * e_vec.dag()
                for e_val, e_vec in zip(evals, evecs)
            )

        return LocalOperator(self.site, log_qutip(self.operator), self.system)

    def partial_trace(self, sites: list):
        system = self.system
        if system is None:
            if self.site in sites:
                return self
            return ProductOperator({}, self.operator.tr())

        dimensions = system.dimensions
        subsystem = system.subsystem(sites)
        local_sites = subsystem.sites
        site = self.site
        prefactors = [
            d
            for s, d in dimensions.items()
            if s != site and s not in local_sites
        ]

        if len(prefactors) > 0:
            prefactor = reduce(lambda x, y: x * y, prefactors)
        else:
            prefactor = 1

        local_op = self.operator
        if site not in local_sites:
            return ScalarOperator(local_op.tr() * prefactor, subsystem)

        return LocalOperator(site, local_op * prefactor, subsystem)

    def to_qutip(self):
        """Convert to a Qutip object"""
        site = self.site
        dimensions = self.system.dimensions
        operator = self.operator
        if isinstance(operator, (int, float, complex)):
            operator = qutip.qeye(dimensions[site]) * operator
        elif isinstance(operator, Operator):
            operator = operator.to_qutip()

        return qutip.tensor(
            [
                operator if s == site else qutip.qeye(d)
                for s, d in dimensions.items()
            ]
        )

    def tr(self):
        result = self.partial_trace([])
        return result.prefactor


class ProductOperator(Operator):
    """Product of operators acting over different sites"""

    def __init__(
        self,
        sites_operators: dict,
        prefactor=1.0,
        system: Optional[SystemDescriptor] = None,
    ):
        assert system is not None
        remove_numbers = False
        for site, local_op in sites_operators.items():
            if isinstance(local_op, (int, float, complex)):
                prefactor *= local_op
                remove_numbers = True

        if remove_numbers:
            sites_operators = {
                s: local_op
                for s, local_op in sites_operators.items()
                if not isinstance(local_op, (int, float, complex))
            }

        self.sites_op = sites_operators
        if any(
            op.data.count_nonzero() == 0 for op in sites_operators.values()
        ):
            prefactor = 0
            self.sites_op = {}
        self.prefactor = prefactor
        self.system = system
        if system is not None:
            self.size = len(system.sites)
            self.dimensions = {
                name: site["dimension"] for name, site in system.sites.items()
            }

    def __bool__(self):
        return bool(self.prefactor) and all(
            bool(factor) for factor in self.sites_op
        )

    def __neg__(self):
        return ProductOperator(self.sites_op, -self.prefactor, self.system)

    def __pow__(self, exp):
        return ProductOperator(
            {s: op**exp for s, op in self.sites_op.items()},
            self.prefactor**exp,
            self.system,
        )

    def __repr__(self):
        result = "  " + str(self.prefactor) + " * (\n  "
        result += "\n  ".join(
            f"({item[1].full()} <-  {item[0]})"
            for item in self.sites_op.items()
        )
        result += " )"
        return result

    def act_over(self):
        return set((site for site in self.sites_op))

    def dag(self):
        """
        Return the adjoint operator
        """
        sites_op_dag = {key: op.dag() for key, op in self.sites_op.items()}
        prefactor = self.prefactor
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        return ProductOperator(sites_op_dag, prefactor, self.system)

    def expm(self):
        sites_op = self.sites_op
        n_ops = len(sites_op)
        if n_ops == 0:
            return ScalarOperator(np.exp(self.prefactor), self.system)
        if n_ops == 1:
            site, operator = next(iter(sites_op.items()))
            result = LocalOperator(
                site, (self.prefactor * operator).expm(), self.system
            )
            return result
        result = super().expm()
        return result

    def inv(self):
        sites_op = self.sites_op
        system = self.system
        prefactor = self.prefactor

        n_ops = len(sites_op)
        sites_op = {
            site: op_local.inv() for site, op_local in sites_op.items()
        }
        if n_ops == 1:
            site, op_local = next(iter(sites_op.items()))
            return LocalOperator(site, op_local / prefactor, system)
        return ProductOperator(sites_op, 1 / prefactor, system)

    @property
    def isherm(self) -> bool:
        if not all(loc_op.isherm for loc_op in self.sites_op.values()):
            return False
        return isinstance(self.prefactor, (int, float))

    def logm(self):
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators.arithmetic import OneBodyOperator

        def log_qutip(loc_op):
            evals, evecs = loc_op.eigenstates()
            evals[abs(evals) < 1.0e-30] = 1.0e-30
            return sum(
                np.log(e_val) * e_vec * e_vec.dag()
                for e_val, e_vec in zip(evals, evecs)
            )

        system = self.system
        terms = tuple(
            LocalOperator(site, log_qutip(loc_op), system)
            for site, loc_op in self.sites_op.items()
        )
        return OneBodyOperator(terms, system, False) + np.log(self.prefactor)

    def partial_trace(self, sites: list):
        full_system_sites = self.system.sites
        dimensions = self.dimensions
        sites_in = tuple(s for s in sites if s in full_system_sites)
        sites_out = tuple(s for s in full_system_sites if s not in sites_in)
        subsystem = self.system.subsystem(sites_in)
        sites_op = self.sites_op
        prefactors = [
            sites_op[s].tr() if s in sites_op else dimensions[s]
            for s in sites_out
        ]
        sites_op = {s: o for s, o in sites_op.items() if s in sites_in}
        prefactor = self.prefactor
        for factor in prefactors:
            if factor == 0:
                return ScalarOperator(factor, subsystem)
            prefactor *= factor

        if len(sites_op) == 0:
            return ScalarOperator(prefactor, subsystem)
        return ProductOperator(sites_op, prefactor, subsystem)

    def simplify(self):
        nops = len(self.sites_op)
        if nops == 0:
            return ScalarOperator(self.prefactor, self.system)
        if nops == 1:
            site, op_local = next(iter(self.sites_op.items()))
            return LocalOperator(site, self.prefactor * op_local, self.system)
        return self

    def to_qutip(self):
        ops = self.sites_op
        system = self.system
        if system:
            return self.prefactor * qutip.tensor(
                [
                    ops.get(site, None) if site in ops else qutip.qeye(dim)
                    for site, dim in self.system.dimensions.items()
                ]
            )
        return self.prefactor * qutip.tensor(ops.values())

    def tr(self):
        result = self.partial_trace([])
        return result.prefactor


class ScalarOperator(ProductOperator):
    """A product operator that acts trivially on every subsystem"""

    def __init__(self, prefactor, system):
        assert system is not None
        super().__init__({}, prefactor, system)

    def __neg__(self):
        return ScalarOperator(-self.prefactor, self.system)

    def __repr__(self):
        result = str(self.prefactor) + " * Identity "
        return result

    def act_over(self):
        return set()

    def dag(self):
        if isinstance(self.prefactor, complex):
            return ScalarOperator(self.prefactor.conjugate(), self.system)
        return self

    @property
    def isherm(self):
        prefactor = self.prefactor
        return not (isinstance(prefactor, complex) and prefactor.imag != 0)

    def logm(self):
        return ScalarOperator(np.log(self.prefactor), self.system)


# ##########################################
#
#        Arithmetic for ScalarOperators
#
# #########################################


@Operator.register_add_handler(
    (
        ScalarOperator,
        ScalarOperator,
    )
)
def _(x_op: ScalarOperator, y_op: ScalarOperator):
    return ScalarOperator(
        x_op.prefactor + y_op.prefactor, x_op.system or y_op.system
    )


@Operator.register_mul_handler(
    (
        ScalarOperator,
        ScalarOperator,
    )
)
def _(x_op: ScalarOperator, y_op: ScalarOperator):
    return ScalarOperator(
        x_op.prefactor * y_op.prefactor, x_op.system or y_op.system
    )


@Operator.register_add_handler(
    (
        ScalarOperator,
        Number,
    )
)
def _(x_op: ScalarOperator, y_value: Number):
    return ScalarOperator(x_op.prefactor + y_value, x_op.system)


@Operator.register_mul_handler(
    (
        ScalarOperator,
        Number,
    )
)
def _(x_op: ScalarOperator, y_value: Number):
    return ScalarOperator(x_op.prefactor * y_value, x_op.system)


@Operator.register_mul_handler(
    (
        Number,
        ScalarOperator,
    )
)
def _(y_value: Number, x_op: ScalarOperator):
    return ScalarOperator(x_op.prefactor * y_value, x_op.system)


# #########################################
#
#        Arithmetic for LocalOperator
#
# #########################################


@Operator.register_add_handler(
    (
        LocalOperator,
        Number,
    )
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
    return LocalOperator(
        x_op.site, x_op.operator + y_op.prefactor, x_op.system or y_op.system
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
        {
            site_x: x_op.operator,
            site_y: y_op.operator,
        },
        1,
        system,
    )


@Operator.register_mul_handler(
    (
        LocalOperator,
        Number,
    )
)
def _(x_op: LocalOperator, y_value: Number):
    return LocalOperator(x_op.site, x_op.operator * y_value, x_op.system)


@Operator.register_mul_handler(
    (
        Number,
        LocalOperator,
    )
)
def _(y_value: Number, x_op: LocalOperator):
    return LocalOperator(x_op.site, x_op.operator * y_value, x_op.system)


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
    system = x_op.system or y_op.system
    site_op = x_op.sites_op.copy()
    site_op_y = y_op.sites_op
    for site, op_local in site_op_y.items():
        site_op[site] = (
            site_op[site] * op_local if site in site_op else op_local
        )
    prefactor = x_op.prefactor * y_op.prefactor
    if len(site_op) == 0 or prefactor == 0:
        return ScalarOperator(prefactor, system)
    if len(site_op) == 1:
        site, op_local = next(iter(site_op.items()))
        return LocalOperator(site, op_local * prefactor, system)
    return ProductOperator(site_op, prefactor, system)


@Operator.register_mul_handler(
    (
        ProductOperator,
        Number,
    )
)
def _(x_op: ProductOperator, y_value: Number):
    if y_value:
        prefactor = x_op.prefactor * y_value
        return ProductOperator(x_op.sites_op, prefactor, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler(
    (
        Number,
        ProductOperator,
    )
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
def _(y_op: ScalarOperator, x_op: ProductOperator):
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
    system = x_op.system or y_op.system
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
    system = x_op.system or y_op.system
    site_op = x_op.sites_op.copy()
    if site in site_op:
        op_local = op_local * site_op[site]

    site_op[site] = op_local

    if len(site_op) == 1:
        site, op_local = next(iter(site_op.items()))
        return LocalOperator(site, op_local * x_op.prefactor, system)
    return ProductOperator(site_op, x_op.prefactor, system)
