"""
Different representations for operators
"""
import sys
from functools import reduce
from typing import Optional

from numbers import Number

import numpy as np
import qutip
from qutip import Qobj

from alpsqutip.model import Operator, SystemDescriptor
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.settings import VERBOSITY_LEVEL


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
        self.site = site
        self.operator = local_operator
        self.system = system

    def __add__(self, operand):
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators.arithmetic import OneBodyOperator

        site = self.site
        if isinstance(operand, ScalarOperator):
            return LocalOperator(
                site, self.operator + operand.prefactor, self.system
            )

        if isinstance(operand, LocalOperator):
            system = self.system or operand.system
            if site == operand.site:
                return LocalOperator(
                    site, self.operator + operand.operator, system
                )
            return OneBodyOperator(
                tuple(
                    (
                        LocalOperator(site, self.operator, system),
                        LocalOperator(operand.site, operand.operator, system),
                    )
                ),
                system,
                check_and_convert=False,
            )

        if isinstance(operand, (int, float, complex)):
            return LocalOperator(site, self.operator + operand, self.system)

        if isinstance(operand, Qobj):
            return QutipOperator(operand) + self.to_qutip_operator()

        try:
            result = operand + self
        except RecursionError:
            if VERBOSITY_LEVEL > 0:
                print("recursion error", type(operand), type(self))
            sys.exit()
        return result

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
        return f"Local Operator on site {self.site}:\n {repr(self.operator.full())}"

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

    def __add__(self, operand):
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators.arithmetic import SumOperator

        if not bool(operand):
            return self
        if self.prefactor == 0:
            return operand

        if isinstance(operand, (int, float, complex)):
            operand = ScalarOperator(operand, self.system)
        else:
            operand = operand.simplify()

        self_simpl = self.simplify()
        if not isinstance(self_simpl, ProductOperator):
            if isinstance(operand, ProductOperator):
                self_simpl, operand = operand, self_simpl
            else:
                return self_simpl + operand
        if isinstance(operand, SumOperator):
            return SumOperator(
                (self_simpl,) + operand.terms,
                self.system,
                operand._isherm and self_simpl.isherm,
            )
        return SumOperator(
            tuple(
                (
                    self_simpl,
                    operand,
                )
            ),
            self.system,
            operand.isherm and self_simpl.isherm,
        )

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

    def __add__(self, operand):
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators.arithmetic import OneBodyOperator, SumOperator

        if isinstance(operand, ScalarOperator):
            operand = operand.prefactor
        if isinstance(operand, Number):
            return ScalarOperator(self.prefactor + operand, self.system)

        def add_product_operator():
            """add scalar to a product operator"""
            sites_op = operand.sites_op.copy()
            first_site, first_op = next(iter(sites_op.items()))
            sites_op[first_site] = first_op + self.prefactor
            return ProductOperator(sites_op, operand.prefactor, self.system)

        def add_default():
            """default case"""
            return SumOperator(
                tuple(
                    (
                        operand,
                        self,
                    )
                ),
                self.system or operand.system,
            )

        dispatch_table = {
            LocalOperator: (
                lambda: LocalOperator(
                    operand.site,
                    operand.operator + self.prefactor,
                    self.system,
                )
            ),
            OneBodyOperator: (
                lambda: OneBodyOperator(
                    tuple(term + self for term in operand.terms), self.system
                )
            ),
            ProductOperator: add_product_operator,
            SumOperator: (
                lambda: SumOperator(
                    operand.terms + (self,), self.system or operand.sytem
                )
            ),
        }

        return dispatch_table.get(type(operand), add_default)()

    def __no_mul__(self, operand):
        if isinstance(operand, ScalarOperator):
            return ScalarOperator(
                self.prefactor * operand.prefactor, self.system
            )
        if isinstance(operand, (int, float, complex)):
            return ScalarOperator(self.prefactor + operand, self.system)
        return super().__mul__(operand)

    def __neg__(self):
        return ScalarOperator(-self.prefactor, self.system)

    def __repr__(self):
        result = str(self.prefactor) + " * Identity "
        return result

    def __rmul__(self, operand):
        if isinstance(operand, ScalarOperator):
            return ScalarOperator(
                self.prefactor * operand.prefactor, self.system
            )
        if isinstance(operand, (int, float, complex)):
            return ScalarOperator(self.prefactor + operand, self.system)
        return super().__mul__(operand)

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
