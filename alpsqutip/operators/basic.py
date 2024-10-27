"""
Different representations for operators
"""

import logging
from functools import reduce
from numbers import Number
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import qutip
from qutip import Qobj

from alpsqutip.model import SystemDescriptor
from alpsqutip.qutip_tools.tools import data_is_diagonal, data_is_scalar, data_is_zero


def check_multiplication(a, b, result, func=None) -> bool:
    if isinstance(a, Qobj) and isinstance(b, Qobj):
        return True
    if isinstance(a, Operator):
        a_qutip = a.to_qutip()
    else:
        a_qutip = a
    if isinstance(b, Operator):
        b_qutip = b.to_qutip()
    else:
        b_qutip = b
    q_trace = (a_qutip * b_qutip).tr()
    tr = result.tr()
    if func is None:
        where = ""
    elif isinstance(func, str):
        where = func
    else:
        where = f"{func}@{func.__module__}:{func.__code__.co_firstlineno}"
    assert (
        abs(q_trace - tr) < 1e-8
    ), f"{type(a)}*{type(b)}->{type(result)} ({where}) failed: traces are different  {tr}!={q_trace}"
    return True


def empty_op(op: Qobj) -> bool:
    """
    Check if op is an sparse operator without
    non-zero elements.
    """
    if not hasattr(op, "data"):
        if isinstance(op, ScalarOperator):
            return op.prefactor == 0
        if hasattr(op, "operator"):
            return empty_op(op.operator)
        if hasattr(op, "sites_op"):
            if op.prefactor == 0:
                return True
            return any(empty_op(op_l) for op_l in sites_op.values())
        raise ValueError(f"Operator of type {type(op)} is not allowed.")
    return data_is_zero(op.data)


def is_diagonal_op(op: Qobj) -> bool:
    """Check if op is a diagonal operator"""
    if not hasattr(op, "data"):
        if isinstance(op, ScalarOperator):
            return True
        if hasattr(op, "operator"):
            return is_diagonal_op(op.operator)
        if hasattr(op, "sites_op"):
            if op.prefactor == 0:
                return True
            return all(is_diagonal_op(op_l) for op_l in sites_op.values())
        raise ValueError(f"Operator of type {type(op)} is not allowed.")
    return data_is_diagonal(op.data)


def is_scalar_op(op: Qobj) -> bool:
    """
    Check if the operator is a
    multiple of the identity
    """
    if not hasattr(op, "data"):
        if isinstance(op, ScalarOperator):
            return True
        if hasattr(op, "operator"):
            return is_scalar_op(op.operator)
        if hasattr(op, "sites_op"):
            return all(is_scalar_op(site_op) for site_op in op.sites_op.values())
        raise ValueError(f"Operator of type {type(op)} is not allowed.")
    return data_is_scalar(op.data)


class Operator:
    """Base class for operators"""

    system: SystemDescriptor
    prefactor: float = 1.0

    # def __add__(self, term):
    #    # pylint: disable=import-outside-toplevel
    #    from alpsqutip.operators import SumOperator
    #
    #    return SumOperator([self, term], self.system)

    # TODO check the possibility of implementing this with multimethods
    __add__dispatch__: Dict[Tuple, Callable] = {}
    __mul__dispatch__: Dict[Tuple, Callable] = {}

    @staticmethod
    def register_add_handler(key: Tuple):
        def register_func(func):
            Operator.__add__dispatch__[key] = func
            return func

        return register_func

    @staticmethod
    def register_mul_handler(key: Tuple):
        def register_func(func):
            Operator.__mul__dispatch__[key] = func
            return func

        return register_func

    def __add__(self, term):
        # Use multiple dispatch to determine how to add
        func = self.__add__dispatch__.get((type(self), type(term)), None)
        if func is not None:
            return func(self, term)

        func = self.__add__dispatch__.get((type(term), type(self)), None)
        if func is not None:
            return func(term, self)

        for key, func in self.__add__dispatch__.items():
            lhf, rhf = key
            if isinstance(self, lhf) and isinstance(term, rhf):
                self.__add__dispatch__[
                    (
                        lhf,
                        rhf,
                    )
                ] = func
                return func(self, term)
            rhf, lhf = key
            if isinstance(self, lhf) and isinstance(term, rhf):
                self.__add__dispatch__[
                    (
                        lhf,
                        rhf,
                    )
                ] = lambda x, y: func(y, x)
                return func(term, self)

        raise ValueError(type(self), "cannot be added with ", type(term))

    def __mul__(self, factor):
        # Use multiple dispatch to determine how to multiply
        key = (
            type(self),
            type(factor),
        )
        func = self.__mul__dispatch__.get(key, None)

        if func is not None:
            result = func(self, factor)
            return result  # func(self, factor)

        for try_key, func in Operator.__mul__dispatch__.items():
            lhf, rhf = try_key
            if isinstance(self, lhf) and isinstance(factor, rhf):
                Operator.__mul__dispatch__[key] = func
                result = func(self, factor)
                return result  # func(self, factor)

        if not hasattr(factor, "to_qutip_operator"):
            raise ValueError(type(self), "cannot be multiplied with ", type(factor))
        factor = factor.to_qutip_operator()
        return self.to_qutip_operator() * factor

    def __neg__(self):
        return -(self.to_qutip_operator())

    def __sub__(self, operand):
        if operand is None:
            raise ValueError("None can not be an operand")
        neg_op = -operand
        return self + neg_op

    def __radd__(self, term):
        # Use multiple dispatch to determine how to add
        return self + term

    def __rmul__(self, factor):
        # Use __mul__dispatch__ to determine how to evaluate the product

        key = (
            type(factor),
            type(self),
        )
        func = self.__mul__dispatch__.get(key, None)
        if func is not None:
            result = func(factor, self)
            return result  # func(factor, self)
        for try_key, func in Operator.__mul__dispatch__.items():
            lhf, rhf = try_key
            if isinstance(factor, lhf) and isinstance(self, rhf):
                Operator.__mul__dispatch__[key] = func
                result = func(factor, self)
                return result  # func(factor, self)

        raise ValueError(type(self), "cannot be multiplied  with ", type(factor))
        # return factor.to_qutip_operator() * self.to_qutip_operator()

    def __rsub__(self, operand):
        if operand is None:
            raise ValueError("None can not be an operand")

        neg_self = -self
        return operand + neg_self

    def __pow__(self, exponent):
        if exponent is None:
            raise ValueError("None can not be an operand")

        return self.to_qutip_operator() ** exponent

    def __truediv__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return self * (1.0 / operand)
        if isinstance(operand, Operator):
            return self * operand.inv()
        raise ValueError("Division of an operator by ", type(operand), " not defined.")

    def _repr_latex_(self):
        """LaTeX Representation"""
        qutip_repr = self.to_qutip()
        if isinstance(qutip_repr, qutip.Qobj):
            # pylint: disable=protected-access
            parts = qutip_repr._repr_latex_().split("$")
            tex = parts[1] if len(parts) > 2 else "-?-"
        else:
            tex = str(qutip_repr)
        return f"${tex}$"

    def acts_over(self) -> Optional[set]:
        """
        Return the list of sites over which the operator acts nontrivially.
        If this cannot be determined, return None.
        """
        return None

    def dag(self):
        """Adjoint operator of quantum object"""
        return self.to_qutip_operator().dag()

    def flat(self):
        """simplifies sums and products"""
        return self

    @property
    def isherm(self) -> bool:
        """Check if the operator is hermitician"""
        return self.to_qutip().isherm

    @property
    def isdiagonal(self) -> bool:
        """Check if the operator is diagonal"""
        return False

    def expm(self):
        """Compute the exponential of the Qutip representation of the operator"""

        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from scipy.sparse.linalg import ArpackError

        from alpsqutip.operators.functions import eigenvalues
        from alpsqutip.operators.qutip import QutipOperator

        op_qutip = self.to_qutip()
        try:
            max_eval = eigenvalues(op_qutip, sort="high", sparse=True, eigvals=3)[0]
        except ArpackError:
            max_eval = max(op_qutip.diag())

        op_qutip = (op_qutip - max_eval).expm()
        return QutipOperator(op_qutip, self.system, prefactor=np.exp(max_eval))

    def inv(self):
        """the inverse of the operator"""
        return self.to_qutip_operator().inv()

    def logm(self):
        """Logarithm of the operator"""
        return self.to_qutip_operator().logm()

    def partial_trace(self, sites: Union[list, SystemDescriptor]):
        """Partial trace over sites not listed in `sites`"""
        raise NotImplementedError

    def simplify(self):
        """Returns a more efficient representation"""
        return self

    def to_qutip(self):
        """Convert to a Qutip object"""
        raise NotImplementedError

    def to_qutip_operator(self):
        """Produce a Qutip representation of the operator"""
        from alpsqutip.operators.qutip import QutipOperator

        qobj = self.to_qutip()
        if isinstance(qobj, qutip.Qobj):
            return QutipOperator(qobj, self.system)
        return ScalarOperator(qobj, self.system)

    # pylint: disable=invalid-name
    def tr(self):
        """The trace of the operator"""
        return self.partial_trace([]).prefactor

    def tidyup(self, atol=None):
        """remove tiny elements of the operator"""
        return self


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
            return not empty_op(operator)
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
        return f"Local Operator on site {self.site}:" f"\n {repr(self.operator.full())}"

    def acts_over(self):
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

    @property
    def isdiagonal(self) -> bool:
        return is_diagonal_op(self.operator)

    def logm(self):
        def log_qutip(loc_op):
            evals, evecs = loc_op.eigenstates()
            evals[abs(evals) < 1.0e-50] = 1.0e-50
            return sum(
                np.log(e_val) * e_vec * e_vec.dag()
                for e_val, e_vec in zip(evals, evecs)
            )

        return LocalOperator(self.site, log_qutip(self.operator), self.system)

    def partial_trace(self, sites: Union[list, SystemDescriptor]):
        system = self.system
        assert system is not None
        dimensions = system.dimensions
        subsystem = (
            sites if isinstance(sites, SystemDescriptor) else system.subsystem(sites)
        )
        local_sites = subsystem.sites
        site = self.site
        prefactors = [
            d for s, d in dimensions.items() if s != site and s not in local_sites
        ]

        if len(prefactors) > 0:
            prefactor = reduce(lambda x, y: x * y, prefactors)
        else:
            prefactor = 1

        local_op = self.operator
        if site not in local_sites:
            return ScalarOperator(local_op.tr() * prefactor, subsystem)
        return LocalOperator(site, local_op * prefactor, subsystem)

    def simplify(self):
        # TODO: reduce multiples of the identity to ScalarOperators
        operator = self.operator
        if not is_scalar_op(operator):
            return self
        value = operator[0, 0] * self.prefactor
        return ScalarOperator(value, self.system)

    def to_qutip(self):
        """Convert to a Qutip object"""
        site = self.site
        system = self.system
        dimensions = system.dimensions
        operator = self.operator
        if isinstance(operator, (int, float, complex)):
            return qutip.qeye(dimensions[site]) * operator
        if isinstance(operator, Operator):
            operator = operator.to_qutip()
        factors_dict = {
            s: operator if s == site else qutip.qeye(d) for s, d in dimensions.items()
        }
        return qutip.tensor(*(factors_dict[site] for site in sorted(dimensions)))

    def tr(self):
        result = self.partial_trace([])
        return result.prefactor

    def tidyup(self, atol=None):
        """remove tiny elements of the operator"""
        return LocalOperator(self.site, self.operator.tidyup(atol), self.system)


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
        if any(empty_op(op) for op in sites_operators.values()):
            prefactor = 0
            self.sites_op = {}
        self.prefactor = prefactor
        assert isinstance(prefactor, (int, float, complex)), f"{type(prefactor)}"
        self.system = system
        if system is not None:
            self.size = len(system.sites)
            self.dimensions = {
                name: site["dimension"] for name, site in system.sites.items()
            }

    def __bool__(self):
        return bool(self.prefactor) and all(bool(factor) for factor in self.sites_op)

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
        result += "  (x)\n  ".join(
            f"({item[1].full()} <-  {item[0]})"
            for item in sorted(self.sites_op.items(), key=lambda x: x[0])
        )
        result += "\n   )"
        return result

    def acts_over(self):
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

    def flat(self):
        nfactors = len(self.sites_op)
        if nfactors == 0:
            return ScalarOperator(self.prefactor, self.system)
        if nfactors == 1:
            name, op_factor = list(self.sites_op.items())[0]
            return LocalOperator(name, self.prefactor * op_factor, self.system)
        return self

    def inv(self):
        sites_op = self.sites_op
        system = self.system
        prefactor = self.prefactor

        n_ops = len(sites_op)
        sites_op = {site: op_local.inv() for site, op_local in sites_op.items()}
        if n_ops == 1:
            site, op_local = next(iter(sites_op.items()))
            return LocalOperator(site, op_local / prefactor, system)
        return ProductOperator(sites_op, 1 / prefactor, system)

    @property
    def isherm(self) -> bool:
        if not all(loc_op.isherm for loc_op in self.sites_op.values()):
            return False
        return isinstance(self.prefactor, (int, float))

    @property
    def isdiagonal(self) -> bool:
        for factor_op in self.sites_op.values():
            if not is_diagonal_op(factor_op):
                return False
        return True

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
        result = OneBodyOperator(terms, system, False)
        result = result + ScalarOperator(np.log(self.prefactor), system)
        return result

    def partial_trace(self, sites: Union[list, SystemDescriptor]):
        full_system_sites = self.system.sites
        dimensions = self.dimensions
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = tuple(sites.sites.keys())
        else:
            subsystem = self.system.subsystem(sites)

        sites_out = tuple(s for s in full_system_sites if s not in sites)
        sites_op = self.sites_op
        prefactors = [
            sites_op[s].tr() if s in sites_op else dimensions[s] for s in sites_out
        ]
        sites_op = {s: o for s, o in sites_op.items() if s in sites}
        prefactor = self.prefactor
        for factor in prefactors:
            if factor == 0:
                return ScalarOperator(factor, subsystem)
            prefactor *= factor

        if len(sites_op) == 0:
            return ScalarOperator(prefactor, subsystem)
        return ProductOperator(sites_op, prefactor, subsystem)

    def simplify(self) -> Operator:
        """
        Simplifies a product operator
           - first, collect all the scalar factors and
             absorbe them in the prefactor.
           - If the prefactor vanishes, or all the factors are scalars,
             return a ScalarOperator.
           - If there is just one nontrivial factor, return a LocalOperator.
           - If no reduction is possible, return self.
        """
        # Remove multiples of the identity
        nontrivial_factors = {}
        prefactor = self.prefactor
        for site, op_factor in self.sites_op.items():
            if is_scalar_op(op_factor):
                prefactor *= op_factor[0, 0]
                assert isinstance(
                    prefactor, (int, float, complex)
                ), f"{type(prefactor)}:{prefactor}"
                if not prefactor:
                    return ScalarOperator(0, self.system)
            else:
                nontrivial_factors[site] = op_factor
        nops = len(nontrivial_factors)
        if nops == 0:
            return ScalarOperator(prefactor, self.system)
        if nops == 1:
            site, op_local = next(iter(nontrivial_factors.items()))
            return LocalOperator(site, self.prefactor * op_local, self.system)
        if nops != len(self.sites_op):
            return ProductOperator(nontrivial_factors, prefactor, self.system)
        return self

    def to_qutip(self):
        ops = self.sites_op
        system = self.system
        dimensions = system.dimensions
        if system:
            factors = {
                site: ops.get(site, None) if site in ops else qutip.qeye(dim)
                for site, dim in dimensions.items()
            }
            if len(factors) == 0:
                return ScalarOperator(
                    self.prefactor * reduce(lambda x, y: x * y, dimensions.keys()),
                    system,
                )
            result = self.prefactor * qutip.tensor(
                *(factors[site] for site in sorted(dimensions.keys()))
            )
            return result
        return self.prefactor * qutip.tensor(ops.values())

    def tr(self):
        result = self.partial_trace([])
        return result.prefactor

    def tidyup(self, atol=None):
        """remove tiny elements of the operator"""
        tidy_site_operators = {
            name: op_s.tidyup(atol) for name, op_s in self.sites_op.items()
        }
        return ProductOperator(tidy_site_operators, self.prefactor, self.system)


class ScalarOperator(ProductOperator):
    """A product operator that acts trivially on every subsystem"""

    def __init__(self, prefactor, system):
        assert system is not None
        super().__init__({}, prefactor, system)

    def __bool__(self):
        return bool(self.prefactor)

    def __neg__(self):
        return ScalarOperator(-self.prefactor, self.system)

    def __repr__(self):
        result = str(self.prefactor) + " * Identity "
        return result

    def acts_over(self):
        return set()

    def dag(self):
        if isinstance(self.prefactor, complex):
            return ScalarOperator(self.prefactor.conjugate(), self.system)
        return self

    @property
    def isherm(self):
        prefactor = self.prefactor
        return not (isinstance(prefactor, complex) and prefactor.imag != 0)

    @property
    def isdiagonal(self) -> bool:
        return True

    def logm(self):
        return ScalarOperator(np.log(self.prefactor), self.system)

    def simplify(self):
        """simplify a scalar operator"""
        return self


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
    return ScalarOperator(x_op.prefactor + y_op.prefactor, x_op.system or y_op.system)


@Operator.register_mul_handler(
    (
        ScalarOperator,
        ScalarOperator,
    )
)
def _(x_op: ScalarOperator, y_op: ScalarOperator):
    return ScalarOperator(x_op.prefactor * y_op.prefactor, x_op.system or y_op.system)


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
        sites_operators={
            site_x: x_op.operator,
            site_y: y_op.operator,
        },
        prefactor=1,
        system=system,
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
