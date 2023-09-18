"""
Different representations for operators
"""
from functools import reduce
from numbers import Number
from typing import List, Optional

import numpy as np
import qutip
from qutip import Qobj

from alpsqutip.alpsmodels import qutip_model_from_dims
from alpsqutip.geometry import GraphDescriptor
from alpsqutip.model import Operator, SystemDescriptor
from alpsqutip.settings import VERBOSITY_LEVEL


class QutipOperator(Operator):
    """Represents a Qutip operator associated with a system"""

    def __init__(
        self,
        qoperator: Qobj,
        system: Optional[SystemDescriptor] = None,
        names=None,
        prefactor=1,
    ):
        if system is None:
            dims = qoperator.dims[0]
            model = qutip_model_from_dims(dims)
            if names is None:
                names = {f"qutip_{i}": i for i in range(len(dims))}
            sitebasis = model.site_basis
            sites = {s: sitebasis[f"qutip_{i}"] for i, s in enumerate(names)}

            graph = GraphDescriptor(
                "disconnected graph",
                {s: {"type": f"qutip_{i}"} for i, s in enumerate(sites)},
                {},
            )
            system = SystemDescriptor(graph, model, sites=sites)
        if names is None:
            names = {s: i for i, s in enumerate(system.sites)}

        self.system = system
        self.operator = qoperator
        self.site_names = names
        self.prefactor = prefactor

    def __add__(self, operand):
        if isinstance(operand, Operator):
            return QutipOperator(
                self.prefactor * self.operator + operand.to_qutip(),
                self.system,
                names=self.site_names,
            )
        if isinstance(operand, (int, float, complex, Qobj)):
            return QutipOperator(
                self.prefactor * self.operator + operand,
                self.system,
                names=self.site_names,
            )
        raise ValueError()

    def __mul__(self, operand):
        if isinstance(operand, Operator):
            return QutipOperator(
                self.prefactor * self.operator * operand.to_qutip(),
                self.system,
                names=self.site_names,
            )
        if isinstance(operand, (int, float, complex, Qobj)):
            return QutipOperator(
                self.prefactor * self.operator * operand,
                self.system,
                names=self.site_names,
            )
        raise ValueError()

    def __neg__(self):
        return QutipOperator(-self.operator, self.system, names=self.site_names)

    def __rmul__(self, operand):
        if isinstance(operand, Operator):
            return QutipOperator(
                self.prefactor * operand.to_qutip() * self.operator,
                self.system,
                names=self.site_names,
            )
        if isinstance(operand, (int, float, complex, Qobj)):
            return QutipOperator(
                self.prefactor * operand * self.operator,
                self.system,
                names=self.site_names,
            )
        raise ValueError()

    def __pow__(self, exponent):
        operator = self.operator
        if exponent < 0:
            operator = operator.inv()
            exponent = -exponent

        return QutipOperator(
            operator**exponent,
            system=self.system,
            names=self.site_names,
            prefactor=1 / self.prefactor,
        )

    def dag(self):
        prefactor = self.prefactor
        operator = self.operator
        if isinstance(prefactor, complex):
            prefactor = prefactor.conj()
        else:
            if operator.isherm:
                return self
        return QutipOperator(operator.dag(), self.system, self.site_names, prefactor)

    def inv(self):
        """the inverse of the operator"""
        operator = self.operator
        return QutipOperator(
            operator.inv(),
            system=self.system,
            names=self.site_names,
            prefactor=1 / self.prefactor,
        )

    def partial_trace(self, sites: list):
        site_names = self.site_names
        sites = sorted(
            [s for s in self.site_names if s in sites], key=lambda s: site_names[s]
        )
        subsystem = self.system.subsystem(sites)
        assert len(sites) == len(subsystem.sites)
        site_indxs = [site_names[s] for s in sites]
        new_site_names = {s: i for i, s in enumerate(sites)}
        if site_indxs:
            op_ptrace = self.operator.ptrace(site_indxs)
        else:
            op_ptrace = self.operator.tr()

        return QutipOperator(
            op_ptrace, subsystem, names=new_site_names, prefactor=self.prefactor
        )

    def to_qutip(self):
        return self.operator * self.prefactor

    def tr(self):
        return self.operator.tr() * self.prefactor


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
        assert isinstance(local_operator, (int, float, complex, Qobj))
        self.site = site
        self.operator = local_operator
        self.system = system

    def __add__(self, operand):
        site = self.site
        if isinstance(operand, LocalOperator):
            system = self.system or operand.system
            if site == operand.site:
                return LocalOperator(site, self.operator + operand.operator, system)
            return OneBodyOperator(
                [
                    LocalOperator(site, self.operator, system),
                    LocalOperator(operand.site, operand.operator, system),
                ],
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
            import sys

            sys.exit()
        return result

    def __bool__(self):
        operator = self.operator
        if isinstance(operator, Qobj):
            return operator.data.count_nonzero() > 0
        return bool(self.operator)

    def __mul__(self, operand):
        site = self.site
        system = self.system or operand.system
        operator = self.operator
        if isinstance(operand, (int, float, complex, Qobj)):
            return LocalOperator(site, operator * operand, system)
        if isinstance(operand, LocalOperator):
            if site == operand.site:
                return LocalOperator(site, operator * operand.operator, system)

            return ProductOperator(
                {site: operator, operand.site: operand.operator}, 1.0, system=system
            )
        if isinstance(operand, ProductOperator):
            sites_op = operand.sites_op
            n_ops = len(sites_op)
            if n_ops == 0:
                return LocalOperator(site, operator * operand.prefactor, system)
            if n_ops == 1 and site in sites_op:
                return LocalOperator(site, operator * sites_op[site], system)
        return ProductOperator({site: operator}, system=system) * operand

    def __rmul__(self, operand):
        site = self.site
        system = self.system or operand.system
        operator = self.operator

        if isinstance(operand, (int, float, complex, Qobj)):
            return LocalOperator(site, operand * operator, system)
        if isinstance(operand, LocalOperator):
            site = self.site
            system = self.system or operand.system
            if site == operand.site:
                return LocalOperator(site, operand.operator * operator, system)

            return ProductOperator(
                {site: operator, operand.site: operand.operator}, 1.0, system=system
            )
        return operand * ProductOperator({site: operator}, system=system)

    def __neg__(self):
        return LocalOperator(self.site, -self.operator, self.system)

    def __pow__(self, exp):
        operator = self.operator
        if exp < 0 and hasattr(operator, "inv"):
            operator = operator.inv()
            exp = -exp

        return LocalOperator(self.site, operator**exp, self.system)

    def __repr__(self):
        return f"Local Operator on site {self.site}:\n {repr(self.operator)}"

    def dag(self):
        """
        Return the adjoint operator
        """
        operator = self.operator
        if isinstance(operator, (int, float)):
            return True
        if isinstance(operator, complex):
            return operator.imag == 0
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
            site, operator.inv() if hasattr(operator, "inv") else 1 / operator, system
        )

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
            d for s, d in dimensions.items() if s != site and s not in local_sites
        ]

        if len(prefactors) > 0:
            prefactor = reduce(lambda x, y: x * y, prefactors)
        else:
            prefactor = 1

        local_op = self.operator
        if hasattr(local_op, "tr") and self.site not in local_sites:
            local_op = local_op.tr()

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
            [operator if s == site else qutip.qeye(
                d) for s, d in dimensions.items()]
        )

    def tr(self):
        result = self.partial_trace([])
        return result.operator


class ProductOperator(Operator):
    """Product of operators acting over different sites"""

    def __init__(
        self,
        sites_operators: dict,
        prefactor=1.0,
        system: Optional[SystemDescriptor] = None,
    ):
        remove_numbers = False
        for site, op in sites_operators.items():
            if isinstance(op, (int, float, complex)):
                prefactor *= op
                remove_numbers = True

        if remove_numbers:
            sites_operators = {
                s: op
                for s, op in sites_operators.items()
                if not isinstance(op, (int, float, complex))
            }

        self.sites_op = sites_operators
        if any(op.data.count_nonzero() == 0 for op in sites_operators.values()):
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
        if self.prefactor == 0:
            return operand
        if isinstance(operand, (int, float, complex)):
            if operand == 0:
                return self
            operand = ProductOperator({}, operand, self.system)
        elif isinstance(operand, ProductOperator):
            if operand.prefactor == 0:
                return self
            if len(operand.sites_op) == 1 and len(self.sites_op) == 1:
                site = next(iter(self.sites_op))
                if site in operand.sites_op:
                    return LocalOperator(
                        site,
                        self.sites_op[site] * self.prefactor
                        + operand.sites_op[site] * operand.prefactor,
                        system=self.system,
                    )
                return OneBodyOperator([operand, self], self.system, True)
            new_terms = [operand, self]
        elif isinstance(operand, LocalOperator):
            site = next(iter(self.sites_op))
            if len(self.sites_op) == 1:
                if site == operand.site:
                    return LocalOperator(
                        site,
                        self.sites_op[site] *
                        self.prefactor + operand.operator,
                        system=self.system,
                    )
                return OneBodyOperator(
                    [
                        operand,
                        LocalOperator(
                            site,
                            self.sites_op[site] * self.prefactor,
                            system=self.system,
                        ),
                    ],
                    self.system,
                    True,
                )
            new_terms = [operand, self]
        elif isinstance(operand, SumOperator):
            new_terms = operand.terms + [self]
        else:
            new_terms = [self, operand]
        return SumOperator(new_terms)

    def __bool__(self):
        return bool(self.prefactor) or all(bool(factor) for factor in self.sites_op)

    def __mul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            new_prefactor = self.prefactor * operand
            if new_prefactor == 0.0:
                return ProductOperator({}, prefactor=new_prefactor, system=self.system)
            return ProductOperator(
                self.sites_op, prefactor=new_prefactor, system=self.system
            )
        if isinstance(operand, LocalOperator):
            sites_op = self.sites_op
            num_ops = len(sites_op)
            system = self.system
            site = operand.site

            if num_ops == 0:
                return self.prefactor * operand
            if num_ops == 1:
                if site in sites_op:
                    return LocalOperator(
                        site, self.prefactor *
                        sites_op[site] * operand.operator, system
                    )
                sites_op = sites_op.copy()
                sites_op[site] = operand.operator
            else:
                sites_op = sites_op.copy()
                if site in sites_op:
                    sites_op[site] = sites_op[site] * operand.operator
                else:
                    sites_op[site] = operand.operator
            return ProductOperator(sites_op, self.prefactor, self.system)

        if isinstance(operand, ProductOperator):
            new_sites_op = self.sites_op.copy()
            for pos, factor in operand.sites_op.items():
                if pos in new_sites_op:
                    new_sites_op[pos] = new_sites_op[pos] * factor
                else:
                    new_sites_op[pos] = factor
            return ProductOperator(
                new_sites_op,
                prefactor=self.prefactor * operand.prefactor,
                system=self.system,
            )
        if isinstance(operand, SumOperator):
            new_terms = [self * op_2 for op_2 in operand.terms]
            new_terms = [term for term in new_terms if term]
            return SumOperator(new_terms)

        raise NotImplementedError(
            "__mul__ not implemented for a Product operator and a", type(operand))

    def __neg__(self):
        return ProductOperator(self.sites_op, -self.prefactor, self.system)

    def __pow__(self, exp):
        return ProductOperator(
            {s: op**exp for s, op in self.sites_op.items()},
            self.prefactor**exp,
            self.system,
        )

    def __repr__(self):
        result = str(self.prefactor) + " * (\n  "
        result += "\n  ".join(str(item) for item in self.sites_op.items())
        result += " )"
        return result

    def __rmul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return self * operand
        return NotImplementedError

    def dag(self):
        """
        Return the adjoint operator
        """
        sites_op_dag = {key: op.dag() for key, op in self.sites_op.items()}
        prefactor = self.prefactor
        if isinstance(prefactor, complex):
            prefactor = prefactor.conj()
        return ProductOperator(sites_op_dag, prefactor, self.system)

    def expm(self):
        sites_op = self.sites_op
        n_ops = len(sites_op)
        if n_ops == 0:
            return ProductOperator({}, np.exp(self.prefactor), self.system)
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
        sites_op = {site: op_local.inv()
                    for site, op_local in sites_op.items()}
        if n_ops == 1:
            site, op_local = next(iter(sites_op.items()))
            return LocalOperator(site, op_local / prefactor, system)
        return ProductOperator(sites_op, 1 / prefactor, system)

    def partial_trace(self, sites: list):
        full_system_sites = self.system.sites
        dimensions = self.dimensions
        sites_in = tuple(s for s in sites if s in full_system_sites)
        sites_out = tuple(s for s in full_system_sites if s not in sites_in)
        subsystem = self.system.subsystem(sites_in)
        sites_op = self.sites_op
        prefactors = [
            sites_op[s].tr() if s in sites_op else dimensions[s] for s in sites_out
        ]
        sites_op = {s: o for s, o in sites_op.items() if s in sites_in}
        prefactor = self.prefactor
        for factor in prefactors:
            if factor == 0:
                return ProductOperator({}, prefactor=factor, system=subsystem)
            prefactor *= factor
        return ProductOperator(sites_op, prefactor, subsystem)

    def simplify(self):
        nops = len(self.sites_op)
        if nops == 0:
            return LocalOperator(
                next(iter(self.system.sites)), self.prefactor, self.system
            )
        if nops == 1:
            site, op_local = next(iter(self.sites_op.items()))
            return LocalOperator(site, self.prefactor * op_local, self.system)
        return self

    def to_qutip(self):
        if self.prefactor == 0 or len(self.system.dimensions) == 0:
            return self.prefactor
        ops = self.sites_op
        return self.prefactor * qutip.tensor(
            [
                ops.get(site, None) if site in ops else qutip.qeye(dim)
                for site, dim in self.system.dimensions.items()
            ]
        )

    def tr(self):
        result = self.partial_trace([])
        return result.prefactor


class SumOperator(Operator):
    """
    Represents a linear combination of operators
    """

    terms: List[Operator]
    system: Optional[SystemDescriptor]

    def __init__(self, terms_coeffs: list, system=None):
        self.terms = terms_coeffs
        assert all(isinstance(t, Operator) for t in terms_coeffs)

        if system is None and terms_coeffs:
            for term in terms_coeffs:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)
        self.system = system

    def __add__(self, operand):
        if isinstance(operand, (int, float, complex)):
            if operand == 0.0:
                return self
            operand = ProductOperator({}, operand, self.system)

        if isinstance(operand, ProductOperator):
            if operand.prefactor == 0:
                return self
            new_terms = self.terms + [operand]
        elif isinstance(operand, LocalOperator):
            new_terms = self.terms + [operand]
        elif isinstance(operand, SumOperator):
            if len(operand.terms) == len(self.terms) == 1:
                return self.terms[0] + operand.terms[0]
            new_terms = self.terms + operand.terms
        else:
            raise ValueError(type(self), type(operand))

        new_terms = [t for t in new_terms if t]
        return SumOperator(new_terms).simplify()

    def __bool__(self):
        if len(self.terms) == 0:
            return False

        if any(bool(t) for t in self.terms):
            return True
        return False

    def __pow__(self, exp):
        if isinstance(exp, int):
            if exp == 0:
                return 1
            if exp == 1:
                return self
            if exp > 1:
                exp -= 1
                return self * (self**exp)
            raise TypeError("SumOperator does not support negative powers")
        raise TypeError(
            (
                f"unsupported operand type(s) for ** or pow(): "
                f"'SumOperator' and '{type(exp).__name__}'"
            )
        )

    def __mul__(self, operand):
        if isinstance(operand, QutipOperator):
            return self.to_qutip_operator() * operand
        if isinstance(operand, (int, float, complex)):
            if operand == 0:
                return operand
            new_terms = [
                operand * operand1 for operand1 in self.terms if operand1]
        elif isinstance(operand, (ProductOperator, LocalOperator)):
            if operand.prefactor:
                new_terms = [operand1 *
                             operand for operand1 in self.terms if operand1]
            else:
                new_terms = []
        elif isinstance(operand, SumOperator):
            new_terms = [
                op_1 * op_2 for op_1 in self.terms for op_2 in operand.terms]
        else:
            raise TypeError(type(operand))

        new_terms = [t for t in new_terms if t]
        if len(new_terms) == 0:
            return 0.0
        if len(new_terms) == 1:
            return new_terms[0]
        return SumOperator(new_terms)

    def __neg__(self):
        return SumOperator([-t for t in self.terms])

    def __repr__(self):
        return "(\n" + "\n  +".join(repr(t) for t in self.terms) + "\n)"

    def __rmul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return self * operand
        return NotImplementedError

    def dag(self):
        """return the adjoint operator"""
        return SumOperator([t.dag() for t in self.terms])

    def partial_trace(self, sites: list):
        return sum(term.prefactor * term.partial_trace(sites) for term in self.terms)

    def simplify(self):
        system = self.system
        general_terms = []
        # First, shallow the list of terms:
        for term in (t.simplify() for t in self.terms):
            if isinstance(term, SumOperator):
                general_terms.extend(term.terms)
            else:
                general_terms.append(term)

        terms = general_terms
        # Now, collect and sum LocalOperator and QutipOperator terms
        general_terms = []
        site_terms = {}
        qutip_terms = []
        for term in terms:
            if isinstance(term, LocalOperator):
                site_terms.setdefault(term.site, []).append(term.operator)
                continue
            if isinstance(term, QutipOperator):
                qutip_terms.append(term)
            else:
                general_terms.append(term)

        loc_ops_lst = [
            LocalOperator(site, sum(l_ops), system)
            for site, l_ops in site_terms.items()
        ]

        qutip_term = sum(qutip_terms)
        qutip_terms = qutip_term if qutip_terms else []
        terms = general_terms + loc_ops_lst + qutip_terms
        return SumOperator(terms, system)

    def to_qutip(self):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ProductOperator({}, 0, self.system).to_qutip()
        return sum(t.to_qutip() for t in self.terms)

    def tr(self):
        return sum(t.tr() for t in self.terms)


NBodyOperator = SumOperator


class OneBodyOperator(SumOperator):
    """A linear combination of local operators"""

    def __init__(self, terms, system=None, check_and_convert=True):
        """
        if check_and_convert is True,
        """
        if check_and_convert:
            terms_by_site = {}

            for term in terms:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

            for term in terms:
                if isinstance(term, LocalOperator):
                    assert isinstance(
                        term.operator, (int, float, complex, Qobj))
                    terms_by_site.setdefault(
                        term.site, []).append(term.operator)
                    continue
                if isinstance(term, ProductOperator):
                    n_factors = len(term.sites_op)
                    if n_factors > 1:
                        raise ValueError("All the terms must be local", term)
                    elif n_factors == 0:
                        if term.system:
                            site = next(iter(term.system.sites))
                            terms_by_site.setdefault(
                                site, []).append(term.prefactor)
                            continue

                        raise ValueError(
                            "A trivial product operator should have a system"
                        )
                    else:
                        site, op_l = next(iter(term.sites_op.items()))
                        assert isinstance(op_l, (Number, Qobj))
                        terms_by_site.setdefault(site, []).append(
                            term.prefactor * op_l)
                        continue
                if isinstance(term, SumOperator):
                    for t_i in OneBodyOperator(
                        term.terms, term.system, check_and_convert=True
                    ).terms:
                        assert isinstance(t_i, (Number, LocalOperator))
                        terms_by_site.setdefault(
                            t_i.site, []).append(t_i.operator)
                    continue
                raise ValueError("Invalid term type", type(term))

            super().__init__(
                [
                    LocalOperator(s, sum(ops), system)
                    for s, ops in terms_by_site.items()
                ],
                system,
            )
        else:
            super().__init__(terms, system)

    def __add__(self, operand):
        system = self.system or operand.system
        if isinstance(operand, OneBodyOperator):
            my_terms = [term for term in self.terms if term]
            other_terms = [term for term in operand.terms if term]
            return OneBodyOperator(my_terms + other_terms, system)
        if isinstance(operand, (int, float, complex)):
            if operand:
                return OneBodyOperator(
                    self.terms + [ProductOperator({}, operand, system)], system
                )
            return self
        if isinstance(operand, LocalOperator):
            return OneBodyOperator(self.terms + [operand], system)
        return super().__mul__(operand)

    def __mul__(self, operand):
        system = self.system or operand.system
        if isinstance(operand, OneBodyOperator):
            my_terms = [term for term in self.terms if term]
            other_terms = [term for term in operand.terms if term]
            return SumOperator(
                [
                    my_term * other_term
                    for my_term in my_terms
                    for other_term in other_terms
                ],
                system,
            )
        if isinstance(operand, (int, float, complex)):
            if operand:
                return OneBodyOperator(
                    [term * operand for term in self.terms if term], system
                )
            return ProductOperator({}, 0.0, system)
        return super().__mul__(operand)

    def __rmul__(self, operand):
        system = self.system or operand.system
        if isinstance(operand, OneBodyOperator):
            my_terms = self.terms
            other_terms = operand.terms
            return SumOperator(
                [
                    other_term * my_term
                    for my_term in my_terms
                    for other_term in other_terms
                ],
                system,
            )
        if isinstance(operand, (int, float, complex)):
            if operand:
                return OneBodyOperator(
                    [operand * term for term in self.terms if term], system
                )
            return ProductOperator({}, 0.0, system)
        return super().__mul__(operand)

    def __neg__(self):
        return OneBodyOperator([-term for term in self.terms], self.system)

    def dag(self):
        return OneBodyOperator(
            [term.dag() for term in self.terms], self.system, check_and_convert=False
        )

    def expm(self):
        sites_op = {}
        for term in self.terms:
            if not bool(term):
                continue
            operator = term.operator
            if hasattr(operator, "expm"):
                sites_op[term.site] = operator.expm()
            else:
                sites_op[term.site] = np.exp(operator)
        return ProductOperator(sites_op, system=self.system)
