"""
Define SystemDescriptors and different kind of operators
"""
from typing import Callable, Dict, Optional, Tuple
from numbers import Number

import numpy as np
import qutip

from alpsqutip.geometry import GraphDescriptor
from alpsqutip.settings import VERBOSITY_LEVEL
from alpsqutip.utils import eval_expr


class SystemDescriptor:
    """
    System Descriptor class.
    """

    def __init__(
        self,
        graph: GraphDescriptor,
        model: dict,
        parms: Optional[dict] = None,
        sites=None,
    ):
        if parms is None:
            parms = {}
        if model:
            model_parms = model.parms.copy()
            model_parms.update(parms)
            parms = model_parms

        self.spec = {"graph": graph, "model": model, "parms": parms}

        site_basis = model.site_basis
        if sites:
            self.sites = sites
        else:
            self.sites = {
                node: site_basis[attr["type"]]
                for node, attr in graph.nodes.items()
            }

        self.dimensions = {
            name: site["dimension"] for name, site in self.sites.items()
        }
        self.operators = {
            "site_operators": {},
            "bond_operators": {},
            "global_operators": {},
        }
        self._load_site_operators()
        self._load_global_ops()

    def __repr__(self):
        result = (
            "graph:"
            + repr(self.spec["graph"])
            + "\n"
            + "sites:"
            + repr(self.sites.keys())
            + "\n"
            + "dimensions:"
            + repr(self.dimensions)
        )
        return result

    def subsystem(self, sites: list):
        """
        Build a subsystem including the sites listed
        in sites
        """
        parms = self.spec["parms"].copy()
        model = self.spec["model"]
        graph = self.spec["graph"].subgraph(tuple(sites))
        return SystemDescriptor(graph, model, parms)

    def _load_site_operators(self):
        for site_name, site in self.sites.items():
            for op_name in site["operators"]:
                op_site = f"{op_name}@{site_name}"
                self.site_operator(op_site)

    def _load_global_ops(self):
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators import (
            LocalOperator,
            OneBodyOperator,
        )

        # First, load conserved quantum numbers:
        for constraint_qn in self.spec["model"].constraints:
            global_qn_terms = []
            for site, site_basis in self.sites.items():
                local_qn = site_basis["qn"].get(constraint_qn, None)
                if local_qn is None:
                    continue
                op_name = local_qn["operator"]
                operator = site_basis["operators"][op_name]

                global_qn_terms.append(LocalOperator(site, operator, self))

            global_qn = OneBodyOperator(tuple(global_qn_terms), self, True)

            if bool(global_qn):
                self.operators["global_operators"][constraint_qn] = global_qn

        names = list(self.spec["model"].global_ops)
        for gop in names:
            self.global_operator(gop)

    def union(self, system):
        """Return a SystemDescritor containing system and self"""
        if system is None or system is self:
            return self
        if all(site in self.sites for site in system.sites):
            return self
        if all(site in system.sites for site in self.sites):
            return system
        raise NotImplementedError(
            "Union of disjoint systems are not implemented."
        )

    def site_operator(self, name: str, site: str = "") -> "Operator":
        """
        Return a global operator representing an operator `name`
        acting over the site `site`. By default, the name is assumed
        to specify both the name and site in the form `"name@site"`.
        """
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators import LocalOperator

        if site != "":
            op_name = name
            name = f"{name}@{site}"
        else:
            op_name, site = name.split("@")

        site_op = (
            self.operators["site_operators"].get(site, {}).get(name, None)
        )
        if site_op is not None:
            return site_op

        local_op = self.sites[site]["operators"].get(op_name, None)
        if local_op is None:
            return None
        result_op = LocalOperator(site, local_op, system=self)
        self.operators["site_operators"].setdefault(site, {})
        self.operators["site_operators"][site][op_name] = result_op
        return result_op

    def bond_operator(
        self, name: str, src: str, dst: str, skip=None
    ) -> "Operator":
        """Bond operator by name and sites"""

        result_op = self.operators["global_operators"].get(
            (
                name,
                src,
                dst,
            ),
            None,
        )
        if result_op is not None:
            return result_op
        # Try to build the bond operator from the descriptors.
        bond_op_descriptors = self.spec["model"].bond_ops
        bond_op_descriptor = bond_op_descriptors.get(name, None)
        if bond_op_descriptor is None:
            return None

        bond_dependencies = [
            bop
            for bop in bond_op_descriptors
            if bond_op_descriptor.find(bop + "@") >= 0
        ]
        bond_op_descriptor = bond_op_descriptor.replace("@", "__")
        src_operators = self.sites[src]["operators"]
        dst_operators = self.sites[dst]["operators"]

        # Load site operators on src and dst
        parms_and_ops = {
            f"{name_src}__src": self.site_operator(name_src, src)
            for name_src in src_operators
        }
        parms_and_ops.update(
            {
                f"{name_dst}__dst": self.site_operator(name_dst, dst)
                for name_dst in dst_operators
            }
        )
        self_parms = self.spec["parms"]
        if self_parms:
            parms_and_ops.update(self_parms)

        # Try to evaluate
        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.operators["bond_operators"][
                (
                    name,
                    src,
                    dst,
                )
            ] = result
            return result
        # Now, try to include existent bond operators
        parms_and_ops.update(
            {
                f"{tup_op[0]}__src_dst": op
                for tup_op, op in self.operators["global_operators"].items()
                if tup_op[0] == src and tup_op[1] == dst
            }
        )
        parms_and_ops.update(
            {
                f"{tup_op[0]}__dst_src": op
                for tup_op, op in self.operators["global_operators"].items()
                if tup_op[0] == dst and tup_op[1] == src
            }
        )

        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.operators["bond_operators"][tuple(name, src, dst)] = result
            return result

        # Finally, try to load other operators
        if skip is None:
            skip = [name]
        else:
            skip.append(name)
        for bop in bond_dependencies:
            # Skip this name
            if bop in skip:
                continue

            # src->dst
            new_bond_op = self.bond_operator(bop, src, dst, skip)
            if new_bond_op is None:
                continue

            parms_and_ops[f"{bop}__src_dst"] = new_bond_op

            new_bond_op = self.bond_operator(bop, dst, src, skip)
            if new_bond_op is None:
                continue

            parms_and_ops[f"{bop}__dst_src"] = new_bond_op

            result = eval_expr(bond_op_descriptor, parms_and_ops)
            if result is not None and not isinstance(result, str):
                self.operators["bond_operators"][
                    (
                        name,
                        src,
                        dst,
                    )
                ] = result
                return result

        # If this fails after exploring all the operators, then it means that
        # the operator is not in the basis.
        # if skip[-1]==name:
        #    self.bond_operators[(name, src, dst,)] = None
        return None

    def site_term_from_descriptor(self, term_spec, graph, parms):
        """Build a site term from a site term specification"""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators import OneBodyOperator

        expr = term_spec["expr"]
        site_type = term_spec.get("type", None)
        term_ops = []
        t_parm = {}
        t_parm.update(term_spec.get("parms", {}))
        if parms:
            t_parm.update(parms)
        for node_name, node in graph.nodes.items():
            node_type = node.get("type", None)
            if site_type is not None and site_type != node_type:
                continue
            s_expr = expr.replace("#", node_type).replace("@", "__")
            s_parm = {
                key.replace("#", node_type): val for key, val in t_parm.items()
            }
            s_parm.update(
                {
                    f"{name_op}_local": local_op
                    for name_op, local_op in self.operators["site_operators"][
                        node_name
                    ].items()
                }
            )
            term_op = eval_expr(s_expr, s_parm)
            if term_op is None or isinstance(term_op, str):
                raise ValueError(f"<<{s_expr}>> could not be evaluated.")
            term_ops.append(term_op)

        return OneBodyOperator(tuple(term_ops), self)

    def bond_term_from_descriptor(self, term_spec, graph, model, parms):
        """Build a bond term from a bond term speficication"""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators import SumOperator

        def process_edge(e_expr, bond, model, t_parm):
            edge_type, src, dst = bond
            e_parms = {
                key.replace("#", f"{edge_type}"): val
                for key, val in t_parm.items()
            }
            for op_idx in ([src, "src"], [dst, "dst"]):
                e_parms.update(
                    {
                        f"{key}__{op_idx[1]}": val
                        for key, val in self.operators["site_operators"][
                            op_idx[0]
                        ].items()
                    }
                )

            # Try to compute using only site terms

            term_op = eval_expr(e_expr, e_parms)
            if not isinstance(term_op, str):
                return term_op

            # Try now adding the bond operators
            for name_bop in model.bond_ops:
                self.bond_operator(name_bop, src, dst)
                self.bond_operator(name_bop, dst, src)

            for src_idx, dst_idx in ((1, 2), (2, 1)):
                e_parms.update(
                    {
                        f"{key[0]}__src_dst": val
                        for key, val in self.operators[
                            "bond_operators"
                        ].items()
                        if key[src_idx] == src and key[dst_idx] == dst
                    }
                )

            return eval_expr(e_expr, e_parms)

        expr = term_spec["expr"]
        term_type = term_spec.get("type", None)
        t_parm = {}
        t_parm.update(term_spec.get("parms", {}))
        if parms:
            t_parm.update(parms)
        result_terms = []
        for edge_type, edges in graph.edges.items():
            if term_type is not None and term_type != edge_type:
                continue
            e_expr = expr.replace("#", edge_type).replace("@", "__")
            for src, dst in edges:
                term_op = process_edge(
                    e_expr, (edge_type, src, dst), model, t_parm
                )
                if isinstance(term_op, str):
                    raise ValueError(
                        f"   Bond term <<{term_op}>> could not be evaluated."
                    )

                result_terms.append(term_op)

        return SumOperator(tuple(result_terms), self, True)

    def global_operator(self, name):
        """Return a global operator by its name"""
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators import OneBodyOperator, SumOperator

        result = self.operators["global_operators"].get(name, None)
        if result is not None:
            return result
        # Build the global_operator from the descriptor
        op_descr = self.spec["model"].global_ops.get(name, None)
        if op_descr is None:
            return None

        graph = self.spec["graph"]
        parms = self.spec["parms"]
        model = self.spec["model"]


        # Process site terms
        try:
            site_terms = (
                self.site_term_from_descriptor(term_spec, graph, parms)
                for term_spec in op_descr["site terms"]
            )
            site_terms = tuple(term for term in site_terms if term)
        except ValueError as exc:
            if VERBOSITY_LEVEL > 2:
                print(*exc.args, f"Aborting evaluation of {name}.")
            model.global_ops.pop(name)
            return None

        # Process bond terms
        try:
            bond_terms = tuple(
                self.bond_term_from_descriptor(term_spec, graph, model, parms)
                for term_spec in op_descr["bond terms"]
            )
            bond_terms = tuple(term for term in bond_terms if term)

        except ValueError as exc:
            if VERBOSITY_LEVEL > 2:
                print(*exc.args, f"Aborting evaluation of {name}.")
            model.global_ops.pop(name)
            return None

        if bond_terms:
            result = SumOperator(site_terms + bond_terms, self, True)
        else:
            result = OneBodyOperator(site_terms, self, True)
        self.operators["global_operators"][name] = result
        return result


class Operator:
    """Base class for operators"""

    system: SystemDescriptor
    prefactor: float = 1.0

    # def __add__(self, term):
    #    # pylint: disable=import-outside-toplevel
    #    from alpsqutip.operators import SumOperator
    #
    #    return SumOperator([self, term], self.system)

    __mul__dispatch__: Dict[Tuple, Callable] = {}

    def __mul__(self, factor):
        # Use multiple dispatch to determine how to multiply
        func = self.__mul__dispatch__.get((type(self), type(factor)), None)

        if func is not None:
            return func(self, factor)

        for key, func in self.__mul__dispatch__.items():
            lhf, rhf = key
            if isinstance(self, lhf) and isinstance(factor, rhf):
                return func(self, factor)

        return self.to_qutip_operator() * factor

    def __neg__(self):
        return -(self.to_qutip_operator())

    def __sub__(self, operand):
        if operand is None:
            raise ValueError("None can not be an operand")
        neg_op = -operand
        return self + neg_op

    def __radd__(self, operand):
        if operand is None:
            raise ValueError("None can not be an operand")
        return self + operand

    def __rmul__(self, factor):
        # Use __mul__dispatch__ to determine how to evaluate the product
        if isinstance(factor, Number):
            return self * factor

        func = self.__mul__dispatch__.get((type(factor), type(self)), None)

        if func is not None:
            return func(factor, self)

        for key, func in self.__mul__dispatch__.items():
            lhf, rhf = key
            if isinstance(factor, lhf) and isinstance(self, rhf):
                return func(factor, self)

        return factor * self.to_qutip_operator()

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
        raise ValueError(
            "Division of an operator by ", type(operand), " not defined."
        )

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

    def act_over(self) -> Optional[set]:
        """
        Return the list of sites over which the operator acts nontrivially.
        If this cannot be determined, return None.
        """
        return None

    def dag(self):
        """Adjoint operator of quantum object"""
        return self.to_qutip_operator().dag()

    @property
    def isherm(self) -> bool:
        """Check if the operator is hermitician"""
        return self.to_qutip().isherm

    def expm(self):
        """Produce a Qutip representation of the operator"""

        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from alpsqutip.operators.qutip import QutipOperator

        op_qutip = self.to_qutip()
        max_eval = op_qutip.eigenenergies(sort="high", sparse=True, eigvals=3)[
            0
        ]
        op_qutip = (op_qutip - max_eval).expm()
        return QutipOperator(op_qutip, self.system, prefactor=np.exp(max_eval))

    def inv(self):
        """the inverse of the operator"""
        return self.to_qutip_operator().inv()

    def partial_trace(self, sites: list):
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

        return QutipOperator(self.to_qutip(), self.system)

    # pylint: disable=invalid-name
    def tr(self):
        """The trace of the operator"""
        return self.partial_trace([]).prefactor


def build_spin_chain(length: int = 2, field=0.0):
    """Build a spin chain of length `l`"""
    # pylint: disable=import-outside-toplevel
    from alpsqutip.alpsmodels import model_from_alps_xml

    # pylint: disable=import-outside-toplevel
    from alpsqutip.geometry import graph_from_alps_xml

    # pylint: disable=import-outside-toplevel
    from alpsqutip.settings import LATTICE_LIB_FILE, MODEL_LIB_FILE

    return SystemDescriptor(
        model=model_from_alps_xml(MODEL_LIB_FILE, "spin"),
        graph=graph_from_alps_xml(
            LATTICE_LIB_FILE, "chain lattice", parms={"L": length, "a": 1}
        ),
        parms={"h": field, "J": 1, "Jz0": 1, "Jxy0": 1},
    )
