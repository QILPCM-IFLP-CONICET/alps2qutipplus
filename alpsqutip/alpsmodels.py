"""
Function to read xml alps libraries to produce model descriptors.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Optional

import qutip

from alpsqutip.utils import eval_expr, find_ref


def list_operators_in_alps_xml(filename="models.xml"):
    """
    List the models available in the library
    """
    result = set()
    xmltree = ET.parse(filename)
    models = xmltree.getroot()

    for h in models.findall("./HAMILTONIAN"):
        name = h.attrib.get("name", None)
        if name:
            result.add(name)

    for h in models.findall("./BASIS"):
        name = h.attrib.get("name", None)
        if name:
            result.add(name)

    return tuple(result)


def build_local_basis_from_qn_descriptors(
    qns: dict, parms: Optional[dict] = None
) -> dict:
    """
    From a quantum number descriptor and a set of parameters,
    build a dictionary with keys `qns` and `basis`.
    `basis` is a list of tuples containing the values of the quantum numbers.
    `qns` is a dict that maps the name of the quantum numbers to the position
    in the tuples in `basis`.
    """
    local_basis = [{}]
    parms = parms.copy() if parms is not None else {}
    while True:
        # The new basis to be built over the previous basis
        new_basis = []
        for state in local_basis:
            parms.update(state)
            # First, look for the next qt with defined limits given
            # the qn in the state
            new_qn = None
            qn = None
            for qn in qns:
                if qn in state:
                    continue
                qn_dict = qns[qn]
                lower = eval_expr(qn_dict["min"], parms)
                upper = eval_expr(qn_dict["max"], parms)
                # fermionic = qn_dict["fermionic"]
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                    new_qn = (qn, lower, int(upper - lower + 1))
                    break
            # If no new qn was found, then return the basis as it is.
            if new_qn is None:
                break
            # Otherwise, add the qn to the state
            for offset in range(new_qn[2]):
                new_state = state.copy()
                new_state.update({new_qn[0]: new_qn[1] + offset})
                new_basis.append(new_state)
        if new_qn is None:
            break
        local_basis = new_basis

    if len(local_basis) == 1 and len(local_basis[0]) == 0:
        logging.warning("empty basis!")
        return None
    qn_indx = {qn: i for i, qn in enumerate(local_basis[0].keys())}
    basis_vectors = [tuple(state[qn] for qn in qn_indx) for state in local_basis]
    return {"qns": qn_indx, "basis": basis_vectors}


def model_from_alps_xml(filename="lattices.xml", name="spin", parms=None):
    """
    Load from `filename` xml library a model of name
    `name`, using `parms` as parameters.
    """

    xmltree = ET.parse(filename)
    models = xmltree.getroot()
    if parms is None:
        parms = {}

    def process_basis(node, parms):
        parms = process_parms(node, parms)
        constraints = {}
        sitebasis = {}
        for constraint_node in node.findall("./CONSTRAINT"):
            constr_attrib = constraint_node.attrib
            qn_name = constr_attrib["quantumnumber"]
            constraints[qn_name] = constr_attrib["value"]
        for sitebasis_node in node.findall("./SITEBASIS"):
            type_name = sitebasis_node.attrib.get("type", "0")
            sitebasis[type_name] = process_sitebasis(
                find_ref(sitebasis_node, models), parms
            )

        bond_operators_descr = process_bond_operators({}, parms)
        global_operators_descr = process_global_operators({}, parms)

        return ModelDescriptor(
            site_basis=sitebasis,
            constraints=constraints,
            bond_op_descr=bond_operators_descr,
            global_op_descr=global_operators_descr,
            parms=parms,
            name=name,
        )

    def process_site_operators(operators: dict, parms: dict):
        """
        Enlarge the  operators` dict with the `SITEOPERATOR` nodes.
        """
        while True:
            found = False
            for node in models.findall("./SITEOPERATOR"):
                siteop_descr = dict(node.items())
                name = siteop_descr.get("name", None)
                if name is None or name in operators:
                    continue
                site = siteop_descr["site"]
                parms_and_ops = parms.copy()
                parms_and_ops["x"] = 0
                parms_and_ops.update(
                    {f"{op}_qutip": qutip_op for op, qutip_op in operators.items()}
                )
                expr = "".join(line.strip() for line in node.itertext())
                expr = expr.replace(f"({site})", "_qutip")
                new_op = eval_expr(expr, parms_and_ops)
                if isinstance(new_op, qutip.Qobj):
                    operators[name] = new_op
                    found = True
            if not found:
                break
        return operators

    def process_bond_operators(operators: dict, parms: dict):
        for node in models.findall("./BONDOPERATOR"):
            descriptor = node.attrib
            name = descriptor["name"]
            src = descriptor["source"]
            dst = descriptor["target"]
            expr = "".join(line.strip() for line in node.itertext())
            expr = expr.replace(f"({src})", "@src")
            expr = expr.replace(f"({dst})", "@dst")
            expr = expr.replace(f"({src},{dst})", "@src_dst")
            expr = expr.replace(f"({dst},{src})", "@dst_src")
            operators[name] = expr
        return operators

    def process_global_operators(operators: dict, parms: dict):
        for node in models.findall("./GLOBALOPERATOR"):
            descriptor = node.attrib
            name = descriptor["name"]
            site_terms = []
            bond_terms = []
            for op in node.findall("./SITETERM"):
                site_terms.append(process_site_term(find_ref(op, models), parms))

            for op in node.findall("./BONDTERM"):
                bond_terms.append(process_bondterm(find_ref(op, models), parms))

            operators[name] = {
                "site terms": site_terms,
                "bond terms": bond_terms,
            }
        return operators

    def process_site_term(node, parms):
        parms_overwrite = process_parms(node, parms)
        parms = {
            key: val
            for key, val in parms_overwrite.items()
            if val != parms.get(key, None)
        }

        descriptor = node.attrib
        site = descriptor.get("site", "i")
        node_type = descriptor.get("type", "0")
        expr = "".join(line.strip() for line in node.itertext())
        expr = expr.replace(f"({site})", "_local")

        return {"expr": expr, "type": node_type, "parms": parms}

    def process_bondterm(node, parms):
        parms_overwrite = process_parms(node, parms)
        parms = {
            key: val
            for key, val in parms_overwrite.items()
            if val != parms.get(key, None)
        }

        descriptor = node.attrib
        src = descriptor["source"]
        dst = descriptor["target"]
        bond_type = descriptor.get("type", "0")
        expr = "".join(line.strip() for line in node.itertext())
        expr = expr.replace(f"({src})", "@src")
        expr = expr.replace(f"({dst})", "@dst")
        expr = expr.replace(f"({src},{dst})", "@src_dst")
        expr = expr.replace(f"({dst},{src})", "@dst_src")

        return {"expr": expr, "type": bond_type, "parms": parms}

    def process_sitebasis(node, parms) -> dict:
        basis_name = node.attrib.get("name", "")
        parms = process_parms(node, parms)
        quantumnumbers = {}
        operators_descr = {}
        operators = {}
        for qnumbers_node in node.findall("./QUANTUMNUMBER"):
            qn_attrib = qnumbers_node.attrib
            quantumnumbers[qn_attrib["name"]] = {
                "min": eval_expr(qn_attrib["min"], parms),
                "max": eval_expr(qn_attrib["max"], parms),
                "fermionic": qn_attrib.get("type", "") == "fermionic",
            }

        for op_node in node.findall("./OPERATOR"):
            op_attrib = op_node.attrib
            name = op_attrib.pop("name")
            changing = {}
            for op_change in op_node.findall("./CHANGE"):
                chng_attr = op_change.attrib
                changing[chng_attr["quantumnumber"]] = chng_attr["change"]
            op_attrib["changing"] = changing
            operators_descr[name] = op_attrib
            if len(changing) == 0:
                matrix_element = operators_descr[name].get("matrixelement", None)
                if matrix_element in quantumnumbers:
                    quantumnumbers[matrix_element]["operator"] = name

        for qn_name, qn_attr in quantumnumbers.items():
            if "operator" not in qn_attr:
                name = qn_name.lower()
                i = 0
                while name in operators_descr:
                    name = name + str(i)
                    i += 1
                operators_descr[name] = {
                    "changing": {},
                    "matrixelement": qn_name,
                }
                qn_attr["operator"] = name

        # Using the quantum number descriptor, build the local basis
        qns_and_basis = build_local_basis_from_qn_descriptors(quantumnumbers, parms)
        qn_pos, local_basis = qns_and_basis["qns"], qns_and_basis["basis"]
        local_basis_pos = {state: i for i, state in enumerate(local_basis)}
        # And the basic local operators.
        parms_qn = parms.copy()
        dim = len(local_basis)

        operators["identity"] = qutip.qeye(dim)
        for name, od in operators_descr.items():
            terms = []
            for qns, src in local_basis_pos.items():
                parms_qn.update({qn: qns[qn_pos[qn]] for qn in qn_pos})
                dest_qn = list(qns)
                fermionic = False
                for qn, offset in od["changing"].items():
                    offset = eval_expr(offset, parms)
                    if quantumnumbers[qn]["fermionic"] and offset % 2 != 0:
                        fermionic = not fermionic
                    dest_qn[qn_pos[qn]] += offset
                coeff = eval_expr(od["matrixelement"], parms_qn)
                dst = local_basis_pos.get(tuple(dest_qn), None)
                if dst is not None:
                    terms.append((dst, src, coeff, fermionic))

            if any(t[-1] for t in terms) and not all(t[-1] for t in terms):
                logging.warning("wrong fermionic parity", name, ":", terms)
                continue
            operators[name] = sum(
                coeff * qutip.projection(dim, src, dst)
                for src, dst, coeff, fermionic in terms
            )
            operators[name].fermionic = any(t[-1] for t in terms)

        # This loop loads all the site operators compatible with the site definition.
        operators = process_site_operators(operators, parms)

        return {
            "name": basis_name,
            "qn": quantumnumbers,
            "dimension": dim,
            "operators": operators,
            "parms": parms,
            "localstates": local_basis,
        }

    def process_hamiltonian(ham, parms):
        parms = process_parms(ham, parms)
        site_terms = []
        bond_terms = []

        basis = None
        for basis_entry in ham.findall("./BASIS"):
            basis = process_basis(find_ref(basis_entry, models), parms)
            break

        assert basis is not None

        for op_site in ham.findall("./SITETERM"):
            site_terms.append(process_site_term(find_ref(op_site, models), parms))

        for op_bond in ham.findall("./BONDTERM"):
            bond_terms.append(process_bondterm(find_ref(op_bond, models), parms))

        basis.global_ops["Hamiltonian"] = {
            "site terms": site_terms,
            "bond terms": bond_terms,
        }
        basis.parms.update(parms)
        return basis

    def process_parms(node, parms):
        """Process the <PARAMETER>s nodes"""
        default_parms = {}
        for parameter in node.findall("./PARAMETER"):
            key_vals = parameter.attrib
            default_parms[key_vals["name"]] = key_vals["default"]
        default_parms.update(parms)
        return default_parms

    # Try to find a Hamiltonian
    for ham in models.findall("./HAMILTONIAN"):
        if ("name", name) in ham.items():
            return process_hamiltonian(ham, parms)

    # Otherwise, try with a basis
    for basis in models.findall("./BASIS"):
        if ("name", name) in basis.items():
            return process_basis(basis, parms)


class ModelDescriptor:
    """
    Describes a model, including the description of the
    local systems, and rules to build site and global operators.
    """

    def __init__(
        self,
        site_basis: dict,
        constraints: Optional[dict] = None,
        bond_op_descr: Optional[dict] = None,
        global_op_descr: Optional[dict] = None,
        parms: Optional[dict] = None,
        name: Optional[str] = "",
    ):
        self.site_basis = site_basis
        self.constraints = constraints or {}
        self.bond_ops = bond_op_descr or {}
        self.global_ops = global_op_descr or {}
        self.parms = parms or {}
        self.name = name

    def __repr__(self):
        return repr(self.__dict__)


def qutip_model_from_dims(dims, local_ops=None, global_ops=None, model_name="qutip"):
    """
    Produce a basic model descriptor from the dimensions
    """
    site_basis = {}
    for i, d in enumerate(dims):
        name = f"qutip_{i}"
        site_basis[name] = {
            "name": name,
            "qn": {"n"},
            "dimension": d,
            "operators": {
                "identity": qutip.qeye(d),
                "n": qutip.num(d),
                "GS": qutip.projection(d, 0, 0),
                "raise": qutip.create(d),
                "lower": qutip.destroy(d),
            },
            "parms": {},
            "localstates": [{"n": i} for i in range(d)],
        }
    return ModelDescriptor(site_basis, name=model_name)
