"""
Utility functions to import and process
ALPS specification files.
"""

import logging

import numpy as np
import qutip
from numpy.random import rand

default_parms = {
    "pi": 3.1415926,
    "e": 2.71828183,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "rand": rand,
}


def eval_expr(expr: str, parms: dict):
    """
    Evaluate the expression `expr` replacing the variables defined in `parms`.
    expr can include python`s arithmetic expressions, and some elementary
    functions.
    """
    # TODO: Improve the workflow in a way that numpy functions
    # and constants be loaded just if they are needed.

    if not isinstance(expr, str):
        return expr

    try:
        return float(expr)
    except (ValueError, TypeError):
        try:
            if expr not in ("J", "j"):
                return complex(expr)
        except (ValueError, TypeError):
            pass

    parms = {
        key.replace("'", "_prima"): val for key, val in parms.items() if val is not None
    }
    expr = expr.replace("'", "_prima")

    while expr in parms:
        expr = parms.pop(expr)
        if not isinstance(expr, str):
            return expr

    # Reduce the parameters
    p_vars = [k for k in parms]
    while True:
        changed = False
        for k in p_vars:
            val = parms.pop(k)
            if not isinstance(val, str):
                parms[k] = val
                continue
            try:
                result = eval_expr(val, parms)
                if result is not None:
                    parms[k] = result
                if val != result:
                    changed = True
            except RecursionError:
                raise
        if not changed:
            break
    parms.update(default_parms)
    try:
        result = eval(expr, parms)
        return result
    except NameError:
        pass
    except TypeError as exc:
        logging.warning("Type Error. Undefined variables in ", expr, exc)
        return None
    return expr


def find_ref(node, root):
    node_items = dict(node.items())
    if "ref" in node_items:
        name_ref = node_items["ref"]
        for refnode in root.findall("./" + node.tag):
            if ("name", name_ref) in refnode.items():
                return refnode
    return node


def operator_to_wolfram(operator) -> str:
    """
    Produce a string with a Wolfram Mathematica expression
    representing the operator.
    """
    from alpsqutip.operators.arithmetic import SumOperator
    from alpsqutip.operators.basic import LocalOperator, Operator, ProductOperator
    from alpsqutip.operators.qutip import Qobj, QutipOperator

    def get_site_identity(site_name):
        site_spec = sites[site_name]
        if "operators" in site_spec:
            return site_spec["operators"]["identity"]
        dim = dimensions[site_name]
        result = qutip.qeye(dim)
        site_spec["operators"] = {"identity": result}
        return result

    if hasattr(operator, "to_wolfram"):
        return operator.to_wolfram()

    if isinstance(operator, Qobj):
        data = operator.data
        if hasattr(data, "toarray"):
            array = data.toarray()
        elif hasattr(data, "to_array"):
            array = data.to_array()
        else:
            raise TypeError(f"Do not know how to convert {type(data)} into a ndarray")

        assert len(array.shape) == 2, f"the shape  {array.shape} is not a matrix"
        return matrix_to_wolfram(array)

    if isinstance(operator, SumOperator):
        assert all(isinstance(term, Operator) for term in operator.terms)
        terms = [operator_to_wolfram(term) for term in operator.terms]
        terms = [term for term in terms if term != "0"]
        return "(" + " + ".join(terms) + ")"

    sites = operator.system.sites
    dimensions = operator.dimensions
    prefactor = operator.prefactor
    if prefactor == 0:
        return "0"

    prefix = "KroneckerProduct["
    if prefactor != 1:
        prefix = f"({prefactor}) * " + prefix

    if isinstance(operator, LocalOperator):
        local_site
        factors = [
            operator.operator if site == local_site else get_site_identity(site)
            for site in sorted(dimensions)
        ]
        factors_str = [operator_to_wolfram(factor) for factor in factors]

        return prefix + ", ".join(factors_str) + "]"

    if isinstance(operator, ProductOperator):
        factors = [
            operator.sites_op.get(site, get_site_identity(site))
            for site in sorted(dimensions)
        ]
        factors_str = [operator_to_wolfram(factor) for factor in factors]

        return prefix + ", ".join(factors_str) + "]"

    if hasattr(operator, "prefactor"):
        return (
            "("
            + str(prefactor)
            + ") * "
            + operator_to_wolfram((operator / prefactor).to_qutip())
        )

    return operator_to_wolfram(operator.to_qutip())


def matrix_to_wolfram(matr: np.ndarray):
    """Produce a string representing the data in the matrix"""
    assert isinstance(
        matr, (np.ndarray, complex, float)
    ), f"{type(matr)} is not ndarray or number"

    def process_number(num):
        assert isinstance(
            num, (float, complex)
        ), f"{type(num)} {num} is not float or complex."
        if isinstance(num, np.ndarray) and len(num) == 1:
            num = num[0]

        if isinstance(num, complex):
            if num.imag == 0:
                return str(num.real).replace("e", "*^")
        return (
            str(num)
            .replace("(", "")
            .replace(")", "")
            .replace("e", "*^")
            .replace("j", "I")
        )

    rows = [
        "{" + (", ".join(process_number(elem) for elem in row)) + "}" for row in matr
    ]
    return "{\n" + ",\n".join(rows) + "\n}"


def next_name(dictionary: dict, s: int = 1, prefix: str = "") -> str:
    """
    Produces a new key for the `dictionary` with a
    `prefix`
    """
    name = f"{prefix}{s}"
    if name in dictionary:
        return next_name(dictionary, s + 1, prefix)
    return name
