import numpy as np
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
        key.replace("'", "_prima"): val
        for key, val in parms.items()
        if val is not None
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
        print("Type Error. Undefined variables in ", expr, exc)
        assert False
        for p in parms:
            if parms[p] is None:
                print("   ", p, "->", parms[p])
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


def next_name(dictionary: dict, s: int = 1, prefix: str = "") -> str:
    """
    Produces a new key for the `dictionary` with a
    `prefix`
    """
    name = f"{prefix}{s}"
    if name in dictionary:
        return next_name(dictionary, s + 1, prefix)
    return name
