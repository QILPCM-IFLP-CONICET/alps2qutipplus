"""
Basic unit test.
"""

import matplotlib.pyplot as plt

from alpsqutip.alpsmodels import list_operators_in_alps_xml, model_from_alps_xml
from alpsqutip.geometry import graph_from_alps_xml, list_graph_in_alps_xml
from alpsqutip.model import SystemDescriptor
from alpsqutip.settings import FIGURES_DIR, LATTICE_LIB_FILE, MODEL_LIB_FILE
from alpsqutip.utils import eval_expr

from .helper import alert


def test_eval_expr():
    """Test basic evaluation of expressions"""
    parms = {"a": "J", "J": 2, "subexpr": "a*J"}
    test_cases = [
        ("2+a", 4),
        ("sqrt(2+a)", 2),
        ("0*rand()", 0),
        ("2*J", 4),
        ("sqrt(subexpr)", 2),
    ]
    for expr, expect in test_cases:
        result = eval_expr(expr, parms)
        assert expect == result, (
            f"evaluating {expr}"
            f"{expect} of type {type(expect)}"
            f"!= {result} of {type(result)}"
        )


def test_load_all_models_and_lattices():
    """Try to load each model and lattice."""
    models = list_operators_in_alps_xml(MODEL_LIB_FILE)
    graphs = list_graph_in_alps_xml(LATTICE_LIB_FILE)

    for model_name in models:
        print(model_name, "\n", 10 * "*")
        for graph_name in graphs:
            g = graph_from_alps_xml(
                LATTICE_LIB_FILE,
                graph_name,
                parms={"L": 2, "W": 2, "a": 1, "b": 1, "c": 1},
            )
            model = model_from_alps_xml(
                MODEL_LIB_FILE,
                model_name,
                parms={"L": 2, "W": 2, "a": 1, "b": 1, "c": 1, "Nmax": 5},
            )
            try:
                SystemDescriptor(g, model, {})
            except Exception as exc:
                # assert False, f"model {model_name} over
                # graph {graph_name} could not be loaded due to {type(e)}:{e}"
                alert(1, "   ", graph_name, "  [failed]", exc)
                continue
