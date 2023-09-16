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


def test_load():
    for name in list_graph_in_alps_xml(LATTICE_LIB_FILE):
        try:
            g = graph_from_alps_xml(
                LATTICE_LIB_FILE, name, parms={"L": 3, "W": 3, "a": 1, "b": 1, "c": 1}
            )
        except Exception as e:
            assert False, f"geometry {name} could not be loaded due to {e}"

        alert(1, g)
        fig = plt.figure()
        if g.lattice and g.lattice["dimension"] > 2:
            ax = fig.add_subplot(projection="3d")
            ax.set_proj_type("persp")
        else:
            ax = fig.add_subplot()
        ax.set_title(name)
        g.draw(ax)
        plt.savefig(FIGURES_DIR + f"/{name}.png")
    alert(1, "models:")

    for modelname in list_operators_in_alps_xml(MODEL_LIB_FILE):
        alert(1, "\n       ", modelname)
        alert(1, 40 * "*")
        try:
            model = model_from_alps_xml(
                MODEL_LIB_FILE, modelname, parms={"Nmax": 3, "local_S": 0.5}
            )
            alert(
                1,
                "site types:",
                {name: lb["name"] for name, lb in model.site_basis.items()},
            )
        except Exception as e:
            assert False, f"{model} could not be loaded due to {e}"


def test_all():
    models = list_operators_in_alps_xml(MODEL_LIB_FILE)
    graphs = list_graph_in_alps_xml(LATTICE_LIB_FILE)

    for model_name in models:
        alert(1, model_name, "\n", 10 * "*")
        for graph_name in graphs:
            g = graph_from_alps_xml(
                LATTICE_LIB_FILE,
                graph_name,
                parms={"L": 3, "W": 3, "a": 1, "b": 1, "c": 1},
            )
            model = model_from_alps_xml(
                MODEL_LIB_FILE,
                model_name,
                parms={"L": 3, "W": 3, "a": 1, "b": 1, "c": 1, "Nmax": 5},
            )
            try:
                system = SystemDescriptor(g, model, {})
                alert(1, system.operators["global_operators"].keys())
            except Exception as e:
                # assert False, f"model {model_name} over graph {graph_name} could not be loaded due to {type(e)}:{e}"
                alert(1, "   ", graph_name, "  [failed]", e)
                continue
            alert(1, "   ", graph_name, "  [OK]")
        alert(1, "\n-------------")
