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


def test_load_and_plot_graph():
    for name in list_graph_in_alps_xml(LATTICE_LIB_FILE):
        try:
            g = graph_from_alps_xml(
                LATTICE_LIB_FILE,
                name,
                parms={"L": 3, "W": 3, "a": 1, "b": 1, "c": 1},
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


def test_graph_operations():
    graph = graph_from_alps_xml(
        filename=LATTICE_LIB_FILE, name="chain lattice", parms={"L": 10, "a": 1}
    )

    graph_a = graph.subgraph(
        tuple((s for i, s in enumerate(graph.nodes.keys()) if i < 3)), name="a"
    )
    graph_b = graph.subgraph(
        tuple((s for i, s in enumerate(graph.nodes.keys()) if 2 <= i < 5)), name="b"
    )
    graph_c = graph.subgraph(
        tuple((s for i, s in enumerate(graph.nodes.keys()) if 5 < i < 8)), name="c"
    )

    print(graph_a.name, "with", graph_a.nodes.keys())
    print(graph_b.name, "with", graph_b.nodes.keys())
    print(graph_c.name, "with", graph_c.nodes.keys())
    cases = {
        "graph_ab": graph_a + graph_b,
        "graph_ac": graph_a + graph_c,
        "graph_bc": graph_b + graph_c,
    }
    for name, case in cases.items():
        print("subgraph", name, "is", case.name, "with", case.nodes.keys())

    assert (
        len(cases["graph_ab"].nodes)
        == len(set(graph_a.nodes).union(set(graph_b.nodes)))
        < 6
    )
    assert (
        len(cases["graph_ac"].nodes)
        == len(set(graph_a.nodes).union(set(graph_c.nodes)))
        < 6
    )
    assert (
        len(cases["graph_bc"].nodes)
        == len(set(graph_b.nodes).union(set(graph_c.nodes)))
        < 6
    )
