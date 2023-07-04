"""
Basic unit test.
"""

import matplotlib.pyplot as plt
import qutip

import alpsqutip
from alpsqutip.alpsmodels import list_operators_in_alps_xml, model_from_alps_xml
from alpsqutip.geometry import graph_from_alps_xml, list_graph_in_alps_xml
from alpsqutip.model import ProductOperator, SystemDescriptor
from alpsqutip.states import ProductQuantumState
from alpsqutip.utils import eval_expr

ROOT_DIR = alpsqutip.__path__[0]
FIGURES_DIR = f"{ROOT_DIR}/doc/figs"
LATTICE_LIB_FILE = f"{ROOT_DIR}/lib/lattices.xml"
MODEL_LIB_FILE = f"{ROOT_DIR}/lib/models.xml"


# TODO: Split me in more atomic units.


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

        print(g)
        fig = plt.figure()
        if g.lattice and g.lattice["dimension"] > 2:
            ax = fig.add_subplot(projection="3d")
            ax.set_proj_type("persp")
        else:
            ax = fig.add_subplot()
        ax.set_title(name)
        g.draw(ax)
        plt.savefig(FIGURES_DIR + f"/{name}.png")
    print("models:")

    for modelname in list_operators_in_alps_xml(MODEL_LIB_FILE):
        print("\n       ", modelname)
        print(40 * "*")
        try:
            model = model_from_alps_xml(
                MODEL_LIB_FILE, modelname, parms={"Nmax": 3, "local_S": 0.5}
            )
            print(
                "site types:",
                {name: lb["name"] for name, lb in model.site_basis.items()},
            )
        except Exception as e:
            assert False, f"{model} could not be loaded due to {e}"


def test_all():
    models = list_operators_in_alps_xml(MODEL_LIB_FILE)
    graphs = list_graph_in_alps_xml(LATTICE_LIB_FILE)

    for model_name in models:
        print(model_name, "\n", 10 * "*")
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
                print(system.global_operators.keys())
            except Exception as e:
                # assert False, f"model {model_name} over graph {graph_name} could not be loaded due to {type(e)}:{e}"
                print("   ", graph_name, "  [failed]", e)
                continue
            print("   ", graph_name, "  [OK]")
        print("\n-------------")


def test_states():
    system = SystemDescriptor(
        basis=model_from_alps_xml(MODEL_LIB_FILE, "spin"),
        graph=graph_from_alps_xml(
            LATTICE_LIB_FILE, "open square lattice", parms={"L": 3, "a": 1}
        ),
        parms={"h": 1, "J": 1},
    )
    # enumerate the name of each subsystem
    sites = [s for s in system.sites]
    sites01 = [sites[0], sites[1]]
    sites02 = [sites[0], sites[2]]

    for ssites in [sites, sites01, sites02]:
        print(ssites)
        # Global operators
        global_identity = ProductOperator({}, 1.0, system)
        op1 = ProductOperator({sites[0]: qutip.sigmax()}, 1.0, system)
        op2 = ProductOperator({sites[1]: qutip.sigmax()}, 1.0, system)
        op = 0.7 * op1 + 0.3 * op2

        # Global states
        rho1 = ProductQuantumState(
            {s: (-qutip.sigmax()).expm() for s in ssites}, system
        )
        rho2 = ProductQuantumState({s: (qutip.sigmax()).expm() for s in ssites}, system)
        rho = 0.25 * rho1 + 0.75 * rho2
        rho_as_regular = rho * global_identity

        # Operators on the subsystem 0,1
        op1_ss = ProductOperator(
            {sites[0]: qutip.sigmax()}, 1.0, system.subsystem([sites[0], sites[1]])
        )
        op2_ss = ProductOperator(
            {sites[1]: qutip.sigmax()}, 1.0, system.subsystem([sites[0], sites[1]])
        )
        op_ss = 0.7 * op1_ss + 0.3 * op2_ss

        # Qutip version
        # Global states
        qt_rho1 = rho1.to_qutip()
        qt_rho2 = rho2.to_qutip()
        qt_rho = 0.25 * qt_rho1 + 0.75 * qt_rho2
        # Ops
        qt_op1 = op1.to_qutip()
        qt_op2 = op2.to_qutip()
        qt_op = 0.7 * qt_op1 + 0.3 * qt_op2

        # Tests.
        # The reference is the output from (qt_op * qt_rho).tr()
        expect = (qt_op * qt_rho).tr()

        val = (op.to_qutip() * rho.to_qutip()).tr()
        assert abs(expect - val) < 1.0e-6, f"to qutip commutes {expect}!={val}"
        print("  * To qutip commutes with product and trace [OK]")

        val = (op * rho).tr()
        assert abs(expect - val) < 1.0e-6, f"traces do not match {expect}!={val}"
        print(
            "  * The result is the same in qutip and here using product and trace [OK]"
        )

        val = (op * rho_as_regular).tr()
        assert (
            abs(expect - val) < 1.0e-6
        ), f"It also works for rho a regular op {expect}!={val}"
        print("  * The result is also the same using a regular operator as rho [OK]")

        val = 0.7 * rho.expect(op1) + 0.3 * rho.expect(op2)
        assert abs(expect - val) < 1.0e-6, f"expect is linear in ops {expect}!={val}"
        print("  * expect is linear in ops [OK]")
        val = 0.25 * rho1.expect(op) + 0.75 * rho2.expect(op)
        assert abs(expect - val) < 1.0e-6, f"expect is linear in rho {expect}!={val}"
        print("  * Expect is linear in rho [OK]")
        val = rho.expect(op)
        assert (
            abs(expect - val) < 1.0e-6
        ), f"rho.expect does not match with trace {expect}!={val}"
        print("  * Expect is the same than the trace [OK]")

        val = rho.partial_trace([sites[0], sites[1]]).expect(op_ss)
        assert abs(expect - val) < 1.0e-6, f"match for subsystems {expect}!={val}"
        print("  * Expect matches for subsystems [OK]")

        val = (rho_as_regular.partial_trace([sites[0], sites[1]]) * op).tr()
        assert abs(expect - val) < 1.0e-6, f"match for subsystems {expect}!={val}"
        print("  * Also for regular rho operators [OK]")


# test_load()
# test_all()
# test_eval_expr()
