import multiprocessing as mp
import pickle

import pytest

from alpsqutip.geometry import GraphDescriptor
from alpsqutip.model import SystemDescriptor
from alpsqutip.operators import Operator

from .helper import (
    FULL_TEST_CASES,
    OPERATOR_TYPE_CASES,
    SYSTEM,
    TEST_CASES_STATES,
    check_operator_equality,
)


def test_serialize_graph():
    print("test serialize graph")
    graph = SYSTEM.spec["graph"]
    a = pickle.dumps(graph)
    reconstructed_graph = pickle.loads(a)
    assert isinstance(reconstructed_graph, GraphDescriptor)
    assert graph == reconstructed_graph


def test_serialize_system():
    print("test serialize system")
    a = pickle.dumps(SYSTEM)
    reconstructed_system = pickle.loads(a)
    assert isinstance(reconstructed_system, SystemDescriptor)
    assert SYSTEM == reconstructed_system


@pytest.mark.parametrize(["name", "operator"], list(FULL_TEST_CASES.items()))
def test_serialize_operator(name, operator):
    print("test serialize", name)

    a = pickle.dumps(operator)
    reconstructed_operator = pickle.loads(a)
    assert isinstance(reconstructed_operator, Operator)
    reconstructed_operator._set_system_(operator.system)
    assert check_operator_equality(operator, reconstructed_operator, tolerance=1e-8)


def worker_add_a_number(q):
    op1, number = q.get()
    if hasattr(op1, "terms"):
        system = op1.system
        for t in op1.terms:
            assert t.system is system, f"{id(t.system)}\n is not {id(system)}."
            print("* OK")

    q.put(op1 + number)


@pytest.mark.parametrize(["name", "operator"], list(FULL_TEST_CASES.items()))
def test_process_add_number(name, operator):
    ctx = mp.get_context("fork")

    print("test process add number", name)

    if hasattr(operator, "terms"):
        system = operator.system
        for t in operator.terms:
            assert t.system is system

    my_queue = ctx.Queue()
    p = ctx.Process(target=worker_add_a_number, args=(my_queue,))
    p.start()
    my_queue.put(
        (
            operator,
            1.0,
        )
    )
    p.join()
    result_worker = my_queue.get()
    result_mine = operator + 1.0
    print(type(result_worker), type(result_mine))

    result_worker._set_system_(operator.system)
    result_mine._set_system_(operator.system)
    p.close()
    assert check_operator_equality(result_worker, result_mine, tolerance=1e-8)


def worker_expect(q):
    state, obs = q.get()
    q.put(state.expect(obs))


@pytest.mark.parametrize(
    ["state_name", "operator_name"],
    [
        (
            state_name,
            operator_name,
        )
        for state_name in TEST_CASES_STATES
        for operator_name in OPERATOR_TYPE_CASES
    ],
)
def test_process_expect(state_name, operator_name):
    ctx = mp.get_context("fork")

    print("test process expect", state_name, operator_name)

    state = TEST_CASES_STATES[state_name]
    operator = OPERATOR_TYPE_CASES[operator_name]
    my_queue = ctx.Queue()
    p = ctx.Process(target=worker_expect, args=(my_queue,))
    p.start()
    my_queue.put(
        (
            state,
            operator,
        )
    )
    p.join()
    result_worker = my_queue.get()
    p.close()
    result_mine = state.expect(operator)
    assert abs(result_worker - result_mine) < 1e-9
