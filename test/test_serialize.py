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
    graph = SYSTEM.spec["graph"]
    a = pickle.dumps(graph)
    reconstructed_graph = pickle.loads(a)
    assert isinstance(reconstructed_graph, GraphDescriptor)
    assert graph == reconstructed_graph


def test_serialize_system():
    a = pickle.dumps(SYSTEM)
    reconstructed_system = pickle.loads(a)
    assert isinstance(reconstructed_system, SystemDescriptor)
    assert SYSTEM == reconstructed_system


@pytest.mark.parametrize(["name", "operator"], list(FULL_TEST_CASES.items()))
def test_serialize_operator(name, operator):
    a = pickle.dumps(operator)
    reconstructed_operator = pickle.loads(a)
    assert isinstance(reconstructed_operator, Operator)
    assert check_operator_equality(operator, reconstructed_operator, tolerance=1e-8)


def worker_add_a_number(q):
    op1, number = q.get()
    q.put(op1 + number)


@pytest.mark.parametrize(["name", "operator"], list(FULL_TEST_CASES.items()))
def test_process_add_number(name, operator):
    from multiprocessing import Process, Queue

    my_queue = Queue()
    p = Process(target=worker_add_a_number, args=(my_queue,))
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
    from multiprocessing import Process, Queue

    state = TEST_CASES_STATES[state_name]
    operator = OPERATOR_TYPE_CASES[operator_name]
    my_queue = Queue()
    p = Process(target=worker_expect, args=(my_queue,))
    p.start()
    my_queue.put(
        (
            state,
            operator,
        )
    )
    p.join()
    result_worker = my_queue.get()
    result_mine = state.expect(operator)
    assert abs(result_worker - result_mine) < 1e-9
