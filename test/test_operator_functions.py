"""
Basic unit test for operator functions.
"""

import numpy as np

from alpsqutip.operator_functions import (
    eigenvalues,
    log_op,
    hermitian_and_antihermitian_parts,
    relative_entropy,
    simplify_sum_operator,
    spectral_norm,
)
from alpsqutip.operators import SumOperator

from .helper import (
    CHAIN_SIZE,
    check_operator_equality,
    check_equality,
    global_identity,
    hamiltonian,
    sites,
    system,
    sx_total,
    sz_total,
    test_cases_states,
)

# from alpsqutip.settings import VERBOSITY_LEVEL


splus0 = system.site_operator(f"Splus@{sites[0]}")
splus1 = system.site_operator(f"Splus@{sites[1]}")

spsp_hc = SumOperator(
    (
        splus0 * splus1,
        (splus0 * splus1).dag(),
    ),
    system,
    True,
)


operators = {
    "Identity": global_identity,
    "sz_total": sz_total,
    "sx_total": sx_total,
    "sx_total_sq": sx_total * sx_total,
    "sx+ 3j sz": sx_total + (3 * 1j) * sz_total,
    "splus*splus": splus0 * splus1,
    "splus*splus+hc": spsp_hc,
    "hamiltonian": hamiltonian,
    "nonhermitician": hamiltonian + (3 * 1j) * sz_total,
}


def compare_spectrum(spectrum1, spectrum2):
    assert (
        max(abs(np.array(sorted(spectrum1)) - np.array(sorted(spectrum2))))
        < 1.0e-12
    )


def test_decompose_hermitician():
    """Test the decomposition as Q=A+iB with
    A=A.dag() and B=B.dag()
    """
    for name, operator in operators.items():
        print("name", name, type(operator))
        op_re, op_im = hermitian_and_antihermitian_parts(operator)
        op_qutip = operator.to_qutip()
        op_re_qutip = 0.5 * (op_qutip + op_qutip.dag())
        op_im_qutip = 0.5 * 1j * (op_qutip.dag() - op_qutip)
        assert op_re.isherm and op_im.isherm
        assert check_operator_equality(op_re.to_qutip(), op_re_qutip)
        assert check_operator_equality(op_im.to_qutip(), op_im_qutip)


def test_simplify_sum_operator():
    def do_test(name, operator):
        if isinstance(operator, list):
            for op_case in operator:
                do_test(name, operator)
            return

        operator_simpl = simplify_sum_operator(operator)
        assert check_equality(operator.to_qutip(), operator_simpl.to_qutip())
        assert operator.to_qutip().isherm == operator_simpl.isherm

    for name, operator_case in operators.items():
        print("name", name, type(operator_case))
        do_test(name, operator_case)


def test_eigenvalues():
    """Tests eigenvalues of different operator objects"""
    spectrum = sorted(eigenvalues(sz_total))
    for s in range(CHAIN_SIZE):
        min_err = min(abs(e_val - s + 0.5 * CHAIN_SIZE) for e_val in spectrum)
        assert min_err < 1e-6, f"closest eigenvalue at min_err"

    # Fully mixed operator
    spectrum = sorted(eigenvalues(test_cases_states["fully mixed"]))
    assert all(abs(s - 0.5**CHAIN_SIZE) < 1e-6 for s in spectrum)

    # Ground state energy
    # Compute the minimum eigenenergy with qutip
    e0_qutip = min(
        hamiltonian.to_qutip().eigenenergies(
            sparse=True, sort="low", eigvals=10
        )
    )
    # use the alpsqutip routine
    e0 = min(eigenvalues(hamiltonian, sparse=True, sort="low", eigvals=10))
    print("e0=", e0)
    assert abs(e0 - e0_qutip) < 1.0e-6

    #  e^(sz)/Tr e^(sz)
    spectrum = sorted(eigenvalues(test_cases_states["gibbs_sz"]))
    expected_local_spectrum = np.array([np.exp(-0.5), np.exp(0.5)])
    expected_local_spectrum = expected_local_spectrum / sum(
        expected_local_spectrum
    )

    expected_spectrum = expected_local_spectrum.copy()
    for i in range(CHAIN_SIZE - 1):
        expected_spectrum = np.append(
            expected_spectrum * expected_local_spectrum[0],
            expected_spectrum * expected_local_spectrum[1],
        )

    compare_spectrum(expected_spectrum, spectrum)


def test_log_op():
    """Check log_op
    Due to numerical errors, log_op(operator.expm())~operator
    only if operator has an small spectral norm.

    """
    for name, operator in operators.items():
        print("name:", name)
        test_op = 0.01 * operator
        op_exp = (test_op).expm()
        op_exp_log = log_op(op_exp)
        delta = test_op - op_exp_log
        spectral_norm_error = max(
            abs(x) for x in delta.to_qutip().eigenenergies()
        )
        print(spectral_norm_error)
        assert spectral_norm_error < 0.1


# test_load()
# test_all()
# test_eval_expr()
