"""
Basic unit test for operator functions.
"""

import numpy as np
import qutip

from alpsqutip.operator_functions import (
    eigenvalues,
    hermitian_and_antihermitian_parts,
    log_op,
    relative_entropy,
    simplify_sum_operator,
    spectral_norm,
)
from alpsqutip.operators import SumOperator
from alpsqutip.utils import matrix_to_wolfram

from .helper import (
    CHAIN_SIZE,
    check_equality,
    check_operator_equality,
    global_identity,
    hamiltonian,
    sites,
    sx_total,
    system,
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
    assert max(abs(np.array(sorted(spectrum1)) - np.array(sorted(spectrum2)))) < 1.0e-12


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
                do_test(name, op_case)
            return

        operator_simpl = simplify_sum_operator(operator)
        assert check_equality(operator.to_qutip(), operator_simpl.to_qutip())
        assert operator.to_qutip().isherm == operator_simpl.isherm

    for name, operator_case in operators.items():
        print("name", name, type(operator_case))
        do_test(name, operator_case)


def test_relative_entropy():
    qutip_states = {
        key: operator.to_qutip() for key, operator in test_cases_states.items()
    }
    clean = True
    for key1, rho in test_cases_states.items():
        assert abs(rho.tr() - 1) < 0.001 and abs(qutip_states[key1].tr() - 1) < 0.001
        print("\n\n", 30 * " ", "rho:", key1)
        check_equality(relative_entropy(rho, rho), 0)
        check_equality(
            qutip.entropy_relative(qutip_states[key1], qutip_states[key1]), 0
        )

        for key2, sigma in test_cases_states.items():
            rel_entr = relative_entropy(rho, sigma)
            rel_entr_qutip = qutip.entropy_relative(
                qutip_states[key1], qutip_states[key2]
            )
            # infinity quantities cannot be compared...
            if rel_entr_qutip == np.inf and rel_entr > 10:
                continue
            if abs(rel_entr - rel_entr_qutip) > 1.0e-6:
                if clean:
                    print("Relative entropy mismatch")
                clean = False
                print("  ", [key1, key2])
                print(f"   {rel_entr} (alps2qutip) !=   {rel_entr_qutip} (qutip)")

                assert clean


def test_eigenvalues():
    """Tests eigenvalues of different operator objects"""
    spectrum = sorted(eigenvalues(sz_total))
    for s in range(CHAIN_SIZE):
        min_err = min(abs(e_val - s + 0.5 * CHAIN_SIZE) for e_val in spectrum)
        assert min_err < 1e-6, f"closest eigenvalue at {min_err}"

    # Fully mixed operator
    spectrum = sorted(eigenvalues(test_cases_states["fully mixed"]))
    assert all(abs(s - 0.5**CHAIN_SIZE) < 1e-6 for s in spectrum)

    # Ground state energy
    # Compute the minimum eigenenergy with qutip
    e0_qutip = min(
        hamiltonian.to_qutip().eigenenergies(sparse=True, sort="low", eigvals=10)
    )
    # use the alpsqutip routine
    e0 = min(eigenvalues(hamiltonian, sparse=True, sort="low", eigvals=10))
    assert abs(e0 - e0_qutip) < 1.0e-6

    #  e^(sz)/Tr e^(sz)
    spectrum = sorted(eigenvalues(test_cases_states["gibbs_sz"]))
    expected_local_spectrum = np.array([np.exp(-0.5), np.exp(0.5)])
    expected_local_spectrum = expected_local_spectrum / sum(expected_local_spectrum)

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

    clean = True
    for name, operator in operators.items():
        test_op = operator + 0.0001
        # logm does now work well with non hermitician operators
        if not test_op.isherm:
            continue
        print("\n\nname:", name, type(operator))
        op_log = log_op(test_op)
        op_log_exp = op_log.expm()
        delta = test_op - op_log_exp
        spectral_norm_error = max(abs(x) for x in delta.to_qutip().eigenenergies())

        if spectral_norm_error > 0.0001:
            clean = False
            print("    exp(log(op))!=op.")
            print("    ", spectral_norm_error)

    for name, operator in operators.items():
        test_op = operator
        if not test_op.isherm:
            continue
        print("name:", name)
        op_exp = (test_op).expm()
        op_exp_log = log_op(op_exp)
        delta = test_op - op_exp_log
        spectral_norm_error = max(abs(x) for x in delta.to_qutip().eigenenergies())
        if spectral_norm_error > 0.000001:
            clean = False
            print("    log(exp(op))!=op.")
            print("    ", spectral_norm_error)
            if True:
                print(
                    "\n  Op=",
                    matrix_to_wolfram(test_op.to_qutip().full()),
                    ";",
                )
                print(
                    "\n  ExpOp=",
                    matrix_to_wolfram(op_exp.to_qutip().full()),
                    ";",
                )
                print(
                    "\n  LogExpOp=",
                    matrix_to_wolfram(op_exp_log.to_qutip().full()),
                    ";",
                )

    assert clean


# test_load()
# test_all()
# test_eval_expr()
