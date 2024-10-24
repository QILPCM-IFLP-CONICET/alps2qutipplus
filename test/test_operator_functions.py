"""
Basic unit test for operator functions.
"""

import numpy as np
import qutip

from alpsqutip.operators import SumOperator
from alpsqutip.operators.functions import (
    eigenvalues,
    hermitian_and_antihermitian_parts,
    log_op,
    relative_entropy,
    simplify_sum_operator,
)
from alpsqutip.utils import operator_to_wolfram

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


def qutip_relative_entropy(qutip_1, qutip_2):
    """Compute the relative entropy"""
    # if both operators are the same operator,
    # the relative entropy is 0.
    if qutip_1 is qutip_2 or qutip_1 == qutip_2:
        return 0.0

    # Now, at least one operator is not None
    dim = 1
    if isinstance(qutip_1, qutip.Qobj):
        dim = qutip_1.data.shape[0]
        if qutip_2 is None:
            qutip_2 = 1 / dim
        # if the second argument is a number, compute in terms of the vn entropy:
        if not isinstance(qutip_2, qutip.Qobj):
            return qutip_1.tr() * np.log(qutip_2) - qutip.entropy_vn(qutip_1)

    elif isinstance(qutip_2, qutip.Qobj):
        dim = qutip_2.data.shape[0]
        if qutip_1 is None:
            qutip_1 = 1 / dim
        # if the first argument is a number, compute in terms of the logarithm
        # of the second:
        if not isinstance(qutip_1, qutip.Qobj):
            return qutip_1 * (np.log(qutip_1) - qutip_2.logm().tr())
    else:  # both are a numbers or None
        if qutip_1 is None:
            qutip_1 = 1
        if qutip_2 is None:
            qutip_2 = 1
        return qutip_1 * np.log(qutip_1 / qutip_2)

    # Now, both operators are qutip operators. It is safe to use
    # the standard routine.
    return qutip.entropy_relative(qutip_1, qutip_2)


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

        if key1 != "mixture of first and second partially polarized":
            continue

        assert abs(rho.tr() - 1) < 1e-6 and abs(qutip_states[key1].tr() - 1) < 1e-6
        check_equality(relative_entropy(rho, rho), 0)
        check_equality(
            qutip.entropy_relative(qutip_states[key1], qutip_states[key1]), 0
        )
        for key2, sigma in test_cases_states.items():
            if key2 != "fully mixed":
                continue
            rel_entr = relative_entropy(rho, sigma)
            rel_entr_qutip = qutip_relative_entropy(
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

                print(key1, operator_to_wolfram(rho))
                print(key1, operator_to_wolfram(rho.to_qutip()))

                print(key2, operator_to_wolfram(sigma))
                print(key2, operator_to_wolfram(sigma.to_qutip()))

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
        print("name:", name, " of type", type(operator))
        op_exp = (test_op).expm()
        print("     exp of type:", type(op_exp))
        op_exp_log = log_op(op_exp)
        print("     log of exp of type:", type(op_exp_log))
        delta = test_op - op_exp_log
        spectral_norm_error = max(abs(x) for x in delta.to_qutip().eigenenergies())
        if spectral_norm_error > 0.000001:
            clean = False
            print("    log(exp(op))!=op.")
            print("    ", spectral_norm_error)
            if True:
                print(
                    "\n  Op=",
                    operator_to_wolfram(test_op.to_qutip().full()),
                    ";",
                )
                print(
                    "\n  ExpOp=",
                    operator_to_wolfram(op_exp.to_qutip().full()),
                    ";",
                )
                print(
                    "\n  LogExpOp=",
                    operator_to_wolfram(op_exp_log.to_qutip().full()),
                    ";",
                )

    assert clean


# test_load()
# test_all()
# test_eval_expr()
