"""
Basic unit test for operator functions.
"""

import numpy as np
import pytest
import qutip

from alpsqutip.operators import SumOperator
from alpsqutip.operators.functions import (
    eigenvalues,
    hermitian_and_antihermitian_parts,
    log_op,
    relative_entropy,
    spectral_norm,
)
from alpsqutip.utils import operator_to_wolfram

from .helper import (
    CHAIN_SIZE,
    HAMILTONIAN,
    OPERATOR_TYPE_CASES,
    OPERATORS,
    SITES,
    SYSTEM,
    SZ_TOTAL,
    TEST_CASES_STATES,
    check_equality,
    check_operator_equality,
)

# from alpsqutip.settings import VERBOSITY_LEVEL


splus0 = SYSTEM.site_operator(f"Splus@{SITES[0]}")
splus1 = SYSTEM.site_operator(f"Splus@{SITES[1]}")

spsp_hc = SumOperator(
    (
        splus0 * splus1,
        (splus0 * splus1).dag(),
    ),
    SYSTEM,
    True,
)


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
    for name, operator in OPERATORS.items():
        print("name", name, type(operator))
        op_re, op_im = hermitian_and_antihermitian_parts(operator)
        op_qutip = operator.to_qutip()
        op_re_qutip = 0.5 * (op_qutip + op_qutip.dag())
        op_im_qutip = 0.5 * 1j * (op_qutip.dag() - op_qutip)
        assert op_re.isherm and op_im.isherm
        assert check_operator_equality(op_re.to_qutip(), op_re_qutip)
        assert check_operator_equality(op_im.to_qutip(), op_im_qutip)


QUTIP_TEST_CASES_STATES = {
    key: operator.to_qutip() for key, operator in TEST_CASES_STATES.items()
}


@pytest.mark.parametrize(
    ["key_rho", "key_sigma"],
    [
        (
            key_rho,
            key_sigma,
        )
        for key_rho in TEST_CASES_STATES
        for key_sigma in TEST_CASES_STATES
    ],
)
def test_relative_entropy(key_rho, key_sigma):
    qutip_states = QUTIP_TEST_CASES_STATES
    rho = TEST_CASES_STATES[key_rho]
    sigma = TEST_CASES_STATES[key_sigma]
    rho_qutip = qutip_states[key_rho]
    sigma_qutip = qutip_states[key_sigma]

    if key_rho == key_sigma:
        assert abs(rho.tr() - 1) < 1e-6 and abs(rho_qutip.tr() - 1) < 1e-6
        check_equality(relative_entropy(rho, rho), 0)
        check_equality(qutip.entropy_relative(rho_qutip, rho_qutip), 0)

    rel_entr = relative_entropy(rho, sigma)
    rel_entr_qutip = qutip_relative_entropy(rho_qutip, sigma_qutip)
    # infinity quantities cannot be compared...
    if rel_entr_qutip == np.inf and rel_entr > 10:
        return
    if abs(rel_entr - rel_entr_qutip) > 1.0e-6:
        print("  ", [key_rho, key_sigma])

        print(key_rho, operator_to_wolfram(rho))
        print(key_rho, operator_to_wolfram(rho.to_qutip()))

        print(key_sigma, operator_to_wolfram(sigma))
        print(key_sigma, operator_to_wolfram(sigma.to_qutip()))
        assert (
            False
        ), f" in S({key_rho}|{key_sigma}),  {rel_entr} (alps2qutip) !=   {rel_entr_qutip} (qutip)"


def test_eigenvalues():
    """Tests eigenvalues of different operator objects"""
    spectrum = sorted(eigenvalues(SZ_TOTAL))
    for s in range(CHAIN_SIZE):
        min_err = min(abs(e_val - s + 0.5 * CHAIN_SIZE) for e_val in spectrum)
        assert min_err < 1e-6, f"closest eigenvalue at {min_err}"

    # Fully mixed operator
    spectrum = sorted(eigenvalues(TEST_CASES_STATES["fully mixed"]))
    assert all(abs(s - 0.5**CHAIN_SIZE) < 1e-6 for s in spectrum)

    # Ground state energy
    # Compute the minimum eigenenergy with qutip
    e0_qutip = min(
        HAMILTONIAN.to_qutip().eigenenergies(sparse=True, sort="low", eigvals=10)
    )
    # use the alpsqutip routine
    e0 = min(eigenvalues(HAMILTONIAN, sparse=True, sort="low", eigvals=10))
    assert abs(e0 - e0_qutip) < 1.0e-6

    #  e^(sz)/Tr e^(sz)
    spectrum = sorted(eigenvalues(TEST_CASES_STATES["gibbs_sz"]))
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
    for name, operator in OPERATORS.items():
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

    for name, operator in OPERATORS.items():
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


@pytest.mark.parametrize(
    ["name", "operator"],
    [(name, operator) for name, operator in OPERATOR_TYPE_CASES.items()],
)
def test_spectral_norm(name, operator):
    """
    Test the spectral norm
    """
    print("spectral norm of", name, "of type", type(operator))
    qutip_sn = spectral_norm(operator.to_qutip())
    op_sn = spectral_norm(operator)
    assert abs(op_sn - qutip_sn) < 1e-6, (
        f"||op_{name}|-|qutip_{name}||="
        f"|{op_sn}-{qutip_sn}|={abs(op_sn-qutip_sn)}>1e-2."
    )


# test_load()
# test_all()
# test_eval_expr()
