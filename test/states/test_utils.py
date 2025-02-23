from test.helper import check_operator_equality, operator_type_cases

import numpy as np

from alpsqutip.operators.states.utils import safe_exp_and_normalize


def test_safe_exp_and_normalize():

    for name, operator in operator_type_cases.items():
        print("checking safe_exp_and_normalize for", name)
        operator_qutip = operator.to_qutip()

        op_exp_qutip = operator_qutip.expm()
        z_qutip = op_exp_qutip.tr()
        op_exp_qutip = op_exp_qutip / z_qutip
        lnz_qutip = np.log(z_qutip)
        assert abs(op_exp_qutip.tr() - 1) < 1e-9

        # Using safe_exp_and_normalize over Qobj objects.
        op_exp_qutip_sen, lnz_sen = safe_exp_and_normalize(operator.to_qutip())

        assert abs(op_exp_qutip_sen.tr() - 1) < 1e-9
        assert (
            abs(lnz_sen - lnz_qutip) < 1e-9
        ), f"{lnz_sen} != {lnz_qutip}  Delta approx {abs(lnz_sen-lnz_qutip)}"
        check_operator_equality(op_exp_qutip_sen, op_exp_qutip)

        op_exp_norm, lnz = safe_exp_and_normalize(operator)

        assert abs(op_exp_norm.tr() - 1) < 1e-9
        assert (
            abs(lnz - lnz_qutip) < 1e-9
        ), f"{lnz} != {lnz_qutip}  Delta approx {abs(lnz-lnz_qutip)}"
        check_operator_equality(op_exp_norm.to_qutip(), op_exp_qutip)
