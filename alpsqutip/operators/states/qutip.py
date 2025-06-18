"""
Qutip representation for density operators.

Be careful: just use this class for states of small systems.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
from qutip import Qobj  # type: ignore[import-untyped]

from alpsqutip.model import SystemDescriptor
from alpsqutip.operators.basic import Operator, ScalarOperator
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states.basic import DensityOperatorMixin


class QutipDensityOperator(DensityOperatorMixin, QutipOperator):
    """
    Qutip representation of a density operator
    """

    def __init__(
        self,
        qoperator: Qobj,
        system: Optional[SystemDescriptor] = None,
        names=None,
        prefactor=1,
        normalized=False,
    ):
        self._normalized = normalized
        super().__init__(qoperator, system, names, prefactor)
        self.normalize()

    def __add__(self, operand) -> Operator:
        if isinstance(operand, (int, float, np.float64)):
            if operand >= 0:
                return QutipDensityOperator(
                    self.operator * self.prefactor + operand,
                    self.system,
                )
            logging.warning(
                f"Adding {operand} to a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )
        if isinstance(operand, (complex, np.complex128)):
            logging.warning(
                f"Adding {operand} to a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )

        # TODO: check me again
        op_qo = operand.to_qutip()
        if isinstance(operand, DensityOperatorMixin):
            op_qo = op_qo * self.prefactor
            return QutipDensityOperator(op_qo, self.system or op_qo.system)
        return QutipOperator(op_qo, self.system or op_qo.system)

    def __mul__(self, operand) -> Operator:
        if isinstance(operand, (int, float, np.float64)):
            if operand >= 0:
                return QutipDensityOperator(
                    self.operator,
                    self.system,
                    self.site_names,
                    self.prefactor * operand,
                )
            logging.warning(
                f"Multiplying {operand} with a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * (self.prefactor * operand),
                self.system,
            )
        if isinstance(operand, (complex, np.complex128)):
            logging.warning(
                f"Multiplying {operand} with a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * (self.prefactor * operand),
                self.system,
            )

        block_self = tuple(self.site_names)
        block_other = tuple(
            (site for site in operand.acts_over() if site not in block_self)
        )
        block = block_self + block_other
        system = self.system.union(operand.system)
        state = self
        # If one of the operators lives in a smaller system, extend it in a way that block
        # is contained on the system of each operator.
        if any(site not in operand.system.sites for site in block_self):
            operand = QutipOperator(
                operand.to_qutip(), system, operand.site_names, operand.prefactor
            )
        if any(site not in self.system.sites for site in block_other):
            state = QutipOperator(
                state.to_qutip(), system, state.site_names, state.prefactor
            )

        op_qo = operand.to_qutip(block)
        rho_qo = state.to_qutip(block)
        return QutipOperator(
            rho_qo * op_qo, names={s: i for i, s in enumerate(block)}, system=system
        )

    def __neg__(self):
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        self.normalize()
        return QutipOperator(self.operator, self.system, self.site_names, -1)

    def __radd__(self, operand) -> Operator:
        if isinstance(operand, (int, float, np.float64)):
            if operand >= 0:
                return QutipDensityOperator(
                    self.operator * self.prefactor + operand,
                    self.system,
                )
            logging.warning(
                f"Adding {operand} to a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )
        if isinstance(operand, (complex, np.complex128)):
            logging.warning(
                f"Adding {operand} to a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )

        # TODO: check me again
        op_qo = operand.to_qutip()
        if isinstance(operand, DensityOperatorMixin):
            op_qo = op_qo * self.prefactor
            return QutipDensityOperator(op_qo, self.system or op_qo.system)
        return QutipOperator(op_qo, self.system or op_qo.system)

    def __rmul__(self, operand) -> Operator:
        if isinstance(operand, (int, float, np.float64)):
            if operand >= 0:
                return QutipDensityOperator(
                    self.operator,
                    self.system,
                    self.site_names,
                    self.prefactor * operand,
                )
            logging.warning(
                f"Multiplying {operand} with a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * (self.prefactor * operand),
                self.system,
            )
        if isinstance(operand, (complex, np.complex128)):
            logging.warning(
                f"Multiplying {operand} with a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * (self.prefactor * operand),
                self.system,
            )
        block_self = tuple(self.site_names)
        block_other = tuple(
            (site for site in operand.acts_over() if site not in block_self)
        )
        block = block_self + block_other
        system = self.system.union(operand.system)
        state = self
        # If one of the operators lives in a smaller system, extend it in a way that block
        # is contained on the system of each operator.
        if any(site not in operand.system.sites for site in block_self):
            operand = QutipOperator(
                operand.to_qutip(), system, operand.site_names, operand.prefactor
            )
        if any(site not in self.system.sites for site in block_other):
            state = QutipOperator(
                state.to_qutip(), system, state.site_names, state.prefactor
            )

        op_qo = operand.to_qutip(block)
        rho_qo = state.to_qutip(block)
        return QutipOperator(
            op_qo * rho_qo, names={s: i for i, s in enumerate(block)}, system=system
        )

    def logm(self):
        self.normalize()
        operator = self.operator
        evals, evecs = operator.eigenstates()
        evals[abs(evals) < 1.0e-30] = 1.0e-30
        log_op = sum(
            np.log(e_val) * e_vec * e_vec.dag() for e_val, e_vec in zip(evals, evecs)
        )
        return QutipOperator(log_op, self.system, self.site_names)

    def normalize(self):
        if self._normalized:
            return self
        qoperator = self.operator
        tr_op = qoperator.tr()
        if tr_op != 1:
            qoperator = qoperator / tr_op
        self.operator = qoperator
        self._normalized = True
        return self

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        self.normalize()
        self_pt = super().partial_trace(sites)
        if isinstance(self_pt, ScalarOperator):
            return self_pt

        return QutipDensityOperator(
            self_pt.operator,
            names=self_pt.site_names,
            system=self_pt.system,
            prefactor=self.prefactor,
        )

    def to_qutip(self, block: Optional[Tuple[str]] = None):
        self.normalize()
        qutip_op = super().to_qutip(block)
        return qutip_op
