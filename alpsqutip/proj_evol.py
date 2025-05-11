import logging
from typing import Callable, List

import numpy as np

# from alpsqutip.operators import safe_expm_and_normalize
from qutip import (  # type: ignore[import-untyped]
    entropy_vn,
    fidelity,
    jmat,
    qeye,
    tensor,
)
from qutip.core.qobj import Qobj  # type: ignore[import-untyped]

from alpsqutip.operators import Operator
from alpsqutip.operators.states import safe_exp_and_normalize
from alpsqutip.scalarprod import gram_matrix, operator_components, orthogonalize_basis


def estimate_log_of_partial_trace(K0, local_sigmas, sites):
    return (
        tensor(
            [
                qeye(dim) if i in sites else local_sigmas[i]
                for i, dim in enumerate(K0.dims[0])
            ]
        )
        * K0
    ).ptrace(sites)


def project_k_to_sep(K, maxit=200):
    length = len(K.dims[0])
    phis = 2 * np.random.rand(length, 3) - 1.0
    loc_ops = jmat(0.5)
    local_Ks = [sum((c * op for c, op in zip(phi, loc_ops))) for phi in phis]
    local_sigmas = [safe_exp_and_normalize(-localK) for localK in local_Ks]
    # Initializes with a random state
    for it in range(maxit):
        for i, sigma in enumerate(local_sigmas):
            new_local_K = estimate_log_of_partial_trace(K, local_sigmas, [i])
            local_Ks[i] = 0.3 * local_Ks[i] + 0.7 * new_local_K

        new_local_sigmas = [safe_exp_and_normalize(-localK) for localK in local_Ks]
        min_fid = min(
            fidelity(old, new) for old, new in zip(local_sigmas, new_local_sigmas)
        )
        if min_fid > 0.995:
            logging.info(f"converged after {it} iterations.")
            break
        local_sigmas = new_local_sigmas
    return local_sigmas


def project_operator(
    operator: Operator, basis: List[Operator], sp: Callable
) -> Operator:
    """
    Build the projection of `operator` over the space generated
    by the orthonormal  `basis` of hermitician operators regarding the scalar product `sp`.
    """
    coeffs = (sp(basis_op, operator) for basis_op in basis)
    return sum(basis_op * coeff for coeff, basis_op in zip(coeffs, basis)).simplify()


class ProjectedEvolver:
    """
    Class that implements the projection evolver.

    """

    def __init__(self, op_basis: dict, sp: Callable, K0: Qobj = None, deep: int = 0):
        """
        `op_basis`: the basis of observables that we want to evolve
        `sp`: the scalar product that defines the notion of orthogonality
        `K0`: the initial `K=-log(rho(0))`. Used to build hierarchical basis
        `deep`: the number of elements in the recursive basis, equivalent to
        the order of convergence for short times.
        """
        self.sp = sp
        self.op_basis = op_basis
        self.deep = deep
        self.build_H_tensor(K0, deep)

    def build_H_tensor(self, K0=None, deep=0):
        """
        Build the matrix that evolves the orthogonal components
        of K(t), as well as the orthogonal basis to expand it.
        """
        sp = self.sp
        H = self.op_basis["H"]
        Id = self.op_basis["Id"]

        def rhs(K):
            """
            Computes the commutator with H, and divides by 1j
            """
            return 1j * (K * H - H * K)

        # Build the orthogonal basis
        basis = []
        # Extend the basis with the hierarchical ops
        if K0 is not None and deep > 0:
            basis += [K0]
            for k in range(deep):
                basis.append(rhs(basis[-1]))

        # Add the operators for which we are interested to compute expectation
        # values
        basis += [op for name, op in self.op_basis.items() if name != "Id"]

        # Build an orthogonal basis from basis, and stores in self.

        basis = [op / (sp(op, op)) ** 0.5 for op in basis]

        orth_basis = orthogonalize_basis(basis, self.sp, idop=Id)

        # Check that orth_basis is an orthonormalized basis
        min_ev = min(np.linalg.eigvalsh(gram_matrix(orth_basis, self.sp)))
        assert min_ev > 0.99, f"min ev: {min_ev}"

        self.orth_basis = orth_basis

        # compute the Htensor matrix
        self.Htensor = np.array(
            [[sp(op2, rhs(op1)) for op1 in orth_basis] for op2 in orth_basis]
        ).real

    def build_state_form_orth_components(self, phi):
        """
        reconstruct a global state from the components of
        `K` in the orthogonal basis.
        """
        # Expensive step...
        K = sum((-c) * op for c, op in zip(phi, self.orth_basis))
        return safe_exp_and_normalize(K)

    def evol_k_averages(self, K0, ts) -> dict:
        """
        Evolve the state exp(-K0) and compute
        the expectation values for the observables in
        `self.op_basis` for each time in `ts`
        Returns a dictionary with the time evolution
        of each observable.
        """
        op_basis = self.op_basis
        result: dict = {key: [] for key in op_basis}
        result["entropy"] = []
        phi_t = self.evol_k_orth_components(K0, ts)
        # Expensive step.
        # TODO: Reimplement me in terms of local operators
        for phi in phi_t:
            sigma = self.build_state_form_orth_components(phi)
            for name in result:
                if name == "entropy":
                    result[name].append(entropy_vn(sigma))
                else:
                    result[name].append((sigma * op_basis[name]).tr().real)

        return result

    def evol_k_orth_components(self, k0, ts):
        """
        Compute `phi_t`, a list with the components of K(t),
        regarding self.orth_basis, for each `t` in `ts`,
        provided K(0)=K0.
        """
        h_tensor = self.Htensor
        phi0 = project_operator(k0, self.orth_basis, self.sp).real
        evals, evecs = np.linalg.eig(h_tensor)
        phi0 = np.linalg.inv(evecs).dot(phi0)
        phi_t = np.array(
            [[np.exp(la * t) * c for la, c in zip(evals, phi0)] for t in ts]
        )
        phi_t = np.array([evecs.dot(phi) for phi in phi_t]).real
        return phi_t
