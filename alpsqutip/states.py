from functools import reduce

from alpsqutip.model import NBodyOperator, ProductOperator


class QuantumState:
    def expect(self, op):
        if isinstance(op, (int, float, complex)):
            return op
        if isinstance(op, NBodyOperator):
            return sum([self.expect(t) for t in op.terms])
        if isinstance(op, ProductOperator):
            return self._compute_expect_over_product_operator(op)
        raise NotImplementedError


class ProductQuantumState(QuantumState, ProductOperator):
    def __init__(self, local_states, system, weight=1, normalize_local_states=True):
        super(ProductQuantumState, self).__init__(
            sites_op=local_states, prefactor=1, system=system
        )
        self.weight = weight
        # Normalize local states and adjust the prefactor
        local_states = self.sites_op
        normalization = reduce(
            lambda x, y: x * y,
            (d for s, d in system.dimensions.items() if s not in local_states),
            1.0,
        )

        if normalize_local_states:
            for s, op in local_states.items():
                local_trace = op.tr()
                if local_trace <= 0:
                    raise ValueError(f"Non-positive trace on site {s}")
                local_states[s] = op / local_trace

        self.prefactor = weight / normalization

    def _compute_expect_over_product_operator(self, op):
        dimensions = self.system.dimensions
        site_ops = op.sites_op
        site_states = self.sites_op
        factors = [
            (site_ops[s] * site_states[s]).tr()
            if s in site_states
            else site_ops[s].tr() / dimensions[s]
            for s in site_ops
        ]
        return reduce(lambda x, y: x * y, factors, op.prefactor)

    def partial_trace(self, sites):
        sites_op = self.sites_op
        sites = [s for s in sites if s in self.system.sites]
        system = self.system.subsystem(sites)
        sites_op = {s: sites_op[s] for s in sites if s in sites_op}
        return ProductQuantumState(
            sites_op, system=system, weight=self.weight, normalize_local_states=False
        )

    def __add__(self, op):
        system = self.system
        if isinstance(op, (int, float)):
            if op == 0:
                return self
            if op > 0:
                rho = ProductQuantumState({}, system, weight=op)
                return MixedQuantumState([self, rho])
        elif isinstance(op, ProductQuantumState):
            if op.prefactor == 0:
                return self
            return MixedQuantumState([self, op])
        elif isinstance(op, MixedQuantumState):
            return MixedQuantumState([self] + op.terms)
        return super(ProductQuantumState, self).__add__(op)

    def __mul__(self, op):
        system = self.system
        if isinstance(op, (int, float)) and op >= 0:
            rho = ProductQuantumState(
                self.sites_op.copy(), system, weight=op * self.weight
            )
            return rho

        return super(ProductQuantumState, self).__mul__(op)


class MixedQuantumState(QuantumState, NBodyOperator):
    def __init__(self, terms):
        super(MixedQuantumState, self).__init__(terms)

    def __add__(self, op):
        system = self.system
        if isinstance(op, (int, float)):
            if op > 0:
                rho = ProductQuantumState({}, system, weight=op)
                return MixedQuantumState(self.terms + [rho])
            elif op == 0:
                return self
        elif isinstance(op, ProductQuantumState):
            if op.prefactor == 0:
                return self
            return MixedQuantumState(self.terms + [op])
        elif isinstance(op, MixedQuantumState):
            return MixedQuantumState(self.terms + op.terms)
        return super(MixedQuantumState, self).__add__(op)

    def __mul__(self, op):
        if isinstance(op, (int, float)) and op >= 0:
            return MixedQuantumState([t * op for t in self.terms])

        return super(MixedQuantumState, self).__mul__(op)

    def _compute_expect_over_product_operator(self, op):
        return sum(
            t.weight * t._compute_expect_over_product_operator(op) for t in self.terms
        )

    def partial_trace(self, sites: list):
        return sum(t.partial_trace(sites) for t in self.terms)
