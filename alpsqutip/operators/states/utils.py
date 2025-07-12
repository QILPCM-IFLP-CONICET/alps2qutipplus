"""
Utility functions for alpsqutip.operators.states

"""

from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Union, cast

import numpy as np
from qutip import Qobj, tensor as qutip_tensor

from alpsqutip.operators.arithmetic import SumOperator
from alpsqutip.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from alpsqutip.operators.quadratic import QuadraticFormOperator
from alpsqutip.operators.qutip import QutipOperator
from alpsqutip.operators.states.basic import (
    DensityOperatorMixin,
    ProductDensityOperator,
)
from alpsqutip.operators.states.qutip import QutipDensityOperator
from alpsqutip.qutip_tools.tools import (
    safe_exp_and_normalize as safe_exp_and_normalize_qobj,
)


def acts_over_order(elem):
    elem_acts_over = elem.acts_over()
    if elem_acts_over is None:
        return 0
    return -len(elem_acts_over)


def compute_expectation_values(
    obs: Operator | Iterable[Operator] | Dict[Any, Operator],
    state: Optional[DensityOperatorMixin],
):
    """
    Compute the expectation value of an operator or operators in an iterable object,
    relative to the state `state`.
    """
    if state is None:
        state = ProductDensityOperator({}, 1, obs.system)
    return state.expect(obs)


def collect_blocks_for_expect(obs_objs: Union[Operator, Iterable]) -> List[frozenset]:
    """
    Find the subsystems required to compute the expectation values
    of obs_objs.

    Parameters
    ==========

    obs_objs: Union[Operator, Iterable]
        an object, or a container of objects.

    Result
    ======

    Tuple:
    a list of `frozenset` objects, sorted from the larger to the smaller in size.

    """
    if isinstance(obs_objs, dict):
        return collect_blocks_for_expect(tuple(obs_objs.values()))
    if isinstance(obs_objs, QuadraticFormOperator):
        obs_objs = obs_objs.as_sum_of_products()

    if isinstance(obs_objs, Operator):
        obs_objs = obs_objs.simplify()
        if isinstance(obs_objs, SumOperator):
            return collect_blocks_for_expect(obs_objs.terms)

        acts_over = obs_objs.acts_over()
        if acts_over is None:
            return []
        return [acts_over]
    # tuple or list
    block_set = set()
    for elem in obs_objs:
        block_set.update(collect_blocks_for_expect(elem))
    return sorted(block_set, key=lambda x: -len(x))


def collect_local_states(
    obs_objs: Union[Operator, Iterable], global_state
) -> Dict[frozenset, DensityOperatorMixin]:
    """
    Build a dict of local states required to compute the expectation values of the observable
    or the observables contained in obs_objs.

    Parameters
    ==========

    obs_objs: Union[Operator], Iterable
       an Operator or an iterable containing the operators required to compute the
       required expectation values.

    Return
    ======
    Dict[frozenset, DensityMatrixMixin]

    A dict of local states associated to the sites enumerated in the keys.

    """
    local_states = {}
    block_objts = collect_blocks_for_expect(obs_objs)
    for obj_block in (frozenset(blk) for blk in block_objts):
        if obj_block in local_states:
            continue
        parent_state = global_state
        for block, candidate in sorted(
            local_states.items(),
            key=lambda x: (len(x[0]) if x[0] is not None else 0),
        ):
            if block is not None and obj_block.issubset(block):
                parent_state = candidate
                break
        local_states[obj_block] = parent_state.partial_trace(obj_block)
    return local_states


def compute_operator_expectation_value(
    obs: Operator, sigma_state: Optional[DensityOperatorMixin]
):
    """ """
    if sigma_state is not None:
        return sigma_state.expect(obs)
    if hasattr(obs, "sites_op"):
        obs_sites_op = obs.sites_op
        factors = (
            qutip_op.tr() / qutip_op.dims[0][0] for qutip_op in obs_sites_op.values()
        )
        return reduce(lambda x, y: x * y, factors, obs.prefactor)

    if hasattr(obs, "terms"):
        return sum(
            compute_operator_expectation_value(term, sigma_state) for term in obs.terms
        )

    # assert isinstance(obs, QutipOperator)
    qutip_op = obs.operator
    prefactor = reduce(
        lambda x, y: x * y, (1.0 / d for d in qutip_op.dims[0]), obs.prefactor
    )
    return qutip_op.tr() * prefactor


def k_by_site_from_operator(k: Operator) -> Dict[str, Operator]:
    """
    Maps an operator `k` to a dictionary where keys are site identifiers and
    values are corresponding operators.

    Args:
        k (Operator): The operator to map.

    Returns:
        Dict[str, Operator]: A dictionary mapping site identifiers to operators.

    Raises:
        TypeError: If the operator type is not supported.
        ValueError: If `QutipOperator` acts on multiple sites.
    """
    offset: float | complex
    if isinstance(k, ScalarOperator):
        system = k.system
        site = next(iter(system.dimensions))
        return {site: k.prefactor * system.site_identity(site)}
    if isinstance(k, LocalOperator):
        return {getattr(k, "site"): getattr(k, "operator")}
    if isinstance(k, ProductOperator):
        prefactor = getattr(k, "prefactor")
        if prefactor == 0:
            return {}
        sites_op = getattr(k, "sites_op")
        if len(sites_op) > 1:
            raise ValueError(
                "k must be a sum of one-body operators, but has a term acting on {k.acts_over()}"
            )
        if len(sites_op) == 0:
            system = k.system
            site = next(iter(system.dimensions))
            return {site: prefactor * system.site_identity(site)}
        if prefactor == 1.0:
            return {site: op for site, op in sites_op.items()}
        return {site: op * prefactor for site, op in sites_op.items()}
    if isinstance(k, SumOperator):
        result = {}
        offset = 0
        for term in getattr(k, "terms"):
            if isinstance(term, LocalOperator):
                site = term.site
                result[site] = term.operator
            elif isinstance(term, ScalarOperator):
                offset += term.prefactor
            elif isinstance(term, SumOperator):
                sub_terms = k_by_site_from_operator(term)
                for sub_site, sub_term in sub_terms.items():
                    if sub_site in result:
                        result[sub_site] += sub_term
                    else:
                        result[sub_site] = sub_term
            else:
                raise TypeError(f"term of {type(term)} not allowed.")

        if offset:
            if result:
                offset = offset / len(result)
                result = {site: op - offset for site, op in result.items()}
            else:
                return k_by_site_from_operator(ScalarOperator(offset, k.system))
        return result
    if isinstance(k, QutipOperator):
        acts_over = k.acts_over()
        if acts_over is not None:
            if len(acts_over) == 0:
                return {}
            if len(acts_over) == 1:
                (site,) = acts_over
                return {site: k.to_qutip(tuple())}
        raise ValueError(
            f"Invalid QutipOperator: acts_over={acts_over}. Expected a single act-over site."
        )
    raise TypeError(f"Unsupported operator type: {type(k)}.")


def reduced_state_by_block(
    term: Operator,
    reduced_states_cache: Dict[Optional[frozenset], DensityOperatorMixin],
):
    acts_over = term.acts_over()
    result = reduced_states_cache.get(acts_over, None)
    if result is not None:
        return result
    if acts_over is None:
        return None
    # No cache
    # return reduced_states_cache.get(None, None)

    size = len(acts_over)
    for block in sorted(
        [block for block in reduced_states_cache if block and len(block) > size],
        key=lambda x: len(x),
    ):
        if acts_over.issubset(block):

            result = reduced_states_cache[block]
            if result is not None:
                result = result.partial_trace(acts_over)
            reduced_states_cache[acts_over] = result
            return result
    result = reduced_states_cache.get(None, None)
    if result is not None:
        result = result.partial_trace(acts_over)
    reduced_states_cache[acts_over] = result
    return result


def safe_exp_and_normalize_localop(operator: LocalOperator):
    system = operator.system
    site = operator.site
    loc_rho, log_z = safe_exp_and_normalize_qobj(operator.operator)
    logz = sum(
        (
            np.log(dim)
            for site_factor, dim in system.dimensions.items()
            if site != site_factor
        ),
        log_z,
    )
    local_states = {
        site_factor: (
            loc_rho
            if site == site_factor
            else system.site_identity(site_factor) / system.dimensions[site_factor]
        )
        for site_factor in system.sites
    }
    return (
        ProductDensityOperator(
            local_states=local_states,
            system=system,
            normalize=False,
        ),
        logz,
    )


def safe_exp_and_normalize_sumop(operator: SumOperator):
    logz: float
    operator = operator.simplify()
    if not isinstance(operator, SumOperator):
        return safe_exp_and_normalize(operator)
    terms = operator.terms
    acts_over_terms_or_none = [term.acts_over() for term in terms]
    if any(
        acts_over is None or len(acts_over) > 1 for acts_over in acts_over_terms_or_none
    ):
        return safe_exp_and_normalize_qutip_operator(operator.to_qutip_operator())

    acts_over_terms: List[frozenset] = cast(List[frozenset], acts_over_terms_or_none)

    system = operator.system
    local_generators: Dict[str, Qobj] = dict()
    logz = 0
    for acts_over, term in zip(acts_over_terms, terms):
        if len(acts_over) == 0:
            logz += np.real(term.prefactor)
            continue
        site = next(iter(acts_over))
        op_qutip = term.to_qutip((site,))
        if site in local_generators:
            local_generators[site] = local_generators[site] + op_qutip
        else:
            local_generators[site] = op_qutip

    local_states = {}
    for site, factor_qutip in local_generators.items():
        local_rho, local_f = safe_exp_and_normalize_qobj(factor_qutip)
        local_states[site] = local_rho
        logz += local_f
    for site in system.sites:
        if site not in local_states:
            dim = system.dimensions[site]
            logz += np.log(dim)
            local_states[site] = system.site_identity(site) / dim

    return (
        ProductDensityOperator(
            local_states=local_states,
            system=system,
            normalize=False,
        ),
        logz,
    )


def safe_exp_and_normalize_qutip_operator(operator):

    system = operator.system
    if isinstance(operator, ScalarOperator):
        ln_z = sum((np.log(dim) for dim in system.dimensions.values()))
        return (ScalarOperator(np.exp(-ln_z), system), ln_z + operator.prefactor)

    site_names = operator.site_names
    block = tuple(sorted(site_names, key=lambda x: site_names[x]))
    rho_qutip, logz = safe_exp_and_normalize_qobj(
        operator.operator * operator.prefactor
    )
    rest = tuple(sorted(site for site in system.sites if site not in block))
    operator = qutip_tensor(
        rho_qutip,
        *(system.site_identity(site) / system.dimensions[site] for site in rest),
    )
    logz = logz + sum(np.log(system.dimensions[site]) for site in rest)
    return (
        QutipDensityOperator(
            operator,
            names={site: pos for pos, site in enumerate(block + rest)},
            system=system,
        ),
        logz,
    )


def safe_exp_and_normalize(operator):
    """
    Compute the decomposition of exp(operator) as rho*exp(f)
    with f = Tr[exp(operator)], for operator a Qutip operator.

    operator: Operator | Qobj

    result: Tuple[Operator|Qobj, float]
         (exp(operator)/f , f)

    """

    if isinstance(operator, ScalarOperator):
        system = operator.system
        ln_z = sum((np.log(dim) for dim in system.dimensions.values()))
        return (ScalarOperator(np.exp(-ln_z), system), ln_z + operator.prefactor)
    if isinstance(operator, LocalOperator):
        return safe_exp_and_normalize_localop(operator)
    if isinstance(operator, SumOperator):
        return safe_exp_and_normalize_sumop(operator)
    if isinstance(operator, QutipOperator):
        return safe_exp_and_normalize_qutip_operator(operator)
    if isinstance(operator, Operator):
        return safe_exp_and_normalize_qutip_operator(operator.to_qutip_operator())

    # assume Qobj or any other class with a compatible interface.
    return safe_exp_and_normalize_qobj(operator)
