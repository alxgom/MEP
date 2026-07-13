"""Application-independent dispatch for the current Sal solver strategies.

The Pygame application owns graph state and solver callbacks.  This module owns
only the mapping from the selected Sal strategy to its routing branch and
candidate room order policy.
"""

from dataclasses import dataclass
from enum import IntEnum
from itertools import permutations


class SalRoutingStrategy(IntEnum):
    """Stable indices for the Sal routing strategy selector."""

    GREEDY_DUAL_SORT = 0
    FIRST_FIT = 1
    BEST_FIT = 2
    NEGOTIATED_CONGESTION = 3
    NEGOTIATED_CONGESTION_FAVOUR_LARGE = 4
    MIN_COST_FLOW_SMALL_PINS = 5
    MIN_COST_FLOW_TWO_STAGE = 6


@dataclass(frozen=True)
class SalFlowRoutingRequest:
    """Inputs shared by Sal's two min-cost-flow strategy branches."""

    room_names: tuple[str, ...]
    pin_node_map: object
    global_pins: object
    shaft_node_idx: int
    chosen_exhaust_pin: str
    chosen_exhaust_target: object
    shaft_path: object


def coerce_routing_strategy(strategy: int | SalRoutingStrategy) -> SalRoutingStrategy:
    """Return a typed strategy or raise a clear error for an unknown selector."""
    return SalRoutingStrategy(strategy)


def is_flow_strategy(strategy: int | SalRoutingStrategy) -> bool:
    """Whether the selected strategy delegates to a Sal min-cost-flow branch."""
    selected = coerce_routing_strategy(strategy)
    return selected in {
        SalRoutingStrategy.MIN_COST_FLOW_SMALL_PINS,
        SalRoutingStrategy.MIN_COST_FLOW_TWO_STAGE,
    }


def is_negotiated_strategy(strategy: int | SalRoutingStrategy) -> bool:
    """Whether the selected strategy uses negotiated-congestion routing."""
    selected = coerce_routing_strategy(strategy)
    return selected in {
        SalRoutingStrategy.NEGOTIATED_CONGESTION,
        SalRoutingStrategy.NEGOTIATED_CONGESTION_FAVOUR_LARGE,
    }


def sequential_room_orders(strategy: int | SalRoutingStrategy, room_names) -> tuple[tuple[str, ...], ...]:
    """Return the deterministic small-duct orders for a sequential strategy.

    Greedy routing retains the existing close-to-far and far-to-close passes.
    First Fit and Best Fit evaluate every permutation.  Flow and negotiated
    strategies own their own routing loop, so they intentionally return none.
    """
    selected = coerce_routing_strategy(strategy)
    rooms = tuple(room_names)
    if selected is SalRoutingStrategy.GREEDY_DUAL_SORT:
        return rooms, tuple(reversed(rooms))
    if selected in {SalRoutingStrategy.FIRST_FIT, SalRoutingStrategy.BEST_FIT}:
        return tuple(permutations(rooms))
    return ()


def should_stop_after_sequential_candidate(strategy: int | SalRoutingStrategy, crossings: int) -> bool:
    """Apply the current First Fit early-exit policy without app-state access."""
    return (
        coerce_routing_strategy(strategy) is SalRoutingStrategy.FIRST_FIT
        and crossings == 0
    )


def dispatch_flow_strategy(
    strategy: int | SalRoutingStrategy,
    request: SalFlowRoutingRequest,
    *,
    run_small_pin_flow,
    run_two_stage_flow,
):
    """Run the selected Sal flow strategy, or return ``None`` for other modes.

    Callbacks remain application adapters: they may rely on the active graph,
    machine placement, and solver backend, none of which belong in this module.
    """
    selected = coerce_routing_strategy(strategy)
    if selected is SalRoutingStrategy.MIN_COST_FLOW_SMALL_PINS:
        return run_small_pin_flow(
            request.room_names,
            request.pin_node_map,
            request.global_pins,
            request.shaft_node_idx,
            request.chosen_exhaust_pin,
            request.chosen_exhaust_target,
            request.shaft_path,
        )
    if selected is SalRoutingStrategy.MIN_COST_FLOW_TWO_STAGE:
        return run_two_stage_flow(
            request.room_names,
            request.pin_node_map,
            request.global_pins,
            request.shaft_path,
        )
    return None
