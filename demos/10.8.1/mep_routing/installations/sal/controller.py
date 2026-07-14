"""Application-independent orchestration for one Sal routing solve."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

from .orchestration import SalRoutingStrategy, coerce_routing_strategy
from .route_plan import SalRoutePlan
from .policy import SalSolverPolicy
from .prepared import SalPreparedRoutingProblem
from .strategy_dispatch import SalStrategyRuntime, solve_prepared_strategy


@dataclass
class SalRoutingControllerContext:
    """Live graph adapters and inputs needed for one routing attempt."""

    preflight_error: Callable[[], str | None]
    grid_available: Callable[[], bool]
    machine_pins: Callable[[], Mapping[str, Any]]
    refresh_graph: Callable[[Mapping[str, Any]], None]
    snap_pins: Callable[[Mapping[str, Any]], Any]
    shaft_entry_nodes: Callable[[], tuple[list[int], int | None]]
    terminal_nodes: Callable[[Any, int | None], Mapping[str, int]]
    block_terminal_edges: Callable[[dict, Mapping[str, int]], Any]
    add_shaft_clearance_weights: Callable[[dict], None]
    run_shaft_search: Callable[[list[int], Any, Mapping[str, Any], float, dict], tuple[Any, Any, str | None, Any]]
    routing_strategy: int | SalRoutingStrategy
    policy: SalSolverPolicy
    route_plan: SalRoutePlan
    strategy_runtime: SalStrategyRuntime
    clock: Callable[[], float] = perf_counter


@dataclass(frozen=True)
class SalRoutingResult:
    routes: Any
    status: str
    elapsed_ms: float
    total_nodes: int
    excluded_overlay_edges: frozenset


def solve_routing(context: SalRoutingControllerContext) -> SalRoutingResult:
    """Run the selected Sal strategy while returning all controller mutation explicitly."""
    started = context.clock()
    excluded_edges = set()
    error = context.preflight_error()
    if error:
        return _result(None, error, started, context, 0, excluded_edges)
    if not context.grid_available():
        return _result(None, "Building grid… press Space to retry", started, context, 0, excluded_edges)

    global_pins = context.machine_pins()
    context.refresh_graph(global_pins)
    pin_node_map = context.snap_pins(global_pins)
    shaft_boundary_nodes, shaft_node_idx = context.shaft_entry_nodes()
    if not pin_node_map or shaft_node_idx is None:
        return _result(None, "Blocked: Missing pins or shaft", started, context, 0, excluded_edges)

    shaft_weights: dict = {}
    excluded_edges.update(context.block_terminal_edges(shaft_weights, context.terminal_nodes(pin_node_map, shaft_node_idx)))
    context.add_shaft_clearance_weights(shaft_weights)
    shaft_path, _cost, chosen_exhaust_pin, chosen_exhaust_target = context.run_shaft_search(
        shaft_boundary_nodes,
        pin_node_map,
        global_pins,
        context.policy.bend_cost,
        shaft_weights,
    )
    if shaft_path is None:
        return _result(None, "Blocked: No path to shaft", started, context, 0, excluded_edges)

    prepared = SalPreparedRoutingProblem(
        route_plan=context.route_plan,
        policy=context.policy,
        global_pins=global_pins,
        pin_node_map=pin_node_map,
        shaft_boundary_nodes=shaft_boundary_nodes,
        shaft_node_idx=shaft_node_idx,
        shaft_path=shaft_path,
        chosen_shaft_pin=chosen_exhaust_pin,
        chosen_shaft_target=chosen_exhaust_target,
    )

    strategy = coerce_routing_strategy(context.routing_strategy)
    outcome = solve_prepared_strategy(strategy, prepared, context.strategy_runtime)
    return _result(
        outcome.routes if outcome.success else None,
        outcome.status,
        started,
        context,
        outcome.total_nodes,
        excluded_edges,
    )


def _result(routes, status: str, started: float, context: SalRoutingControllerContext, total_nodes: int, excluded_edges: set) -> SalRoutingResult:
    elapsed_ms = (context.clock() - started) * 1000.0
    return SalRoutingResult(routes, f"{status} in {elapsed_ms:.1f}ms", elapsed_ms, total_nodes, frozenset(excluded_edges))
