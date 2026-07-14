"""Application-independent orchestration for one Sal routing solve."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

from .orchestration import (
    SalFlowRoutingRequest,
    SalRoutingStrategy,
    coerce_routing_strategy,
    dispatch_flow_strategy,
    is_negotiated_strategy,
    sequential_room_orders,
    should_stop_after_sequential_candidate,
)
from .route_plan import SalRoutePlan


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
    bend_cost: float
    route_plan: SalRoutePlan
    run_small_pin_flow: Callable[..., tuple[bool, Any, str, int]]
    run_two_stage_flow: Callable[..., tuple[bool, Any, str, int]]
    run_negotiated: Callable[..., Any]
    run_sequential: Callable[..., tuple[bool, Any, str, int]]
    count_crossings: Callable[[Any], int]
    score_routes: Callable[[Any, int], float]
    conflict_summary: Callable[[Any], str]
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
        context.bend_cost,
        shaft_weights,
    )
    if shaft_path is None:
        return _result(None, "Blocked: No path to shaft", started, context, 0, excluded_edges)

    other_rooms = context.route_plan.small_routes
    strategy = coerce_routing_strategy(context.routing_strategy)
    flow_result = dispatch_flow_strategy(
        strategy,
        SalFlowRoutingRequest(
            room_names=tuple(other_rooms),
            pin_node_map=pin_node_map,
            global_pins=global_pins,
            shaft_node_idx=shaft_node_idx,
            chosen_exhaust_pin=chosen_exhaust_pin,
            chosen_exhaust_target=chosen_exhaust_target,
            shaft_path=shaft_path,
        ),
        run_small_pin_flow=context.run_small_pin_flow,
        run_two_stage_flow=context.run_two_stage_flow,
    )
    if flow_result is not None:
        success, routes, status, total_nodes = flow_result
        if not success:
            return _result(None, f"Routing Blocked: {status}", started, context, 0, excluded_edges)
        if strategy is SalRoutingStrategy.MIN_COST_FLOW_SMALL_PINS:
            message = f"Success: Min-cost flow small pins ({context.conflict_summary(routes)})"
        else:
            message = f"Success: Two-stage MCMF {status} ({context.conflict_summary(routes)})"
        return _result(routes, message, started, context, total_nodes, excluded_edges)

    if is_negotiated_strategy(strategy):
        negotiated = context.run_negotiated(
            context.route_plan, other_rooms, pin_node_map, global_pins,
            shaft_boundary_nodes, shaft_node_idx, strategy,
        )
        if negotiated.success:
            return _result(
                negotiated.routes,
                f"Success: Routed all (tried {negotiated.attempts} iters, {context.conflict_summary(negotiated.routes)})",
                started,
                context,
                negotiated.total_nodes,
                excluded_edges,
            )
        return _result(None, f"Routing Blocked (tried {negotiated.attempts} iters)", started, context, 0, excluded_edges)

    best_routes = None
    best_score = float("inf")
    best_nodes = 0
    attempts = 0
    for room_order in sequential_room_orders(strategy, other_rooms):
        attempts += 1
        success, routes, _status, total_nodes = context.run_sequential(
            context.route_plan,
            room_order,
            pin_node_map,
            global_pins,
            shaft_node_idx,
            chosen_exhaust_pin,
            chosen_exhaust_target,
            shaft_path,
        )
        if not success:
            continue
        crossings = context.count_crossings(routes)
        score = context.score_routes(routes, crossings)
        if score < best_score:
            best_routes, best_score, best_nodes = routes, score, total_nodes
        if should_stop_after_sequential_candidate(strategy, crossings):
            break
    if best_routes is not None:
        return _result(
            best_routes,
            f"Success: Routed all (tried {attempts} perms, {context.conflict_summary(best_routes)})",
            started,
            context,
            best_nodes,
            excluded_edges,
        )
    return _result(None, f"Routing Blocked (tried {attempts} perms)", started, context, 0, excluded_edges)


def _result(routes, status: str, started: float, context: SalRoutingControllerContext, total_nodes: int, excluded_edges: set) -> SalRoutingResult:
    elapsed_ms = (context.clock() - started) * 1000.0
    return SalRoutingResult(routes, f"{status} in {elapsed_ms:.1f}ms", elapsed_ms, total_nodes, frozenset(excluded_edges))
