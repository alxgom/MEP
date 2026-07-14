"""Prepared-problem dispatch for Sal routing strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .orchestration import (
    SalRoutingStrategy,
    coerce_routing_strategy,
    sequential_room_orders,
    should_stop_after_sequential_candidate,
)
from .prepared import SalPreparedRoutingProblem


@dataclass(frozen=True)
class SalStrategyOutcome:
    success: bool
    routes: Any = None
    status: str = ""
    total_nodes: int = 0


@dataclass
class SalStrategyRuntime:
    """Algorithm adapters and scoring operations used by Sal dispatch."""

    run_small_pin_flow: Callable[[SalPreparedRoutingProblem], tuple]
    run_two_stage_flow: Callable[[SalPreparedRoutingProblem], tuple]
    run_negotiated: Callable[[SalPreparedRoutingProblem, bool], Any]
    run_sequential: Callable[[SalPreparedRoutingProblem, tuple[str, ...]], tuple]
    count_crossings: Callable[[Any], int]
    score_routes: Callable[[Any, int], float]
    conflict_summary: Callable[[Any], str]


def solve_prepared_strategy(
    strategy: int | SalRoutingStrategy,
    prepared: SalPreparedRoutingProblem,
    runtime: SalStrategyRuntime,
) -> SalStrategyOutcome:
    """Execute one Sal strategy and preserve its established status policy."""
    selected = coerce_routing_strategy(strategy)

    if selected is SalRoutingStrategy.MIN_COST_FLOW_SMALL_PINS:
        success, routes, status, total_nodes = runtime.run_small_pin_flow(prepared)
        if not success:
            return SalStrategyOutcome(False, status=f"Routing Blocked: {status}")
        return SalStrategyOutcome(
            True,
            routes,
            f"Success: Min-cost flow small pins ({runtime.conflict_summary(routes)})",
            total_nodes,
        )

    if selected is SalRoutingStrategy.MIN_COST_FLOW_TWO_STAGE:
        success, routes, status, total_nodes = runtime.run_two_stage_flow(prepared)
        if not success:
            return SalStrategyOutcome(False, status=f"Routing Blocked: {status}")
        return SalStrategyOutcome(
            True,
            routes,
            f"Success: Two-stage MCMF {status} ({runtime.conflict_summary(routes)})",
            total_nodes,
        )

    if selected in {
        SalRoutingStrategy.NEGOTIATED_CONGESTION,
        SalRoutingStrategy.NEGOTIATED_CONGESTION_FAVOUR_LARGE,
    }:
        negotiated = runtime.run_negotiated(
            prepared,
            selected is SalRoutingStrategy.NEGOTIATED_CONGESTION_FAVOUR_LARGE,
        )
        if negotiated.success:
            return SalStrategyOutcome(
                True,
                negotiated.routes,
                f"Success: Routed all (tried {negotiated.attempts} iters, {runtime.conflict_summary(negotiated.routes)})",
                negotiated.total_nodes,
            )
        return SalStrategyOutcome(
            False,
            status=f"Routing Blocked (tried {negotiated.attempts} iters)",
        )

    best_routes = None
    best_score = float("inf")
    best_nodes = 0
    attempts = 0
    for room_order in sequential_room_orders(selected, prepared.route_plan.small_routes):
        attempts += 1
        success, routes, _status, total_nodes = runtime.run_sequential(
            prepared,
            room_order,
        )
        if not success:
            continue
        crossings = runtime.count_crossings(routes)
        score = runtime.score_routes(routes, crossings)
        if score < best_score:
            best_routes, best_score, best_nodes = routes, score, total_nodes
        if should_stop_after_sequential_candidate(selected, crossings):
            break

    if best_routes is not None:
        return SalStrategyOutcome(
            True,
            best_routes,
            f"Success: Routed all (tried {attempts} perms, {runtime.conflict_summary(best_routes)})",
            best_nodes,
        )
    return SalStrategyOutcome(
        False,
        status=f"Routing Blocked (tried {attempts} perms)",
    )
