"""Concrete bridge between the interactive application and Sal routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .controller import SalRoutingControllerContext, SalRoutingResult, solve_routing
from .flow_runtime import SalFlowRuntime
from .policy import SalSolverPolicy
from .strategy_dispatch import SalStrategyRuntime


@dataclass(frozen=True)
class SalSolverSettings:
    bend_cost: float
    crossing_penalty_multiplier: float
    duct_buffer_ratio: float
    shaft_clearance_mm: float
    machine_clearance_soft_margin_mm: float
    overlap_block_weight: float
    min_piece_factor: float
    heuristic_mode: int
    routing_strategy: int

    def policy(self) -> SalSolverPolicy:
        return SalSolverPolicy(
            bend_cost=self.bend_cost,
            crossing_penalty_multiplier=self.crossing_penalty_multiplier,
            duct_buffer_ratio=self.duct_buffer_ratio,
            shaft_clearance_mm=self.shaft_clearance_mm,
            machine_clearance_soft_margin_mm=self.machine_clearance_soft_margin_mm,
            overlap_block_weight=self.overlap_block_weight,
            min_piece_factor=self.min_piece_factor,
            heuristic_mode=self.heuristic_mode,
        )


@dataclass
class SalApplicationHooks:
    """Graph-session operations supplied by the interactive application."""

    preflight_error: Callable[[], str | None]
    grid_available: Callable[[], bool]
    machine_pins: Callable[[], Any]
    refresh_graph: Callable[[Any], None]
    current_env: Callable[[], Any]
    snap_pins: Callable[[Any], Any]
    shaft_entry_nodes: Callable[[], Any]
    terminal_nodes: Callable[[Any, int | None, str], Any]
    block_terminal_edges: Callable[[dict, Any, float], Any]
    routing_runtime: Callable[[SalSolverPolicy], Any]
    source_start_nodes: Callable[[Any], Any]
    weighted_edge_cost: Callable[..., Any]
    line_graph_direction: Callable[..., Any]
    route_start_nodes: Callable[[str], Any]
    route_segments_from_path: Callable[..., Any]
    build_routes_from_paths: Callable[..., Any]
    route_diameter: Callable[[str], float]
    count_crossings: Callable[[Any], int]
    score_routes: Callable[[Any, int, SalSolverPolicy], float]
    conflict_summary: Callable[[Any], str]


@dataclass
class SalApplicationAdapter:
    installation: Any
    terminals: Any
    machine_center: tuple[float, float]
    machine_angle: float
    small_diameter: int
    large_diameter: int
    settings: SalSolverSettings
    hooks: SalApplicationHooks

    def solve(self) -> SalRoutingResult:
        policy = self.settings.policy()
        route_plan = self.installation.build_route_plan(self.terminals, self.machine_center)
        routing_runtime = None
        flow_runtime = None

        def get_routing_runtime():
            nonlocal routing_runtime
            if routing_runtime is None:
                routing_runtime = self.hooks.routing_runtime(policy)
            return routing_runtime

        def get_flow_runtime():
            nonlocal flow_runtime
            if flow_runtime is None:
                runtime = get_routing_runtime()
                route_segments = lambda *args: self.hooks.route_segments_from_path(route_plan, *args)
                flow_runtime = SalFlowRuntime(
                    env=self.hooks.current_env(),
                    route_plan=route_plan,
                    terminals=self.terminals,
                    small_diameter=self.small_diameter,
                    large_diameter=self.large_diameter,
                    policy=policy,
                    source_start_nodes=self.hooks.source_start_nodes,
                    weighted_edge_cost=self.hooks.weighted_edge_cost,
                    line_graph_direction=self.hooks.line_graph_direction,
                    record_edge_weight_overlay=lambda weights, _env: runtime.record_edge_weight_overlay(weights),
                    route_start_nodes=self.hooks.route_start_nodes,
                    route_segments_from_path=route_segments,
                    build_routes_from_paths=lambda *args: self.hooks.build_routes_from_paths(route_plan, *args),
                    route_axis_records=runtime.route_axis_records,
                    add_static_clearance_weights=lambda weights, diameter, _env, **kwargs: runtime.add_static_clearance_weights(
                        weights, diameter, **kwargs,
                    ),
                    add_machine_clearance_weights=lambda weights, diameter, _env: runtime.add_machine_clearance_weights(
                        weights, diameter,
                    ),
                    add_route_clearance_weights=lambda weights, name, _env: runtime.add_route_clearance_weights(
                        weights, name, shaft_route_name=route_plan.shaft_route,
                    ),
                    add_route_interaction_weights=lambda axes, diameter, weights, _env: runtime.add_route_interaction_weights(
                        axes, diameter, weights,
                    ),
                    route_diameter=self.hooks.route_diameter,
                    run_search=lambda _env, starts, pins, pin_map, _global_pins, _angle, bend, **kwargs: runtime.run_super_sink_search(
                        starts, pins, pin_map, bend_cost=bend, **kwargs,
                    ),
                    count_crossings=self.hooks.count_crossings,
                    score_routes=lambda routes, crossings: self.hooks.score_routes(routes, crossings, policy),
                    terminal_node_indices=lambda pin_map, shaft_idx: self.hooks.terminal_nodes(
                        pin_map, shaft_idx, route_plan.shaft_route,
                    ),
                    set_terminal_block_weight=runtime.set_terminal_block_weight,
                )
            return flow_runtime

        strategy_runtime = SalStrategyRuntime(
            run_small_pin_flow=lambda prepared: get_flow_runtime().run_prepared_small_pin_flow(
                prepared, machine_angle=self.machine_angle,
            ),
            run_two_stage_flow=lambda prepared: get_flow_runtime().run_prepared_two_stage(prepared),
            run_negotiated=lambda prepared, favour_large: get_flow_runtime().run_prepared_negotiated(
                prepared, favour_large, machine_angle=self.machine_angle,
            ),
            run_sequential=lambda prepared, room_order: get_flow_runtime().run_prepared_sequential(
                prepared, room_order, machine_angle=self.machine_angle,
            ),
            count_crossings=self.hooks.count_crossings,
            score_routes=lambda routes, crossings: self.hooks.score_routes(routes, crossings, policy),
            conflict_summary=self.hooks.conflict_summary,
        )

        context = SalRoutingControllerContext(
            preflight_error=self.hooks.preflight_error,
            grid_available=self.hooks.grid_available,
            machine_pins=self.hooks.machine_pins,
            refresh_graph=self.hooks.refresh_graph,
            snap_pins=self.hooks.snap_pins,
            shaft_entry_nodes=self.hooks.shaft_entry_nodes,
            terminal_nodes=lambda pin_map, shaft_idx: self.hooks.terminal_nodes(
                pin_map, shaft_idx, route_plan.shaft_route,
            ),
            block_terminal_edges=lambda weights, terminal_map: self.hooks.block_terminal_edges(
                weights, terminal_map, policy.overlap_block_weight,
            ),
            add_shaft_clearance_weights=lambda weights: get_routing_runtime().add_route_clearance_weights(
                weights, route_plan.shaft_route, shaft_route_name=route_plan.shaft_route,
            ),
            run_shaft_search=lambda boundary, pin_map, global_pins, bend, weights: get_routing_runtime().run_super_sink_search(
                boundary, list(route_plan.large_ports), pin_map, bend_cost=bend, edge_weights=weights,
            ),
            routing_strategy=self.settings.routing_strategy,
            policy=policy,
            route_plan=route_plan,
            strategy_runtime=strategy_runtime,
        )
        return solve_routing(context)
