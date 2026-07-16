"""Live Sal routing-runtime composition for the interactive application."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.routing import RoutingRuntime, RoutingWeightRuntimeContext

from .application import SalApplicationAdapter, SalApplicationHooks, SalSolverSettings
from .route_analysis import SalRouteAnalysis


@dataclass(frozen=True)
class SalLiveRoutingCallbacks:
    shaft_entry_nodes: object
    terminal_nodes: object
    block_terminal_edges: object
    source_start_nodes: object
    weighted_edge_cost: object
    line_graph_direction: object
    route_start_nodes: object
    route_segments_from_path: object
    build_routes_from_paths: object
    count_crossings: object
    score_routes: object
    conflict_summary: object


@dataclass(frozen=True)
class SalLiveRoutingSession:
    installation: object
    machine_spec: object
    workspace: object
    routing_region: object
    rooms: tuple
    walls: tuple
    wall_polygons: tuple
    shafts: tuple
    machine_center: tuple[float, float]
    machine_angle: float
    settings: SalSolverSettings
    search_backend_index: int
    estimate_turns: object

    def application_adapter(self, machine_session, terminals, callbacks):
        hooks = SalApplicationHooks(
            preflight_error=machine_session.preflight_error,
            grid_available=lambda: self.workspace.grid_available,
            machine_pins=lambda: machine_session.pins,
            refresh_graph=machine_session.refresh_graph,
            current_env=lambda: self.workspace.env,
            snap_pins=machine_session.snap_pins,
            shaft_entry_nodes=callbacks.shaft_entry_nodes,
            terminal_nodes=callbacks.terminal_nodes,
            block_terminal_edges=callbacks.block_terminal_edges,
            routing_runtime=lambda policy: self.routing_runtime(self.workspace.env, policy),
            source_start_nodes=callbacks.source_start_nodes,
            weighted_edge_cost=callbacks.weighted_edge_cost,
            line_graph_direction=callbacks.line_graph_direction,
            route_start_nodes=callbacks.route_start_nodes,
            route_segments_from_path=callbacks.route_segments_from_path,
            build_routes_from_paths=callbacks.build_routes_from_paths,
            route_diameter=self.machine_spec.route_diameter_mm,
            count_crossings=callbacks.count_crossings,
            score_routes=callbacks.score_routes,
            conflict_summary=callbacks.conflict_summary,
        )
        return SalApplicationAdapter(
            installation=self.installation,
            terminals=terminals,
            machine_center=self.machine_center,
            machine_angle=self.machine_angle,
            small_diameter=self.machine_spec.small_duct_diameter_mm,
            large_diameter=self.machine_spec.large_duct_diameter_mm,
            settings=self.settings,
            hooks=hooks,
        )

    @property
    def policy(self):
        return self.settings.policy()

    def weight_context(self, env, policy=None):
        policy = policy or self.policy
        return RoutingWeightRuntimeContext(
            edge_list=self.workspace.edge_list,
            edge_coords=self.workspace.edge_coords,
            nodes=env.nodes,
            routing_region=self.routing_region,
            room_polygons=[room.polygon for room in self.rooms],
            walls=self.walls,
            wall_polygons=self.wall_polygons,
            shafts=self.shafts,
            machine_center=self.machine_center,
            machine_angle_deg=self.machine_angle,
            machine_overall_width_mm=self.machine_spec.overall_width_mm,
            machine_body_height_mm=self.machine_spec.body_height_mm,
            buffer_ratio=policy.duct_buffer_ratio,
            shaft_clearance_mm=policy.shaft_clearance_mm,
            machine_soft_margin_mm=policy.machine_clearance_soft_margin_mm,
            crossing_penalty=policy.crossing_penalty,
            clearance_penalty=policy.clearance_penalty,
            block_weight=policy.overlap_block_weight,
            route_diameter=self.machine_spec.route_diameter_mm,
        )

    def routing_runtime(self, env, policy=None):
        policy = policy or self.policy
        return RoutingRuntime(
            env,
            self.weight_context(env, policy),
            self.workspace.static_clearance_cache,
            self.workspace.overlay,
            search_backend=self.installation.search_backends[self.search_backend_index],
            heuristic_mode=policy.heuristic_mode,
            bend_cost=policy.bend_cost,
            estimate_turns_fn=self.estimate_turns,
        )

    def route_analysis(self, shaft_extraction=None, policy=None):
        return SalRouteAnalysis(
            self.machine_spec,
            policy or self.policy,
            self.routing_region,
            shaft_extraction,
        )
