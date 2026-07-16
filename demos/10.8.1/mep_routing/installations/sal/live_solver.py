"""Live Sal routing-runtime composition for the interactive application."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.routing import RoutingRuntime, RoutingWeightRuntimeContext

from .application import SalSolverSettings


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
