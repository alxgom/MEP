"""Live-graph adapter for Sal's min-cost-flow strategies."""

from dataclasses import dataclass

from mep_routing.routing import (
    RoutingProblem,
    RoutingRequest,
    pin_target_specs,
    solve_pin_flow,
)

from .negotiated import SalNegotiatedContext, run_negotiated_congestion
from .strategies import (
    SalFlowContext,
    run_direct_small_pin_flow,
    run_sequential_routing,
    run_small_flow_stage,
    search_large_route_candidates,
    select_two_stage_routing,
)
from .route_plan import SalRoutePlan
from .policy import SalSolverPolicy
from .prepared import SalPreparedRoutingProblem


@dataclass
class SalFlowRuntime:
    """Application callbacks and values required by Sal flow strategies."""

    env: object
    route_plan: SalRoutePlan
    terminals: object
    small_diameter: int
    large_diameter: int
    policy: SalSolverPolicy
    source_start_nodes: object
    weighted_edge_cost: object
    line_graph_direction: object
    record_edge_weight_overlay: object
    route_start_nodes: object
    route_segments_from_path: object
    build_routes_from_paths: object
    route_axis_records: object
    add_static_clearance_weights: object
    add_machine_clearance_weights: object
    add_route_clearance_weights: object
    add_route_interaction_weights: object
    route_diameter: object
    run_search: object
    count_crossings: object
    score_routes: object
    terminal_node_indices: object = None
    set_terminal_block_weight: object = None

    def run_pin_flow(self, route_names, target_specs_by_route, terminal_points_by_route, edge_weights=None):
        if not route_names:
            return {}, {}, 0.0, 0
        self.record_edge_weight_overlay(edge_weights, self.env)
        starts = {name: self.source_start_nodes(terminal_points_by_route[name]) for name in route_names}
        if any(not target_specs_by_route.get(name) for name in route_names):
            return None, None, float("inf"), 0
        if any(not starts[name] for name in route_names):
            return None, None, 0.0, 0
        problem = RoutingProblem(
            self.env,
            tuple(RoutingRequest(
                name,
                tuple(starts[name]),
                tuple(target_specs_by_route.get(name, ())),
            ) for name in route_names),
            edge_weights if edge_weights is not None else {},
        )
        result = solve_pin_flow(
            problem,
            edge_cost_fn=lambda weights, u, v, distance: self.weighted_edge_cost(weights, u, v, distance),
            direction_fn=self.line_graph_direction,
            bend_penalty=self.policy.bend_cost,
            overlap_block_weight=self.policy.overlap_block_weight,
        )
        if not result.success:
            return None, None, result.objective_cost, result.completed_request_count
        paths = {route.request_key: list(route.route.path) for route in result.routes}
        targets = {route.request_key: route.route.target for route in result.routes}
        return paths, targets, result.objective_cost, result.completed_request_count

    def run_small_pin_flow(self, room_names, pin_node_map, edge_weights=None):
        targets = pin_target_specs(room_names, pin_node_map, self.route_plan.small_ports)
        starts = {name: self.route_start_nodes(name) for name in room_names}
        return self.run_pin_flow(room_names, targets, starts, edge_weights=edge_weights)

    def route_one_pin_flow(self, route_name, target_pin, terminal_point, pin_node_map, edge_weights=None):
        paths, targets, cost, flow = self.run_pin_flow(
            [route_name], {route_name: pin_node_map.get(target_pin, [])},
            {route_name: terminal_point}, edge_weights=edge_weights,
        )
        if flow < 1 or paths is None:
            return None, None, cost
        return paths[route_name], targets[route_name], cost

    def build_small_flow_weights(self, prior_axes, diameter, env):
        weights = {}
        self.add_static_clearance_weights(weights, diameter, env, allow_shaft_entry=False)
        self.add_machine_clearance_weights(weights, diameter, env)
        self.add_route_interaction_weights(prior_axes, diameter, weights, env)
        return weights

    def run_small_stage(self, room_names, pin_node_map, global_pins, prior_axes):
        return run_small_flow_stage(
            room_names, pin_node_map, global_pins, prior_axes,
            small_diameter=self.small_diameter, env=self.env,
            build_weights=self.build_small_flow_weights,
            run_flow=self.run_small_pin_flow,
            build_routes=self.build_routes_from_paths,
        )

    def run_large_search(self, pin_node_map, shaft_nodes, edge_weights=None):
        return search_large_route_candidates(
            pin_node_map, shaft_nodes, env=self.env, terminals=self.terminals,
            route_start_nodes=self.route_start_nodes, route_one_pin_flow=self.route_one_pin_flow,
            route_segments_from_path=self.route_segments_from_path, route_axis_records=self.route_axis_records,
            add_route_clearance_weights=self.add_route_clearance_weights,
            add_route_interaction_weights=self.add_route_interaction_weights,
            route_diameter=self.route_diameter, count_crossings=self.count_crossings,
            score_routes=self.score_routes, route_plan=self.route_plan,
            initial_edge_weights=edge_weights,
        )

    def flow_context(self):
        return SalFlowContext(
            env=self.env, small_diameter=self.small_diameter, large_diameter=self.large_diameter,
            build_routes=self.build_routes_from_paths, route_axis_records=self.route_axis_records,
            run_small_stage=self.run_small_stage, run_large_search=self.run_large_search,
            build_weights=self.build_small_flow_weights, route_plan=self.route_plan,
        )

    def run_direct_small_pin_flow(self, room_names, pin_node_map, global_pins, chosen_pin, chosen_target, shaft_path, *, machine_angle):
        return run_direct_small_pin_flow(
            room_names, pin_node_map, global_pins, chosen_pin, chosen_target, shaft_path,
            route_plan=self.route_plan, env=self.env, machine_angle=machine_angle, bend_cost=self.policy.bend_cost,
            route_start_nodes=self.route_start_nodes, route_segments_from_path=self.route_segments_from_path,
            route_axis_records=self.route_axis_records, add_route_clearance_weights=self.add_route_clearance_weights,
            add_route_interaction_weights=self.add_route_interaction_weights, route_diameter=self.route_diameter,
            run_search=self.run_search, run_small_stage=self.run_small_stage,
        )

    def run_prepared_small_pin_flow(self, prepared: SalPreparedRoutingProblem, *, machine_angle):
        return self.run_direct_small_pin_flow(
            prepared.route_plan.small_routes,
            prepared.pin_node_map,
            prepared.global_pins,
            prepared.chosen_shaft_pin,
            prepared.chosen_shaft_target,
            prepared.shaft_path,
            machine_angle=machine_angle,
        )

    def run_two_stage(self, room_names, pin_node_map, global_pins, shaft_path):
        context = self.flow_context()
        return select_two_stage_routing(
            lambda: context.run_big_first(room_names, pin_node_map, global_pins, shaft_path),
            lambda: context.run_small_first(room_names, pin_node_map, global_pins, shaft_path),
            self.count_crossings, self.score_routes,
        )

    def run_prepared_two_stage(self, prepared: SalPreparedRoutingProblem):
        return self.run_two_stage(
            prepared.route_plan.small_routes,
            prepared.pin_node_map,
            prepared.global_pins,
            prepared.shaft_path,
        )

    def run_prepared_sequential(self, prepared: SalPreparedRoutingProblem, room_order, *, machine_angle):
        return run_sequential_routing(
            room_order,
            prepared.pin_node_map,
            prepared.global_pins,
            prepared.shaft_node_idx,
            prepared.chosen_shaft_pin,
            prepared.chosen_shaft_target,
            prepared.shaft_path,
            route_plan=prepared.route_plan,
            env=self.env,
            machine_angle=machine_angle,
            bend_cost=prepared.policy.bend_cost,
            route_start_nodes=self.route_start_nodes,
            route_segments_from_path=self.route_segments_from_path,
            run_search=self.run_search,
            terminal_node_indices=self.terminal_node_indices,
            set_terminal_block_weight=self.set_terminal_block_weight,
            add_route_clearance_weights=self.add_route_clearance_weights,
            add_route_interaction_weights=self.add_route_interaction_weights,
            route_diameter=self.route_diameter,
            route_axis_records=self.route_axis_records,
        )

    def negotiated_context(self):
        return SalNegotiatedContext(
            env=self.env,
            route_start_nodes=self.route_start_nodes,
            terminal_node_indices=self.terminal_node_indices,
            set_terminal_block_weight=self.set_terminal_block_weight,
            add_route_clearance_weights=self.add_route_clearance_weights,
            add_route_interaction_weights=self.add_route_interaction_weights,
            route_diameter=self.route_diameter,
            route_segments_from_path=self.route_segments_from_path,
            route_axis_records=self.route_axis_records,
            run_search=self.run_search,
            count_crossings=self.count_crossings,
            score_routes=self.score_routes,
        )

    def run_prepared_negotiated(self, prepared: SalPreparedRoutingProblem, favour_large, *, machine_angle):
        return run_negotiated_congestion(
            prepared.route_plan.small_routes,
            prepared.pin_node_map,
            prepared.global_pins,
            prepared.shaft_boundary_nodes,
            prepared.shaft_node_idx,
            route_plan=prepared.route_plan,
            policy=prepared.policy,
            context=self.negotiated_context(),
            machine_angle=machine_angle,
            favour_large=favour_large,
        )
