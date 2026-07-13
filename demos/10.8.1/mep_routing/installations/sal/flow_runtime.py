"""Live-graph adapter for Sal's min-cost-flow strategies."""

from dataclasses import dataclass

from mep_routing.routing import (
    build_pin_min_cost_flow_network,
    min_cost_flow,
    small_pin_target_specs,
    trace_flow_path,
)

from .strategies import SalFlowContext, run_direct_small_pin_flow, run_small_flow_stage, search_large_route_candidates, select_two_stage_routing


@dataclass
class SalFlowRuntime:
    """Application callbacks and values required by Sal flow strategies."""

    env: object
    terminals: object
    small_diameter: int
    large_diameter: int
    bend_cost: float
    overlap_block_weight: float
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

    def run_pin_flow(self, route_names, target_specs_by_route, terminal_points_by_route, edge_weights=None):
        if not route_names:
            return {}, {}, 0.0, 0
        self.record_edge_weight_overlay(edge_weights, self.env)
        starts = {name: self.source_start_nodes(terminal_points_by_route[name]) for name in route_names}
        network = build_pin_min_cost_flow_network(
            route_names, target_specs_by_route, starts, self.env.adj,
            lambda u, v, distance: self.weighted_edge_cost(edge_weights, u, v, distance),
            lambda u, v: self.line_graph_direction(self.env, u, v),
            self.bend_cost, self.overlap_block_weight,
        )
        if network is None:
            return None, None, float("inf"), 0
        graph, source, sink, route_nodes = network
        flow, cost = min_cost_flow(graph, source, sink, len(route_names))
        if flow < len(route_names):
            return None, None, cost, flow
        paths, targets = {}, {}
        for name in route_names:
            path, target = trace_flow_path(graph, route_nodes[name], sink)
            if path is None:
                return None, None, cost, flow
            paths[name], targets[name] = path, target
        return paths, targets, cost, flow

    def run_small_pin_flow(self, room_names, pin_node_map, edge_weights=None):
        targets = small_pin_target_specs(room_names, pin_node_map)
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
            score_routes=self.score_routes, initial_edge_weights=edge_weights,
        )

    def flow_context(self):
        return SalFlowContext(
            env=self.env, small_diameter=self.small_diameter, large_diameter=self.large_diameter,
            build_routes=self.build_routes_from_paths, route_axis_records=self.route_axis_records,
            run_small_stage=self.run_small_stage, run_large_search=self.run_large_search,
            build_weights=self.build_small_flow_weights,
        )

    def run_direct_small_pin_flow(self, room_names, pin_node_map, global_pins, chosen_pin, chosen_target, shaft_path, *, machine_angle):
        return run_direct_small_pin_flow(
            room_names, pin_node_map, global_pins, chosen_pin, chosen_target, shaft_path,
            env=self.env, machine_angle=machine_angle, bend_cost=self.bend_cost,
            route_start_nodes=self.route_start_nodes, route_segments_from_path=self.route_segments_from_path,
            route_axis_records=self.route_axis_records, add_route_clearance_weights=self.add_route_clearance_weights,
            add_route_interaction_weights=self.add_route_interaction_weights, route_diameter=self.route_diameter,
            run_search=self.run_search, run_small_stage=self.run_small_stage,
        )

    def run_two_stage(self, room_names, pin_node_map, global_pins, shaft_path):
        context = self.flow_context()
        return select_two_stage_routing(
            lambda: context.run_big_first(room_names, pin_node_map, global_pins, shaft_path),
            lambda: context.run_small_first(room_names, pin_node_map, global_pins, shaft_path),
            self.count_crossings, self.score_routes,
        )
