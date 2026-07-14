"""Sal's negotiated-congestion routing policy.

The policy is independent of Pygame and receives live graph/search operations
through a narrow context supplied by the application adapter.
"""

from dataclasses import dataclass

from .route_plan import SalRoutePlan


@dataclass
class SalNegotiatedContext:
    """Application adapters used by Sal's negotiated-congestion strategy."""

    env: object
    route_start_nodes: object
    terminal_node_indices: object
    set_terminal_block_weight: object
    add_route_clearance_weights: object
    add_route_interaction_weights: object
    route_diameter: object
    route_segments_from_path: object
    route_axis_records: object
    run_search: object
    count_crossings: object
    score_routes: object


@dataclass(frozen=True)
class NegotiatedRoutingResult:
    """Best complete route set returned by negotiated congestion routing."""

    success: bool
    routes: object
    total_nodes: int
    attempts: int
    crossing_free: bool


def run_negotiated_congestion(
    room_names,
    pin_node_map,
    global_pins,
    shaft_boundary_nodes,
    shaft_node_idx,
    *,
    route_plan: SalRoutePlan,
    context: SalNegotiatedContext,
    machine_angle,
    bend_cost,
    favour_large=False,
    iterations=20,
    present_penalty=20_000.0,
    history_penalty=4_000.0,
):
    """Route Sal's shaft, kitchen, and small ducts with negotiated congestion.

    This retains the live solver's pin choices, terminal blocking, congestion
    history, and zero-crossing early success policy.  The caller owns elapsed
    time measurement and presentation of the resulting status text.
    """
    nets = list(route_plan.all_routes)
    current_paths = {}
    current_pins = {}
    current_pin_targets = {}
    history_congestion = {}
    node_history_congestion = {}
    best_routes = None
    best_score = float("inf")
    best_total_nodes = 0

    for attempt in range(1, iterations + 1):
        for route_name in nets:
            if route_name == route_plan.shaft_route:
                start_nodes = shaft_boundary_nodes
                target_pins = list(route_plan.large_ports)
            elif route_name == route_plan.kitchen_route:
                start_nodes = context.route_start_nodes(route_plan.kitchen_route)
                if not start_nodes:
                    continue
                shaft_pin = current_pins.get(route_plan.shaft_route, route_plan.large_ports[0])
                target_pins = [route_plan.kitchen_port_for(shaft_pin)]
            else:
                start_nodes = context.route_start_nodes(route_name)
                if not start_nodes:
                    continue
                used_small_pins = [
                    current_pins[name]
                    for name in room_names
                    if name != route_name and name in current_pins
                ]
                target_pins = [pin for pin in route_plan.small_ports if pin not in used_small_pins]
                if not target_pins:
                    target_pins = [route_plan.small_ports[0]]

            current_paths[route_name] = None
            edge_usage, node_usage = _path_usage(current_paths.values())
            edge_weights = _weights_for_route(
                route_name,
                current_paths,
                pin_node_map,
                shaft_node_idx,
                edge_usage,
                node_usage,
                history_congestion,
                node_history_congestion,
                context,
                route_plan,
                favour_large,
                present_penalty,
            )
            path, _, chosen_pin, chosen_target = context.run_search(
                context.env,
                start_nodes,
                target_pins,
                pin_node_map,
                global_pins,
                machine_angle,
                bend_cost,
                edge_weights=edge_weights,
            )
            if path is not None:
                current_paths[route_name] = path
                current_pins[route_name] = chosen_pin
                current_pin_targets[route_name] = chosen_target

        routes, total_nodes = _build_routes(nets, current_paths, current_pins, current_pin_targets, global_pins, context)
        if routes is None:
            continue

        crossings = context.count_crossings(routes)
        score = context.score_routes(routes, crossings)
        if score < best_score:
            best_score = score
            best_routes = routes
            best_total_nodes = total_nodes
        if crossings == 0:
            return NegotiatedRoutingResult(True, routes, total_nodes, attempt, True)

        _update_history(current_paths.values(), history_congestion, node_history_congestion, history_penalty)

    if best_routes is None:
        return NegotiatedRoutingResult(False, None, 0, iterations, False)
    return NegotiatedRoutingResult(True, best_routes, best_total_nodes, iterations, False)


def _weights_for_route(
    route_name,
    current_paths,
    pin_node_map,
    shaft_node_idx,
    edge_usage,
    node_usage,
    history_congestion,
    node_history_congestion,
    context,
    route_plan,
    favour_large,
    present_penalty,
):
    weights = {}
    for terminal_route, terminal_node in context.terminal_node_indices(pin_node_map, shaft_node_idx).items():
        if terminal_route == route_name or terminal_node not in context.env.adj:
            continue
        for neighbour, _, _ in context.env.adj[terminal_node]:
            context.set_terminal_block_weight(weights, terminal_node, neighbour)

    for node, neighbours in context.env.adj.items():
        for neighbour, distance, _ in neighbours:
            edge = (min(node, neighbour), max(node, neighbour))
            if weights.get(edge, 0.0) >= 1e9:
                continue
            congestion = (
                edge_usage.get(edge, 0) * present_penalty
                + history_congestion.get(edge, 0.0)
                + max(node_usage.get(node, 0), node_usage.get(neighbour, 0)) * present_penalty
                + max(node_history_congestion.get(node, 0.0), node_history_congestion.get(neighbour, 0.0))
            )
            if favour_large and route_name in route_plan.large_routes:
                congestion *= 0.35
            weights[edge] = distance + congestion

    context.add_route_clearance_weights(weights, route_name, context.env)
    prior_axes = []
    for other_name, other_path in current_paths.items():
        if other_path is None or other_name == route_name:
            continue
        other_segments = context.route_segments_from_path(other_name, other_path)
        prior_axes.extend(context.route_axis_records(other_name, other_segments))
    context.add_route_interaction_weights(prior_axes, context.route_diameter(route_name), weights, context.env)
    return weights


def _build_routes(nets, paths, pins, targets, global_pins, context):
    routes = []
    total_nodes = 0
    for route_name in nets:
        path = paths.get(route_name)
        if path is None:
            return None, 0
        segments = context.route_segments_from_path(
            route_name,
            path,
            pins.get(route_name),
            global_pins,
            targets.get(route_name),
        )
        routes.append((route_name, segments))
        total_nodes += len(path)
    return routes, total_nodes


def _path_usage(paths):
    edge_usage, node_usage = {}, {}
    for path in paths:
        if path is None:
            continue
        for node in path:
            node_usage[node] = node_usage.get(node, 0) + 1
        for index in range(len(path) - 1):
            edge = (min(path[index], path[index + 1]), max(path[index], path[index + 1]))
            edge_usage[edge] = edge_usage.get(edge, 0) + 1
    return edge_usage, node_usage


def _update_history(paths, edge_history, node_history, penalty):
    edge_usage, node_usage = _path_usage(paths)
    for edge, count in edge_usage.items():
        if count > 1:
            edge_history[edge] = edge_history.get(edge, 0.0) + penalty
    for node, count in node_usage.items():
        if count > 1:
            node_history[node] = node_history.get(node, 0.0) + penalty
